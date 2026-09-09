# -*- coding: utf-8 -*-
"""Tests for nn_mge_jax module."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from dolphin.util import enable_jax_x64

# has to happen before JAX is imported, hence the import order below
enable_jax_x64()

import jax
from jax import numpy as jnp

import jaxtronomy.ImSim.de_lens as de_lens
import jaxtronomy.ImSim.MultiBand.multi_linear as multi_linear
import jaxtronomy.ImSim.MultiBand.single_band_multi_model as single_band_multi_model
import jaxtronomy.Workflow.fitting_sequence as fitting_sequence
from jaxtronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from jaxtronomy.Util.class_creator import create_im_sim

from dolphin.processor.core import Processor
from dolphin.processor.nn_mge import (
    NonNegativeMGESingleBandMultiModel as LenstronomyNonNegativeMGE,
)
from dolphin.processor.nn_mge import get_param_bounded_WLS as get_param_bounded_WLS_np
from dolphin.processor.nn_mge import nn_mge_solver
from dolphin.processor.nn_mge_jax import (
    NonNegativeMGESingleBandMultiModel,
    get_param_bounded_WLS,
)

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"

_NUM_BOUNDED = 8
_NUM_FREE = 4


def _get_kwargs_model(config):
    """Get `kwargs_model` with the shapelet order set at profile initialization.

    JAXtronomy's `ShapeletSetStatic` needs `n_max` at initialization to be usable with
    the linear solver, and `FittingSequence.likelihood_class` moves it there from the
    fixed keyword arguments. The image models built here are not built by a fitting
    sequence, so the same move is made by hand.

    :param config: config to create the model keyword arguments from
    :type config: `dolphin.processor.config.ModelConfig`
    :return: model keyword arguments
    :rtype: `dict`
    """
    kwargs_model = config.get_kwargs_model()
    source_light_model_list = kwargs_model["source_light_model_list"]

    if "SHAPELETS" in source_light_model_list:
        fixed = config.get_kwargs_params()["source_model"][2]
        kwargs_model["source_light_profile_kwargs_list"] = [
            {"n_max": fixed[i]["n_max"]} if model == "SHAPELETS" else {}
            for i, model in enumerate(source_light_model_list)
        ]

    return kwargs_model


def _weighted_chi2(A, C_D_inv, d, param):
    """The objective the bounded solver minimizes.

    Two active set implementations can pin different amplitudes to the bound when
    several of them sit on it, so the objective is the invariant to compare on.

    :param A: response matrix, Nd x Ns
    :type A: `numpy.ndarray`
    :param C_D_inv: inverse data variance, 1d of length Nd
    :type C_D_inv: `numpy.ndarray`
    :param d: data array, 1d of length Nd
    :type d: `numpy.ndarray`
    :param param: linear parameters, 1d of length Ns
    :type param: `numpy.ndarray`
    :return: weighted sum of squared residuals
    :rtype: `float`
    """
    return float(np.sum(C_D_inv * (A.dot(param) - d) ** 2))


class TestGetParamBoundedWLS(object):
    def setup_method(self):
        rng = np.random.default_rng(42)

        self.num_param = _NUM_FREE + _NUM_BOUNDED
        self.A = rng.normal(size=(500, self.num_param))
        self.C_D_inv = np.full(500, 4.0)
        self.bounded = np.concatenate(
            [np.zeros(_NUM_FREE, dtype=bool), np.ones(_NUM_BOUNDED, dtype=bool)]
        )
        self.lower_bounds = np.where(self.bounded, 0.0, -np.inf)

        # a solution that is already feasible, and one that is not
        self.param_positive = rng.uniform(1.0, 2.0, size=self.num_param)
        self.param_mixed = np.concatenate(
            [rng.uniform(1.0, 2.0, size=_NUM_FREE), rng.uniform(-2.0, -1.0, size=4)]
            + [rng.uniform(1.0, 2.0, size=_NUM_BOUNDED - 4)]
        )
        self.d_positive = self.A.dot(self.param_positive) + rng.normal(
            scale=0.01, size=500
        )
        self.d_mixed = self.A.dot(self.param_mixed) + rng.normal(scale=0.01, size=500)

    def test_agrees_with_unconstrained_when_feasible(self):
        """Test that a feasible problem is solved like the unconstrained solver."""
        param, _, model = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_positive, self.bounded, inv_bool=False
        )
        param_unconstrained, _, _ = de_lens.get_param_WLS(
            self.A, self.C_D_inv, self.d_positive, inv_bool=False
        )

        npt.assert_allclose(param, param_unconstrained, rtol=1e-8)
        npt.assert_allclose(model, self.A.dot(np.asarray(param)), rtol=1e-8)

    def test_enforces_lower_bounds(self):
        """Test that the bounded parameters are non-negative when the unconstrained
        solution is not."""
        param_unconstrained, _, _ = de_lens.get_param_WLS(
            self.A, self.C_D_inv, self.d_mixed, inv_bool=False
        )
        # guard the premise of the test
        assert np.any(np.asarray(param_unconstrained)[_NUM_FREE:] < 0)

        param, _, _ = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_mixed, self.bounded, inv_bool=False
        )

        assert np.all(np.asarray(param)[_NUM_FREE:] >= 0)

    def test_matches_lenstronomy_solver(self):
        """Test that the JAX active set reproduces the `scipy.optimize.lsq_linear`
        solution the lenstronomy implementation computes."""
        param_np, _, _ = get_param_bounded_WLS_np(
            self.A, self.C_D_inv, self.d_mixed, self.lower_bounds, inv_bool=False
        )
        param, _, _ = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_mixed, self.bounded, inv_bool=False
        )
        param = np.asarray(param)

        npt.assert_allclose(param, param_np, atol=1e-8)
        npt.assert_allclose(
            _weighted_chi2(self.A, self.C_D_inv, self.d_mixed, param),
            _weighted_chi2(self.A, self.C_D_inv, self.d_mixed, param_np),
            rtol=1e-10,
        )

    def test_no_bounded_parameters(self):
        """Test that an all-free problem falls through to the unconstrained solution."""
        param, _, _ = get_param_bounded_WLS(
            self.A,
            self.C_D_inv,
            self.d_mixed,
            np.zeros(self.num_param, dtype=bool),
            inv_bool=False,
        )
        param_unconstrained, _, _ = de_lens.get_param_WLS(
            self.A, self.C_D_inv, self.d_mixed, inv_bool=False
        )

        npt.assert_allclose(param, param_unconstrained, rtol=1e-8)

    def test_all_bounded_parameters(self):
        """Test the pure non-negative least squares case."""
        param, _, _ = get_param_bounded_WLS(
            self.A,
            self.C_D_inv,
            self.d_mixed,
            np.ones(self.num_param, dtype=bool),
            inv_bool=False,
        )
        param_np, _, _ = get_param_bounded_WLS_np(
            self.A,
            self.C_D_inv,
            self.d_mixed,
            np.zeros(self.num_param),
            inv_bool=False,
        )

        assert np.all(np.asarray(param) >= 0)
        npt.assert_allclose(param, param_np, atol=1e-8)

    def test_is_scale_invariant(self):
        """Test that the active set does not depend on the units of the data.

        The amplitudes and the data are in instrumental units, so the normal equations
        can be scaled arbitrarily and the tolerances of the active set method have to be
        relative to that scale.
        """
        param, _, _ = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_mixed, self.bounded, inv_bool=False
        )

        for scale in [1e-8, 1e8]:
            param_scaled, _, _ = get_param_bounded_WLS(
                self.A,
                self.C_D_inv,
                self.d_mixed * scale,
                self.bounded,
                inv_bool=False,
            )
            npt.assert_allclose(np.asarray(param_scaled) / scale, param, rtol=1e-8)

    def test_ill_conditioned_returns_zeros(self):
        """Test that a singular normal matrix gives the zero solution, as
        `de_lens.get_param_WLS` does."""
        A = np.copy(self.A)
        A[:, -1] = A[:, -2]

        # the bounded solver degenerates exactly where the unconstrained one does, so
        # this also guards the premise that the problem is singular enough to trip the
        # shared conditioning check
        param_unconstrained, _, _ = de_lens.get_param_WLS(
            A, self.C_D_inv, self.d_mixed, inv_bool=False
        )
        npt.assert_array_equal(param_unconstrained, np.zeros(self.num_param))

        param, cov_param, model = get_param_bounded_WLS(
            A, self.C_D_inv, self.d_mixed, self.bounded, inv_bool=True
        )

        npt.assert_array_equal(param, np.zeros(self.num_param))
        npt.assert_array_equal(cov_param, np.zeros((self.num_param,) * 2))
        npt.assert_array_equal(model, np.zeros(500))

    def test_covariance_matrix(self):
        """Test that the unconstrained covariance matrix is returned when asked for."""
        _, cov_param, _ = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_mixed, self.bounded, inv_bool=True
        )
        _, cov_param_np, _ = get_param_bounded_WLS_np(
            self.A, self.C_D_inv, self.d_mixed, self.lower_bounds, inv_bool=True
        )

        npt.assert_allclose(cov_param, cov_param_np, rtol=1e-8)

        _, cov_param, _ = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_mixed, self.bounded, inv_bool=False
        )
        assert cov_param is None

    def test_is_differentiable(self):
        """Test that the solution can be differentiated with respect to the response
        matrix, which the gradient descent recipes need."""
        A = jnp.asarray(self.A)
        C_D_inv = jnp.asarray(self.C_D_inv)
        d = jnp.asarray(self.d_mixed)
        scaling = jnp.linspace(1.0, 2.0, self.num_param)

        def objective(factor):
            param, _, _ = get_param_bounded_WLS(
                A * (1.0 + factor * scaling), C_D_inv, d, self.bounded, inv_bool=False
            )
            return jnp.sum(param**2)

        gradient = float(jax.grad(objective)(0.3))
        step = 1e-5
        finite_difference = float(
            (objective(0.3 + step) - objective(0.3 - step)) / (2 * step)
        )

        assert np.isfinite(gradient)
        npt.assert_allclose(gradient, finite_difference, rtol=1e-4)


class TestNonNegativeMGESingleBandMultiModel(object):
    def setup_method(self):
        self.processor = Processor(_TEST_IO_DIR)
        self.config = self.processor.get_lens_config("lens_system2_mge")
        self.kwargs_data_joint = self.processor.get_kwargs_data_joint(
            "lens_system2_mge"
        )

        self.stock_model = self._get_image_model(self.config, SingleBandMultiModel)
        self.nn_model = self._get_image_model(
            self.config, NonNegativeMGESingleBandMultiModel
        )

        # the MGE amplitudes are the last 20 of the 35 linear parameters
        self.mge_slice = slice(15, 35)

    def _get_image_model(self, config, image_model_class, **kwargs):
        """Create the image model for the first band of a given config.

        :param config: config to create the image model from
        :type config: `dolphin.processor.config.ModelConfig`
        :param image_model_class: image model class to instantiate
        :type image_model_class: `class`
        :return: image model for the first band
        :rtype: `image_model_class`
        """
        return image_model_class(
            self.kwargs_data_joint["multi_band_list"],
            _get_kwargs_model(config),
            likelihood_mask_list=config.get_masks(),
            band_index=0,
            **kwargs,
        )

    def _get_kwargs(self, config=None):
        """Get a fresh copy of the initial model keyword arguments.

        :param config: config to take the initial values from
        :type config: `dolphin.processor.config.ModelConfig` or `None`
        :return: lens, source, lens light, and point source keyword arguments
        :rtype: `dict`
        """
        kwargs_params = (self.config if config is None else config).get_kwargs_params()

        return {
            key: [deepcopy(kwargs) for kwargs in kwargs_params[model][0]]
            for key, model in [
                ("kwargs_lens", "lens_model"),
                ("kwargs_source", "source_model"),
                ("kwargs_lens_light", "lens_light_model"),
                ("kwargs_ps", "point_source_model"),
            ]
        }

    def test_image_linear_solve_is_non_negative(self):
        """Test that the MGE amplitudes solved for are non-negative."""
        _, _, _, param_stock = self.stock_model.image_linear_solve(**self._get_kwargs())
        # guard the premise of the test
        assert np.any(np.asarray(param_stock)[self.mge_slice] < 0)

        _, _, _, param = self.nn_model.image_linear_solve(**self._get_kwargs())

        assert np.all(np.asarray(param)[self.mge_slice] >= 0)

    def test_agrees_with_lenstronomy_solver(self):
        """Test that the JAX solver reproduces the lenstronomy one on the same model."""
        lenstronomy_model = LenstronomyNonNegativeMGE(
            self.kwargs_data_joint["multi_band_list"],
            self.config.get_kwargs_model(),
            likelihood_mask_list=self.config.get_masks(),
            band_index=0,
        )

        logL, param = self.nn_model.likelihood_data_given_model(**self._get_kwargs())
        logL_lenstronomy, param_lenstronomy = (
            lenstronomy_model.likelihood_data_given_model(**self._get_kwargs())
        )

        # the two active set implementations can pin different amplitudes to the bound
        # when several of them sit on it, so the fit rather than the solution vector is
        # what has to agree
        npt.assert_allclose(float(logL), logL_lenstronomy, rtol=1e-6)
        npt.assert_allclose(np.asarray(param), param_lenstronomy, rtol=1e-4, atol=1e-4)

    def test_likelihood_data_given_model_is_constrained(self):
        """Test that the likelihood uses the constrained solution.

        `ImageLinearFit.likelihood_data_given_model` reaches the solver through an
        unbound `ImageLinearFit.image_linear_solve(self, ...)` call, which bypasses the
        `image_linear_solve` override. If the `likelihood_data_given_model` override is
        ever dropped, this test fails while the one above still passes.
        """
        logL_stock, _ = self.stock_model.likelihood_data_given_model(
            **self._get_kwargs()
        )
        logL, param = self.nn_model.likelihood_data_given_model(**self._get_kwargs())

        assert float(logL) != float(logL_stock)
        assert np.all(np.asarray(param)[self.mge_slice] >= 0)

    def test_likelihood_is_differentiable(self):
        """Test that the likelihood can be differentiated through the bounded solve.

        The gradient descent recipes are only available through JAXtronomy, so a
        non-differentiable solver would make the constraint unusable with them.
        """
        kwargs = self._get_kwargs()

        def logL(theta_E):
            kwargs_lens = deepcopy(kwargs["kwargs_lens"])
            kwargs_lens[0]["theta_E"] = theta_E

            return self.nn_model.likelihood_data_given_model(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs["kwargs_source"],
                kwargs_lens_light=kwargs["kwargs_lens_light"],
                kwargs_ps=kwargs["kwargs_ps"],
            )[0]

        theta_E = float(kwargs["kwargs_lens"][0]["theta_E"])
        gradient = float(jax.grad(logL)(theta_E))
        step = 1e-5
        finite_difference = float(
            (logL(theta_E + step) - logL(theta_E - step)) / (2 * step)
        )

        assert np.isfinite(gradient)
        npt.assert_allclose(gradient, finite_difference, rtol=1e-4)

    def test_falls_back_to_jaxtronomy_without_mge(self):
        """Test that a model without an MGE profile is solved by JAXtronomy."""
        config = deepcopy(self.config)
        config.settings["model"]["lens_light"] = ["SERSIC_ELLIPSE"]

        stock_model = self._get_image_model(config, SingleBandMultiModel)
        nn_model = self._get_image_model(config, NonNegativeMGESingleBandMultiModel)

        _, _, _, param_stock = stock_model.image_linear_solve(
            **self._get_kwargs(config)
        )
        _, _, _, param = nn_model.image_linear_solve(**self._get_kwargs(config))
        npt.assert_array_equal(param, param_stock)

        logL_stock, _ = stock_model.likelihood_data_given_model(
            **self._get_kwargs(config)
        )
        logL, _ = nn_model.likelihood_data_given_model(**self._get_kwargs(config))
        npt.assert_array_equal(logL, logL_stock)

    def test_linear_solver_false(self):
        """Test that the amplitudes are taken from the keyword arguments when the linear
        solver is turned off."""
        stock_model = self._get_image_model(
            self.config, SingleBandMultiModel, linear_solver=False
        )
        nn_model = self._get_image_model(
            self.config, NonNegativeMGESingleBandMultiModel, linear_solver=False
        )

        # the amplitudes are not solved for, so they have to be given
        kwargs = self._get_kwargs()
        for kwargs_light in kwargs["kwargs_lens_light"]:
            kwargs_light["amp"] = np.ones(20)
        for kwargs_light in kwargs["kwargs_source"]:
            kwargs_light["amp"] = np.ones(15)

        logL_stock, param_stock = stock_model.likelihood_data_given_model(**kwargs)
        logL, param = nn_model.likelihood_data_given_model(**kwargs)

        assert param is None and param_stock is None
        npt.assert_array_equal(logL, logL_stock)

    def test_covariance_matrix(self):
        """Test that the covariance matrix of the source amplitudes is returned, which
        the model plots need."""
        _, _, cov_param, _ = self.nn_model.image_linear_solve(
            **self._get_kwargs(), inv_bool=True
        )
        _, _, cov_param_stock, _ = self.stock_model.image_linear_solve(
            **self._get_kwargs(), inv_bool=True
        )

        assert cov_param.shape == (35, 35)
        npt.assert_allclose(cov_param[:15, :15], cov_param_stock[:15, :15], rtol=1e-8)


class TestNNMGESolverJAX(object):
    def setup_method(self):
        self.processor = Processor(_TEST_IO_DIR)
        self.config = self.processor.get_lens_config("lens_system2_mge")

    def test_patches_and_restores(self):
        """Test that the image model class is replaced and then restored."""
        original_multi_linear = multi_linear.SingleBandMultiModel
        original_single_band = single_band_multi_model.SingleBandMultiModel
        original_fitting_sequence = fitting_sequence.SingleBandMultiModel

        with nn_mge_solver(use_jax=True) as enabled:
            assert enabled
            assert (
                multi_linear.SingleBandMultiModel is NonNegativeMGESingleBandMultiModel
            )
            assert (
                single_band_multi_model.SingleBandMultiModel
                is NonNegativeMGESingleBandMultiModel
            )
            # `psf_iteration` builds a lenstronomy image model even under JAXtronomy
            assert fitting_sequence.SingleBandMultiModel is LenstronomyNonNegativeMGE

        assert multi_linear.SingleBandMultiModel is original_multi_linear
        assert single_band_multi_model.SingleBandMultiModel is original_single_band
        assert fitting_sequence.SingleBandMultiModel is original_fitting_sequence

    def test_restores_on_exception(self):
        """Test that the image model class is restored if the context raises."""
        original = multi_linear.SingleBandMultiModel

        with pytest.raises(ValueError):
            with nn_mge_solver(use_jax=True):
                raise ValueError

        assert multi_linear.SingleBandMultiModel is original

    def test_disabled(self):
        """Test that nothing is patched when the solver is disabled."""
        original = multi_linear.SingleBandMultiModel

        with nn_mge_solver(enabled=False, use_jax=True) as enabled:
            assert not enabled
            assert multi_linear.SingleBandMultiModel is original

    def test_lenstronomy_is_untouched(self):
        """Test that the JAXtronomy patch leaves lenstronomy alone, and vice versa."""
        import lenstronomy.ImSim.MultiBand.multi_linear as lenstronomy_multi_linear

        with nn_mge_solver(use_jax=True):
            assert (
                lenstronomy_multi_linear.SingleBandMultiModel
                is not LenstronomyNonNegativeMGE
            )

        original = multi_linear.SingleBandMultiModel
        with nn_mge_solver(use_jax=False):
            assert multi_linear.SingleBandMultiModel is original

    def test_create_im_sim_uses_patched_class(self):
        """Test that JAXtronomy builds the patched image model class."""
        kwargs_data_joint = self.processor.get_kwargs_data_joint("lens_system2_mge")
        kwargs = dict(
            multi_band_list=kwargs_data_joint["multi_band_list"],
            multi_band_type="multi-linear",
            kwargs_model=_get_kwargs_model(self.config),
            image_likelihood_mask_list=self.config.get_masks(),
        )

        with nn_mge_solver(use_jax=True):
            im_sim = create_im_sim(**kwargs)
            assert isinstance(
                im_sim._image_model_list[0], NonNegativeMGESingleBandMultiModel
            )

        im_sim = create_im_sim(**kwargs)
        assert not isinstance(
            im_sim._image_model_list[0], NonNegativeMGESingleBandMultiModel
        )

    def test_patched_class_is_picklable_by_reference(self):
        """Test that the image model can be sent to the multiprocessing workers.

        The sampler pool spawns its workers, so the image model instances are pickled
        and the class needs to be importable.
        """
        import dill

        assert b"dolphin.processor.nn_mge_jax" in dill.dumps(
            NonNegativeMGESingleBandMultiModel
        )
