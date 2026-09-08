# -*- coding: utf-8 -*-
"""Tests for nn_mge module."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import lsq_linear

import lenstronomy.ImSim.de_lens as de_lens
import lenstronomy.ImSim.MultiBand.multi_linear as multi_linear
import lenstronomy.ImSim.MultiBand.single_band_multi_model as single_band_multi_model
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from lenstronomy.Util.class_creator import create_im_sim

from dolphin.processor.core import Processor
from dolphin.processor.nn_mge import (
    NonNegativeMGESingleBandMultiModel,
    get_param_bounded_WLS,
    mge_lower_bounds,
    nn_mge_solver,
)

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"

_NUM_BOUNDED = 8
_NUM_FREE = 4


class TestGetParamBoundedWLS(object):
    def setup_method(self):
        rng = np.random.default_rng(42)

        self.num_param = _NUM_FREE + _NUM_BOUNDED
        self.A = rng.normal(size=(500, self.num_param))
        self.C_D_inv = np.full(500, 4.0)
        self.lower_bounds = np.concatenate(
            [np.full(_NUM_FREE, -np.inf), np.zeros(_NUM_BOUNDED)]
        )

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
        """Test that the bounded solution matches the unconstrained one when the latter
        already satisfies the bounds."""
        param_wls, cov_wls, image_wls = de_lens.get_param_WLS(
            self.A, self.C_D_inv, self.d_positive
        )
        # guard the premise of the test
        assert np.all(param_wls[_NUM_FREE:] > 0)

        param, cov, image = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_positive, self.lower_bounds
        )

        npt.assert_allclose(param, param_wls, rtol=1e-8)
        npt.assert_allclose(image, image_wls, rtol=1e-8)
        npt.assert_allclose(cov, cov_wls, rtol=1e-8)

    def test_enforces_lower_bounds(self):
        """Test that the bounded amplitudes are non-negative and that the free
        amplitudes are refitted rather than left alone."""
        param_wls, _, _ = de_lens.get_param_WLS(self.A, self.C_D_inv, self.d_mixed)
        assert np.any(param_wls[_NUM_FREE:] < 0)

        param, _, _ = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_mixed, self.lower_bounds
        )

        assert np.all(param[_NUM_FREE:] >= 0)
        # at least one amplitude sits at the bound, so the constraint is active
        assert np.any(param[_NUM_FREE:] == 0.0)
        # the free amplitudes absorb the truncation, which clipping would not do
        assert not np.allclose(param[:_NUM_FREE], param_wls[:_NUM_FREE])

    def test_matches_full_design_bounded_solve(self):
        """Test that reducing to the normal equations gives the same solution as solving
        the full whitened design matrix."""
        param, _, _ = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_mixed, self.lower_bounds
        )

        weight = np.sqrt(self.C_D_inv)
        param_full = lsq_linear(
            self.A * weight[:, None],
            self.d_mixed * weight,
            bounds=(self.lower_bounds, np.full(self.num_param, np.inf)),
            method="bvls",
        ).x

        npt.assert_allclose(param, param_full, atol=1e-9)

    def test_ill_conditioned_returns_zeros(self):
        """Test that a singular normal matrix gives the zero solution, as in
        `de_lens.get_param_WLS`."""
        A = np.copy(self.A)
        A[:, -1] = A[:, -2]

        param, cov, image = get_param_bounded_WLS(
            A, self.C_D_inv, self.d_mixed, self.lower_bounds
        )
        npt.assert_array_equal(param, np.zeros(self.num_param))
        npt.assert_array_equal(image, np.zeros(len(self.d_mixed)))
        npt.assert_array_equal(cov, np.zeros((self.num_param, self.num_param)))

        _, cov, _ = get_param_bounded_WLS(
            A, self.C_D_inv, self.d_mixed, self.lower_bounds, inv_bool=False
        )
        assert cov is None

    def test_upper_bounds(self):
        """Test that upper bounds are applied when given and infinite otherwise."""
        upper_bounds = np.full(self.num_param, 1.2)

        param, _, _ = get_param_bounded_WLS(
            self.A,
            self.C_D_inv,
            self.d_positive,
            self.lower_bounds,
            upper_bounds=upper_bounds,
        )
        assert np.all(param <= 1.2 + 1e-12)
        assert np.any(param >= 1.2 - 1e-12)

        param, _, _ = get_param_bounded_WLS(
            self.A, self.C_D_inv, self.d_positive, self.lower_bounds
        )
        assert np.any(param > 1.2)


class TestMGELowerBounds(object):
    def setup_method(self):
        self.processor = Processor(_TEST_IO_DIR)
        self.config = self.processor.get_lens_config("lens_system2_mge")
        self.kwargs_data_joint = self.processor.get_kwargs_data_joint(
            "lens_system2_mge"
        )
        self.image_model = self._get_image_model(self.config)

        kwargs_params = self.config.get_kwargs_params()
        self.kwargs_source = kwargs_params["source_model"][0]
        self.kwargs_lens_light = kwargs_params["lens_light_model"][0]

    def _get_image_model(self, config):
        """Create the image model for the first band of a given config.

        :param config: config to create the image model from
        :type config: `dolphin.processor.config.ModelConfig`
        :return: image model for the first band
        :rtype: `NonNegativeMGESingleBandMultiModel`
        """
        return NonNegativeMGESingleBandMultiModel(
            self.kwargs_data_joint["multi_band_list"],
            config.get_kwargs_model(),
            likelihood_mask_list=config.get_masks(),
            band_index=0,
        )

    def test_shapelets_source_with_mge_lens_light(self):
        """Test the bounds for a shapelet source and an MGE lens light."""
        # n_max = 4 gives 15 shapelet coefficients, n_comp = 20 gives 20 Gaussians
        lower_bounds = mge_lower_bounds(
            self.image_model.SourceModel,
            self.image_model.LensLightModel,
            self.kwargs_source,
            self.kwargs_lens_light,
            num_param=35,
        )

        assert lower_bounds.shape == (35,)
        assert np.all(np.isneginf(lower_bounds[:15]))
        npt.assert_array_equal(lower_bounds[15:], np.zeros(20))

    def test_offset_with_leading_non_mge_lens_light(self):
        """Test that a non-MGE lens light profile ahead of the MGE shifts the bounds."""
        config = deepcopy(self.config)
        config.settings["model"]["lens_light"] = ["SERSIC_ELLIPSE", "MGE_SET_ELLIPSE"]
        config.settings["lens_light_options"]["mge_config"] = {1: {"n_comp": 20}}
        image_model = self._get_image_model(config)

        kwargs_lens_light = config.get_kwargs_params()["lens_light_model"][0]
        lower_bounds = mge_lower_bounds(
            image_model.SourceModel,
            image_model.LensLightModel,
            self.kwargs_source,
            kwargs_lens_light,
            num_param=36,
        )

        assert np.all(np.isneginf(lower_bounds[:16]))
        npt.assert_array_equal(lower_bounds[16:], np.zeros(20))

    def test_returns_none_without_mge(self):
        """Test that a lens light model without an MGE profile is unconstrained."""
        config = deepcopy(self.config)
        config.settings["model"]["lens_light"] = ["SERSIC_ELLIPSE"]
        image_model = self._get_image_model(config)

        lower_bounds = mge_lower_bounds(
            image_model.SourceModel,
            image_model.LensLightModel,
            self.kwargs_source,
            config.get_kwargs_params()["lens_light_model"][0],
            num_param=16,
        )

        assert lower_bounds is None

    def test_too_few_columns_raises(self):
        """Test that too few response matrix columns raises an error."""
        with pytest.raises(ValueError):
            mge_lower_bounds(
                self.image_model.SourceModel,
                self.image_model.LensLightModel,
                self.kwargs_source,
                self.kwargs_lens_light,
                num_param=20,
            )


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
            config.get_kwargs_model(),
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
        assert np.any(param_stock[self.mge_slice] < 0)

        kwargs = self._get_kwargs()
        _, _, _, param = self.nn_model.image_linear_solve(**kwargs)

        assert np.all(param[self.mge_slice] >= 0)
        # the amplitudes are also written back into the keyword arguments
        assert np.all(np.atleast_1d(kwargs["kwargs_lens_light"][0]["amp"]) >= 0)

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

        kwargs = self._get_kwargs()
        logL, param = self.nn_model.likelihood_data_given_model(**kwargs)

        assert logL != logL_stock
        assert np.all(param[self.mge_slice] >= 0)
        assert np.all(np.atleast_1d(kwargs["kwargs_lens_light"][0]["amp"]) >= 0)

    def test_falls_back_to_lenstronomy_without_mge(self):
        """Test that a model without an MGE profile is solved by lenstronomy."""
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


class TestNNMGESolver(object):
    def setup_method(self):
        self.processor = Processor(_TEST_IO_DIR)
        self.config = self.processor.get_lens_config("lens_system2_mge")

    def test_patches_and_restores(self):
        """Test that the image model class is replaced and then restored."""
        original_multi_linear = multi_linear.SingleBandMultiModel
        original_single_band = single_band_multi_model.SingleBandMultiModel

        with nn_mge_solver() as enabled:
            assert enabled
            assert (
                multi_linear.SingleBandMultiModel is NonNegativeMGESingleBandMultiModel
            )
            assert (
                single_band_multi_model.SingleBandMultiModel
                is NonNegativeMGESingleBandMultiModel
            )

        assert multi_linear.SingleBandMultiModel is original_multi_linear
        assert single_band_multi_model.SingleBandMultiModel is original_single_band

    def test_restores_on_exception(self):
        """Test that the image model class is restored if the context raises."""
        original = multi_linear.SingleBandMultiModel

        with pytest.raises(ValueError):
            with nn_mge_solver():
                raise ValueError

        assert multi_linear.SingleBandMultiModel is original

    def test_disabled(self):
        """Test that nothing is patched when the solver is disabled."""
        original = multi_linear.SingleBandMultiModel

        with nn_mge_solver(enabled=False) as enabled:
            assert not enabled
            assert multi_linear.SingleBandMultiModel is original

    def test_create_im_sim_uses_patched_class(self):
        """Test that lenstronomy builds the patched image model class."""
        kwargs_data_joint = self.processor.get_kwargs_data_joint("lens_system2_mge")
        kwargs = dict(
            multi_band_list=kwargs_data_joint["multi_band_list"],
            multi_band_type="multi-linear",
            kwargs_model=self.config.get_kwargs_model(),
            image_likelihood_mask_list=self.config.get_masks(),
        )

        with nn_mge_solver():
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

        assert b"dolphin.processor.nn_mge" in dill.dumps(
            NonNegativeMGESingleBandMultiModel
        )
