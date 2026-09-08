# -*- coding: utf-8 -*-
"""This module implements the non-negative MGE (nn-MGE) semi-linear inversion of He et
al. 2024 (MNRAS 532, 2441).

lenstronomy solves the semi-linear inversion with an unconstrained weighted least
squares, which routinely returns negative Gaussian amplitudes for MGE_SET and
MGE_SET_ELLIPSE lens light. This module constrains those amplitudes to be non-negative
while leaving every other linear amplitude free, since the shapelet source basis is
signed and needs negative coefficients.
"""

__author__ = "bwedig"

import contextlib
import importlib
import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import lsq_linear

import lenstronomy.ImSim.de_lens as de_lens
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel

MGE_PROFILE_NAMES = ("MGE_SET", "MGE_SET_ELLIPSE")

# Modules holding a `SingleBandMultiModel` reference that lenstronomy resolves at
# instantiation time: the first covers `multi_band_type="multi-linear"`, the second
# `"single-band"` (used by `ModelPlot`).
_PATCH_TARGETS = (
    ("lenstronomy.ImSim.MultiBand.multi_linear", "SingleBandMultiModel"),
    ("lenstronomy.ImSim.MultiBand.single_band_multi_model", "SingleBandMultiModel"),
)


def get_param_bounded_WLS(
    A, C_D_inv, d, lower_bounds, upper_bounds=None, inv_bool=True
):
    """Solve the bounded weighted least squares problem for the linear amplitudes.

    This is the constrained analogue of `lenstronomy.ImSim.de_lens.get_param_WLS` and
    returns the same tuple. The problem is reduced to the normal equations,
    `M = A^T C^-1 A` and `R = A^T C^-1 d`, and solved on the Cholesky factor `M = L L^T`.
    Since `||L^T x - L^-1 R||^2` differs from the original objective only by a constant,
    the reduced solution is exact, and the system is `n x n` in the number of amplitudes
    rather than the number of pixels.

    :param A: response matrix, Nd x Ns (Nd data points, Ns linear parameters)
    :type A: `numpy.ndarray`
    :param C_D_inv: inverse data variance, 1d array of length Nd
    :type C_D_inv: `numpy.ndarray`
    :param d: data array, 1d of length Nd
    :type d: `numpy.ndarray`
    :param lower_bounds: lower bound for each linear parameter, `-numpy.inf` for the
        unconstrained ones
    :type lower_bounds: `numpy.ndarray`
    :param upper_bounds: upper bound for each linear parameter, `None` for no bound
    :type upper_bounds: `numpy.ndarray` or `None`
    :param inv_bool: if `True`, also return the parameter covariance matrix
    :type inv_bool: `bool`
    :return: linear parameters, covariance matrix, model image
    :rtype: `tuple`
    """
    M = A.T.dot(np.multiply(C_D_inv, A.T).T)
    num_param = A.shape[1]

    if not de_lens._cond_inv(M):
        return _degenerate_solution(A, M, num_param, inv_bool)

    R = A.T.dot(np.multiply(C_D_inv, d))

    try:
        L = linalg.cholesky(M, lower=True)
    except linalg.LinAlgError:
        # M is positive semi-definite by construction, so this only fires on numerical
        # rank deficiency. Match lenstronomy's degenerate weighted-least-squares path
        # rather than regularizing a solution the unconstrained solver would not give.
        warnings.warn(
            "Cholesky decomposition of the normal matrix failed, returning zero linear "
            "amplitudes.",
            RuntimeWarning,
        )
        return _degenerate_solution(A, M, num_param, inv_bool)

    g = linalg.solve_triangular(L, R, lower=True)

    if upper_bounds is None:
        upper_bounds = np.full(num_param, np.inf)

    # `bvls` is an exact active-set method, appropriate for the small dense system here.
    # The default `trf` is iterative and roughly three times slower.
    param = lsq_linear(L.T, g, bounds=(lower_bounds, upper_bounds), method="bvls").x

    # the solution can sit a floating point epsilon outside the bounds, which would
    # leave amplitudes at around -1e-15 instead of 0
    param = np.clip(param, lower_bounds, upper_bounds)

    # The covariance is that of the *unconstrained* problem. Its only consumers are
    # `de_lens.marginalization_new` (only when `source_marg` is True) and
    # `ImageLinearFit.error_map_source`, which reads the source block alone, and the
    # source amplitudes are never bounded here. It is not a valid uncertainty for
    # amplitudes sitting at the bound.
    cov_param = de_lens._stable_inv(M) if inv_bool else None

    return param, cov_param, A.dot(param)


def _degenerate_solution(A, M, num_param, inv_bool):
    """Return the zero solution for an ill-conditioned normal matrix.

    Matches the behavior of `lenstronomy.ImSim.de_lens.get_param_WLS`.

    :param A: response matrix, Nd x Ns
    :type A: `numpy.ndarray`
    :param M: normal matrix, Ns x Ns
    :type M: `numpy.ndarray`
    :param num_param: number of linear parameters, Ns
    :type num_param: `int`
    :param inv_bool: if `True`, return a zero covariance matrix instead of `None`
    :type inv_bool: `bool`
    :return: linear parameters, covariance matrix, model image
    :rtype: `tuple`
    """
    param = np.zeros(num_param)
    cov_param = np.zeros_like(M) if inv_bool else None

    return param, cov_param, A.dot(param)


def mge_lower_bounds(
    source_model, lens_light_model, kwargs_source, kwargs_lens_light, num_param
):
    """Create the lower bound for each column of the linear response matrix.

    The columns of the response matrix are ordered as source light, then lens light,
    then point sources (see `ImageLinearFit.linear_response_matrix`). Only the MGE lens
    light columns are bounded; the rest are left free.

    :param source_model: source `LightModel` instance for this band
    :type source_model: `lenstronomy.LightModel.light_model.LightModel`
    :param lens_light_model: lens light `LightModel` instance for this band
    :type lens_light_model: `lenstronomy.LightModel.light_model.LightModel`
    :param kwargs_source: list of source light keyword arguments for this band
    :type kwargs_source: `list` of `dict`
    :param kwargs_lens_light: list of lens light keyword arguments for this band
    :type kwargs_lens_light: `list` of `dict`
    :param num_param: total number of linear parameters, i.e. columns of the response
        matrix
    :type num_param: `int`
    :return: lower bound for each linear parameter, or `None` if the lens light model
        has no MGE profile
    :rtype: `numpy.ndarray` or `None`
    """
    profile_type_list = lens_light_model.profile_type_list

    if not any(m in MGE_PROFILE_NAMES for m in profile_type_list):
        return None

    num_source = source_model.num_param_linear(kwargs_source)
    num_lens_light_list = lens_light_model.num_param_linear_list(kwargs_lens_light)

    if num_source + int(np.sum(num_lens_light_list)) > num_param:
        raise ValueError(
            "The source ({}) and lens light ({}) linear parameters do not fit within "
            "the {} columns of the response matrix.".format(
                num_source, int(np.sum(num_lens_light_list)), num_param
            )
        )

    lower_bounds = np.full(num_param, -np.inf)

    index = num_source
    for model, num_model in zip(profile_type_list, num_lens_light_list):
        if model in MGE_PROFILE_NAMES:
            lower_bounds[index : index + num_model] = 0.0
        index += num_model

    return lower_bounds


class NonNegativeMGESingleBandMultiModel(SingleBandMultiModel):
    """Drop-in replacement for lenstronomy's `SingleBandMultiModel` that solves the
    semi-linear inversion with non-negative MGE amplitudes.

    Both `image_linear_solve` and `likelihood_data_given_model` must be overridden.
    lenstronomy reaches the solver through explicit unbound calls of the form
    `ImageLinearFit.image_linear_solve(self, ...)`, which bypass subclass overrides, so
    overriding `image_linear_solve` alone would leave the sampling path unconstrained.
    """

    @property
    def _has_mge(self):
        """Whether this band's lens light model contains an MGE profile.

        :return: `True` if the lens light model list has an MGE profile
        :rtype: `bool`
        """
        if getattr(self, "_has_mge_cache", None) is None:
            self._has_mge_cache = any(
                m in MGE_PROFILE_NAMES for m in self.LensLightModel.profile_type_list
            )

        return self._has_mge_cache

    def _use_bounded_solver(self):
        """Whether the bounded solver applies to this band.

        The pixel-based and interferometric solvers are handled by lenstronomy.

        :return: `True` if the bounded solver should be used
        :rtype: `bool`
        """
        return (
            self._has_mge
            and not self._pixelbased_bool
            and self.Data.likelihood_method() == "diagonal"
        )

    def image_linear_solve(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        inv_bool=False,
    ):
        """Compute the image, solving for the linear amplitudes with the MGE amplitudes
        constrained to be non-negative.

        :param kwargs_lens: list of dicts containing lens model keyword arguments
        :type kwargs_lens: `list` of `dict`
        :param kwargs_source: list of dicts containing source model keyword arguments
        :type kwargs_source: `list` of `dict`
        :param kwargs_lens_light: list of dicts containing lens light model keyword arguments
        :type kwargs_lens_light: `list` of `dict`
        :param kwargs_ps: list of dicts containing point source keyword arguments
        :type kwargs_ps: `list` of `dict`
        :param kwargs_extinction: list of dicts containing extinction keyword arguments
        :type kwargs_extinction: `list` of `dict`
        :param kwargs_special: dict containing special keyword arguments
        :type kwargs_special: `dict`
        :param inv_bool: if `True`, also return the covariance matrix of the solution
        :type inv_bool: `bool`
        :return: model image, model error, covariance matrix, linear parameters
        :rtype: `tuple`
        """
        if not self._use_bounded_solver():
            return super().image_linear_solve(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                kwargs_extinction,
                kwargs_special,
                inv_bool=inv_bool,
            )

        (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
        ) = self.select_kwargs(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_extinction
        )

        # The unbound calls are deliberate: the `SingleBandMultiModel` versions would
        # apply `select_kwargs` a second time.
        A = ImageLinearFit.linear_response_matrix(
            self,
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
            kwargs_special,
        )
        C_D_response, model_error = ImageModel.error_response(
            self, kwargs_lens_i, kwargs_ps_i, kwargs_special=kwargs_special
        )
        d = self.data_response

        lower_bounds = mge_lower_bounds(
            self.SourceModel,
            self.LensLightModel,
            kwargs_source_i,
            kwargs_lens_light_i,
            num_param=A.shape[0],
        )
        param, cov_param, wls_model = get_param_bounded_WLS(
            A.T, 1.0 / C_D_response, d, lower_bounds, inv_bool=inv_bool
        )

        model = self.array_masked2image(wls_model)
        _, _, _, _ = ImageLinearFit.update_linear_kwargs(
            self,
            param,
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
        )

        return model, model_error, cov_param, param

    def likelihood_data_given_model(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        source_marg=False,
        linear_prior=None,
        check_positive_flux=False,
        linear_solver=None,
    ):
        """Compute the likelihood of the data given the model, with the MGE amplitudes
        constrained to be non-negative.

        :param kwargs_lens: list of dicts containing lens model keyword arguments
        :type kwargs_lens: `list` of `dict`
        :param kwargs_source: list of dicts containing source model keyword arguments
        :type kwargs_source: `list` of `dict`
        :param kwargs_lens_light: list of dicts containing lens light model keyword arguments
        :type kwargs_lens_light: `list` of `dict`
        :param kwargs_ps: list of dicts containing point source keyword arguments
        :type kwargs_ps: `list` of `dict`
        :param kwargs_extinction: list of dicts containing extinction keyword arguments
        :type kwargs_extinction: `list` of `dict`
        :param kwargs_special: dict containing special keyword arguments
        :type kwargs_special: `dict`
        :param source_marg: if `True`, marginalize over the linear parameters
        :type source_marg: `bool`
        :param linear_prior: linear prior width in eigenvalues
        :type linear_prior: `float` or `None`
        :param check_positive_flux: if `True`, penalize negative flux components
        :type check_positive_flux: `bool`
        :param linear_solver: if `None`, uses `self.linear_solver`
        :type linear_solver: `bool` or `None`
        :return: log likelihood, linear parameters
        :rtype: `tuple`
        """
        if linear_solver is None:
            linear_solver = self.linear_solver

        if not linear_solver or not self._use_bounded_solver():
            return super().likelihood_data_given_model(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                kwargs_extinction,
                kwargs_special,
                source_marg=source_marg,
                linear_prior=linear_prior,
                check_positive_flux=check_positive_flux,
                linear_solver=linear_solver,
            )

        (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            _,
        ) = self.select_kwargs(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_extinction
        )

        # `image_linear_solve` applies `select_kwargs` itself, so it takes the full
        # kwargs while the solution below takes the band-selected ones.
        im_sim, model_error, cov_matrix, param = self.image_linear_solve(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            kwargs_extinction,
            kwargs_special,
            inv_bool=source_marg,
        )

        logL = self.likelihood_data_given_model_solution(
            im_sim,
            model_error,
            cov_matrix,
            param,
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            source_marg=source_marg,
            linear_prior=linear_prior,
            check_positive_flux=check_positive_flux,
        )

        return logL, param


@contextlib.contextmanager
def nn_mge_solver(enabled=True):
    """Make lenstronomy build image models that solve for non-negative MGE amplitudes.

    lenstronomy offers no hook to inject an image model class, so the class it resolves
    at instantiation time is replaced for the duration of the context and restored
    afterwards.

    :param enabled: if `False`, this is a no-op
    :type enabled: `bool`
    :return: whether the solver is active
    :rtype: `bool`
    """
    if not enabled:
        yield False
        return

    modules = [importlib.import_module(name) for name, _ in _PATCH_TARGETS]
    originals = [
        getattr(module, attribute)
        for module, (_, attribute) in zip(modules, _PATCH_TARGETS)
    ]

    for module, (_, attribute) in zip(modules, _PATCH_TARGETS):
        setattr(module, attribute, NonNegativeMGESingleBandMultiModel)

    try:
        yield True
    finally:
        for module, (_, attribute), original in zip(modules, _PATCH_TARGETS, originals):
            setattr(module, attribute, original)
