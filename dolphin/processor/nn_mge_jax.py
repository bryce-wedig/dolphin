# -*- coding: utf-8 -*-
"""JAXtronomy implementation of the non-negative MGE (nn-MGE) semi-linear inversion of
He et al. 2024 (MNRAS 532, 2441).

This is the JAXtronomy counterpart of `dolphin.processor.nn_mge`, which does the same
thing for lenstronomy. The bounded solve cannot be shared between the two: every
JAXtronomy `ImSim` method is `jax.jit` compiled, and the gradient descent optimizer
differentiates straight through the likelihood, so `scipy.optimize.lsq_linear` is not
available here. The active set is instead found with a JAX-native Lawson-Hanson loop and
the amplitudes are then solved on the free columns alone, which keeps the solve
differentiable.

This module imports JAX at import time, so it must only ever be imported lazily.
"""

__author__ = "bwedig"

from functools import partial

import numpy as np

from ..util import enable_jax_x64

# has to happen before JAX is imported, hence the import order below
enable_jax_x64()

from jax import jit, lax, numpy as jnp

import jaxtronomy.ImSim.de_lens as de_lens
from jaxtronomy.ImSim.de_lens import EPSILON
from jaxtronomy.ImSim.image_linear_solve import ImageLinearFit
from jaxtronomy.ImSim.image_model import ImageModel
from jaxtronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel

from .nn_mge import MGE_PROFILE_NAMES, mge_lower_bounds

# Modules holding a `SingleBandMultiModel` reference that JAXtronomy resolves at
# instantiation time: the first covers `multi_band_type="multi-linear"`, the second
# `"single-band"`.
PATCH_TARGETS = (
    ("jaxtronomy.ImSim.MultiBand.multi_linear", "SingleBandMultiModel"),
    ("jaxtronomy.ImSim.MultiBand.single_band_multi_model", "SingleBandMultiModel"),
)

# `jaxtronomy.Workflow.fitting_sequence` binds *lenstronomy*'s `SingleBandMultiModel` at
# import time and builds one in `psf_iteration`. Patching the lenstronomy module alone
# does not reach a name that is already bound, so this target takes the lenstronomy
# class from `nn_mge`.
LENSTRONOMY_PATCH_TARGETS = (
    ("jaxtronomy.Workflow.fitting_sequence", "SingleBandMultiModel"),
)

# Tolerances for the active set method, relative to the scale of the problem. The
# amplitudes and the data are in instrumental units, so the normal equations can be
# scaled arbitrarily and an absolute tolerance would either stop early or spend
# iterations chasing rounding error.

# a bounded amplitude closer to zero than this fraction of the largest amplitude is
# treated as sitting on its bound
_ZERO_TOL = 1e-12

# an amplitude is only brought back into the free set if the objective improves by more
# than this fraction of the largest gradient component
_OPTIMALITY_TOL = 1e-10


def _masked_solve(M, R, free):
    """Solve the normal equations on the free columns only, with zeros elsewhere.

    The rows and columns outside the free set are replaced by the identity, so the
    system keeps its shape and the solution is zero there. This is the same reduction
    the active set method applies at every iteration.

    :param M: normal matrix, Ns x Ns
    :type M: `jax.numpy.ndarray`
    :param R: right hand side, 1d of length Ns
    :type R: `jax.numpy.ndarray`
    :param free: boolean mask of the columns to solve for
    :type free: `jax.numpy.ndarray`
    :return: solution, zero outside the free set
    :rtype: `jax.numpy.ndarray`
    """
    identity = jnp.eye(M.shape[0], dtype=M.dtype)

    M_free = jnp.where(free[:, None] & free[None, :], M, identity)
    R_free = jnp.where(free, R, 0.0)

    param = jnp.nan_to_num(
        jnp.linalg.solve(M_free, R_free), nan=0.0, posinf=0.0, neginf=0.0
    )

    return jnp.where(free, param, 0.0)


def _active_set(M, R, bounded):
    """Find the free set of the bounded least squares problem.

    This is the Lawson-Hanson active set method applied to the normal equations,
    specialized to a lower bound of zero on a subset of the parameters. The columns that
    are not bounded are free throughout. The outer loop brings in the bounded column
    that most improves the objective, and the inner loop backtracks toward the feasible
    region whenever the resulting solution goes negative.

    The result is a boolean mask rather than the solution, which is what makes the
    method usable under `jax.grad`: a boolean array carries no tangent, so reverse mode
    never has to transpose the `lax.while_loop`. See `_bounded_solve`.

    :param M: normal matrix, Ns x Ns
    :type M: `jax.numpy.ndarray`
    :param R: right hand side, 1d of length Ns
    :type R: `jax.numpy.ndarray`
    :param bounded: boolean mask of the parameters bounded below by zero
    :type bounded: `jax.numpy.ndarray`
    :return: boolean mask of the free parameters
    :rtype: `jax.numpy.ndarray`
    """
    num_param = M.shape[0]

    # every bounded parameter can enter the free set once, and the inner loop can only
    # remove parameters, so this is a generous cap on a terminating iteration
    max_iter = 3 * num_param + 10

    free_init = ~bounded

    # the normal equations carry the units of the data, so the tolerance is taken
    # relative to the scale of the problem
    optimality_tol = _OPTIMALITY_TOL * jnp.max(jnp.abs(R))

    def outer_condition(state):
        _, _, iteration, converged = state

        return (~converged) & (iteration < max_iter)

    def outer_body(state):
        param, free, iteration, _ = state

        # the negative gradient of the objective; a bounded parameter sitting on its
        # bound can only improve the fit if its gradient points into the feasible region
        gradient = R - M.dot(param)
        candidates = bounded & (~free) & (gradient > optimality_tol)
        any_candidate = jnp.any(candidates)

        index = jnp.argmax(jnp.where(candidates, gradient, -jnp.inf))
        free_new = free.at[index].set(True)
        solution = _masked_solve(M, R, free_new)

        def inner_condition(inner_state):
            _, free_inner, solution_inner = inner_state

            return jnp.any(bounded & free_inner & (solution_inner < 0.0))

        def inner_body(inner_state):
            param_inner, free_inner, solution_inner = inner_state

            # step as far toward the unconstrained solution as feasibility allows
            infeasible = bounded & free_inner & (solution_inner < 0.0)
            step = jnp.min(
                jnp.where(
                    infeasible,
                    param_inner / jnp.maximum(param_inner - solution_inner, 1e-300),
                    jnp.inf,
                )
            )
            param_inner = param_inner + step * (solution_inner - param_inner)
            # the line search drives at least one bounded amplitude to zero, up to
            # rounding, so the amplitudes are compared to their own scale
            at_bound = jnp.abs(param_inner) <= _ZERO_TOL * jnp.max(jnp.abs(param_inner))
            free_inner = free_inner & ~(bounded & at_bound)

            return param_inner, free_inner, _masked_solve(M, R, free_inner)

        _, free_new, solution = lax.while_loop(
            inner_condition, inner_body, (param, free_new, solution)
        )

        return (
            jnp.where(any_candidate, solution, param),
            jnp.where(any_candidate, free_new, free),
            iteration + 1,
            ~any_candidate,
        )

    _, free, _, _ = lax.while_loop(
        outer_condition,
        outer_body,
        (_masked_solve(M, R, free_init), free_init, 0, False),
    )

    return free


def _bounded_solve(M, R, bounded):
    """Solve the normal equations with the bounded parameters constrained to be
    non-negative.

    The active set is found on values detached from the computational graph and returned
    as a boolean mask, and the amplitudes are then solved on the free columns with an
    ordinary linear solve. Holding the active set fixed is the correct derivative of the
    constrained solution wherever the active set is locally constant, which is
    everywhere except the measure-zero set where a parameter is entering or leaving its
    bound.

    :param M: normal matrix, Ns x Ns
    :type M: `jax.numpy.ndarray`
    :param R: right hand side, 1d of length Ns
    :type R: `jax.numpy.ndarray`
    :param bounded: boolean mask of the parameters bounded below by zero
    :type bounded: `jax.numpy.ndarray`
    :return: linear parameters
    :rtype: `jax.numpy.ndarray`
    """
    free = lax.stop_gradient(
        _active_set(lax.stop_gradient(M), lax.stop_gradient(R), bounded)
    )

    param = _masked_solve(M, R, free)

    # the solve can leave a freed parameter a floating point epsilon below its bound
    return jnp.where(bounded, jnp.maximum(param, 0.0), param)


@partial(jit, static_argnums=4)
def get_param_bounded_WLS(A, C_D_inv, d, bounded, inv_bool=True):
    """Solve the bounded weighted least squares problem for the linear amplitudes.

    This is the constrained analogue of `jaxtronomy.ImSim.de_lens.get_param_WLS` and
    returns the same tuple. The problem is reduced to the normal equations,
    `M = A^T C^-1 A` and `R = A^T C^-1 d`, and solved on them directly, which is the
    same reduction JAXtronomy's unconstrained solver makes.

    :param A: response matrix, Nd x Ns (Nd data points, Ns linear parameters)
    :type A: `jax.numpy.ndarray`
    :param C_D_inv: inverse data variance, 1d array of length Nd
    :type C_D_inv: `jax.numpy.ndarray`
    :param d: data array, 1d of length Nd
    :type d: `jax.numpy.ndarray`
    :param bounded: boolean mask of the parameters bounded below by zero
    :type bounded: `jax.numpy.ndarray`
    :param inv_bool: if `True`, also return the parameter covariance matrix
    :type inv_bool: `bool`
    :return: linear parameters, covariance matrix, model image
    :rtype: `tuple`
    """
    A = jnp.asarray(A, dtype=float)
    C_D_inv = jnp.asarray(C_D_inv, dtype=float)
    d = jnp.asarray(d, dtype=float)
    bounded = jnp.asarray(bounded, dtype=bool)

    M = A.T.dot(jnp.multiply(C_D_inv, A.T).T)
    R = A.T.dot(jnp.multiply(C_D_inv, d))

    # matches `de_lens.get_param_WLS`, which returns the zero solution rather than a
    # solution to an ill-conditioned system
    stability_check = jnp.linalg.cond(M) < 5 / EPSILON

    param = jnp.where(
        stability_check, _bounded_solve(M, R, bounded), jnp.zeros(M.shape[0])
    )

    # The covariance is that of the *unconstrained* problem. Its only consumers are
    # `de_lens.marginalization_new` (only when `source_marg` is True) and
    # `ImageLinearFit.error_map_source`, which reads the source block alone, and the
    # source amplitudes are never bounded here. It is not a valid uncertainty for
    # amplitudes sitting at the bound.
    if inv_bool:
        cov_param = jnp.where(
            stability_check, de_lens._stable_inv(M), jnp.zeros_like(M)
        )
    else:
        cov_param = None

    return param, cov_param, A.dot(param)


class NonNegativeMGESingleBandMultiModel(SingleBandMultiModel):
    """Drop-in replacement for JAXtronomy's `SingleBandMultiModel` that solves the
    semi-linear inversion with non-negative MGE amplitudes.

    Both `image_linear_solve` and `likelihood_data_given_model` must be overridden.
    JAXtronomy reaches the solver through explicit unbound calls of the form
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

        The interferometric solver is handled by JAXtronomy, which has no pixel-based
        solver.

        :return: `True` if the bounded solver should be used
        :rtype: `bool`
        """
        return self._has_mge and self.Data.likelihood_method() == "diagonal"

    def _bounded_mask(self, kwargs_source, kwargs_lens_light, num_param):
        """Boolean mask of the linear parameters that are bounded below by zero.

        :param kwargs_source: list of source light keyword arguments for this band
        :type kwargs_source: `list` of `dict`
        :param kwargs_lens_light: list of lens light keyword arguments for this band
        :type kwargs_lens_light: `list` of `dict`
        :param num_param: total number of linear parameters
        :type num_param: `int`
        :return: boolean mask of the bounded parameters
        :rtype: `numpy.ndarray`
        """
        lower_bounds = mge_lower_bounds(
            self.SourceModel,
            self.LensLightModel,
            kwargs_source,
            kwargs_lens_light,
            num_param=num_param,
        )

        return np.isfinite(lower_bounds)

    @partial(jit, static_argnums=(0, 7))
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

        bounded = self._bounded_mask(
            kwargs_source_i, kwargs_lens_light_i, num_param=A.shape[0]
        )
        param, cov_param, wls_model = get_param_bounded_WLS(
            A.T, 1.0 / C_D_response, d, bounded, inv_bool=inv_bool
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

    @partial(jit, static_argnums=(0, 7, 8, 9, 10))
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
