# -*- coding: utf-8 -*-
"""Tests for MGE_SET_ELLIPSE lens light fitting."""

from pathlib import Path

from lenstronomy.Workflow.fitting_sequence import FittingSequence

from dolphin.processor.core import Processor
from dolphin.processor.config import ModelConfig
from dolphin.processor.recipe import Recipe

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestMGEFitting(object):
    def setup_method(self):
        self.processor = Processor(_TEST_IO_DIR)
        self.config = self.processor.get_lens_config("lens_system_mge")

    def _build_fitting_sequence(self):
        """Build a FittingSequence for the MGE system."""
        kwargs_data_joint = self.processor.get_kwargs_data_joint(
            "lens_system_mge",
            psf_supersampled_factor=self.config.get_psf_supersampled_factor(),
        )
        fitting_sequence = FittingSequence(
            kwargs_data_joint,
            self.config.get_kwargs_model(),
            self.config.get_kwargs_constraints(),
            self.config.get_kwargs_likelihood(),
            self.config.get_kwargs_params(),
        )
        return fitting_sequence, kwargs_data_joint

    def test_mge_initial_likelihood(self):
        """Test that the MGE model at initial parameters produces reasonable
        likelihood (not a blank model)."""
        fitting_sequence, _ = self._build_fitting_sequence()

        logL = fitting_sequence.best_fit_likelihood()
        num_data = fitting_sequence.likelihoodModule.num_data
        reduced_chi2 = -2.0 * logL / num_data

        print(f"Initial logL: {logL}")
        print(f"num_data: {num_data}")
        print(f"Initial reduced chi^2: {reduced_chi2}")

        assert reduced_chi2 < 2.0, (
            f"Initial reduced chi^2 = {reduced_chi2:.2f} (need < 2). "
            f"The MGE default parameters likely produce a blank model."
        )

    def test_mge_pso_convergence(self):
        """Test that PSO with MGE_SET_ELLIPSE converges to reduced chi^2 < 2."""
        fitting_sequence, kwargs_data_joint = self._build_fitting_sequence()
        recipe = Recipe(self.config)
        fitting_kwargs_list = recipe.get_recipe(
            kwargs_data_joint=kwargs_data_joint, recipe_name="galaxy-quasar"
        )

        fitting_sequence.fit_sequence(fitting_kwargs_list)

        logL = fitting_sequence.best_fit_likelihood()
        num_data = fitting_sequence.likelihoodModule.num_data
        reduced_chi2 = -2.0 * logL / num_data

        print(f"Best fit logL: {logL}")
        print(f"num_data: {num_data}")
        print(f"Best fit reduced chi^2: {reduced_chi2}")

        assert reduced_chi2 < 2.0, (
            f"Best fit reduced chi^2 = {reduced_chi2:.2f} (need < 2). "
            f"MGE PSO fitting did not converge to an acceptable model."
        )
        