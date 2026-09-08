# -*- coding: utf-8 -*-
"""This module creates `fitting_kwargs_list` for `FittingSequence.fit_sequence()` with
pre-defined recipes."""

__author__ = "ajshajib"

from copy import deepcopy
from importlib.util import find_spec
from itertools import count
import numpy as np
from scipy import ndimage

# radius (arcsec) of the region kept clear at the image center in `get_arc_mask`
# when no Einstein radius guess is available
DEFAULT_CLEAR_CENTER = 0.4
# when an Einstein radius guess is available, the clear center is scaled to this
# multiple of it, so that it stays inside the arcs for small lenses as well
CLEAR_CENTER_THETA_E_FACTOR = 0.5
# ... but never smaller than this many pixels, below which the arc finder starts
# marking the deflector's own light as arc
CLEAR_CENTER_MIN_PIXELS = 2


"""Default scaling of the PSO seeding box relative to the per-parameter sigmas. 
Overridable per lens with `fitting.pso_settings.sigma_scale:`."""
DEFAULT_PSO_SIGMA_SCALE = 1.0

"""Packages required by the gradient descent optimizer, which runs through
JAXtronomy's `optax` fitting routine."""
GRADIENT_DESCENT_PACKAGES = ("jax", "jaxtronomy", "numpyro", "optax")

"""Default settings for the gradient descent optimizer, overridable per lens with
`fitting.gradient_descent_settings:`. `rng_seed` defaults to a fixed value so that a
run repeats: with `None`, JAXtronomy draws a fresh random seed every time."""
DEFAULT_GRADIENT_DESCENT_SETTINGS = {
    "maxiter": 1000,
    "num_chains": 8,
    "tolerance": 1e-6,
    "sigma_scale": 1.0,
    "rng_seed": 0,
    "warm_start": True,
    "grad_tolerance": 1e-5,
}

"""Names of the optimization stages of the galaxy-galaxy recipe, in the order they are
emitted within one epoch. Used to key per-stage optimizer settings."""
GALAXY_GALAXY_STAGES = (
    "lens_light",  # lens light only, arc mask
    "source",  # source only, science mask, shapelet beta fixed
    "lens_source",  # lens mass (minus gamma and shear) + source, beta fixed
    "lens_source_beta",  # as above, beta freed
    "all",  # everything free except gamma = 2 and the external shear
)

"""Recipe names that run the gradient descent optimizer regardless of the
`fitting.gradient_descent:` flag."""
GRADIENT_DESCENT_RECIPE_NAMES = (
    "galaxy-galaxy-gradient-descent",
    "galaxy-galaxy-pso-gradient-descent",
)

"""Per-stage overrides for the `galaxy-galaxy-gradient-descent` recipe, merged over
`DEFAULT_GRADIENT_DESCENT_SETTINGS`. Overridable per lens with
`fitting.gradient_descent_schedule.stages:`.

The first chain starts at the current parameter state and a chain that ends worse than
it started is never accepted, so `sigma_scale` controls only how far the *exploring*
chains are thrown. Early stages can therefore explore freely, since the worst case is
wasted time rather than a worse model, while the late stages -- already in the right
basin, and freeing every block at once -- refine from the warm start alone."""
DEFAULT_GRADIENT_DESCENT_STAGE_SCHEDULE = {
    "lens_light": {"sigma_scale": 1.0, "num_chains": 4},
    "source": {"sigma_scale": 1.0, "num_chains": 4},
    "lens_source": {"sigma_scale": 0.5, "num_chains": 2},
    "lens_source_beta": {"sigma_scale": 0.2, "num_chains": 1},
    "all": {"sigma_scale": 0.1, "num_chains": 1},
}

"""How the per-stage schedule is narrowed in epochs after the first, where the model is
already close and wide perturbations only waste chains."""
DEFAULT_GRADIENT_DESCENT_EPOCH_DECAY = {"sigma_scale": 0.3, "max_chains": 1}

"""Settings for the `galaxy-galaxy-pso-gradient-descent` recipe, which pairs the
particle swarm with a gradient descent inside the galaxy-galaxy staging.

The two optimizers are given the jobs they are respectively good at. A swarm is a
basin finder: it covers a stage's subspace and finds the region the optimum is in
within a few tens of iterations. What it is bad at is the last few digits, because its
spatial scale collapses geometrically and it needs a great many further evaluations to
buy each additional one. A gradient descent is the opposite: hopeless at deciding which
basin, and very fast at driving one to its exact bottom.

That gives three settings, all of which were arrived at by measurement on
`lens_system1` against the `galaxy-galaxy` recipe it has to beat. See
`GALAXY_GALAXY_RECIPE.rst` for the numbers.

**One descent, at the final stage, with several chains.** `stages` is `["all"]`, and
`all` gets `num_chains: 5`. The final stage is the only one with every parameter block
free, so a descent at any earlier stage is re-optimized by this one anyway -- and each
descent step costs a full XLA compilation, because `FittingSequence.fit_sequence`
rebuilds the likelihood and clears the JAX compilation cache before every step. On a
GPU that compilation, not the iterations, is most of what an intermediate descent
costs. The chains are what make the single descent reliable: the first warm-starts from
the swarm's answer, the rest from draws around it, and a chain that ends worse than the
state it was handed is never accepted, so extra chains can only cost time. They are
what lets the descent leave a mediocre basin -- in one measured run the swarm finished
at `logL = -4457` and the chains found `-4206` -- and they are cheap, because a
gradient now costs about twice a likelihood evaluation rather than ten times it.

**The final stage keeps its full swarm; the earlier ones are cut to a quarter.** The
`all` stage is where the swarm earns its keep, since every block is free at once and
the likelihood is most multimodal there; cutting it is what made runs land several
hundred in `logL` short. The four restricted stages are cheap to redo and their swarm
budget makes no measurable difference to the answer, so they run at
`pso_iteration_scale: 0.25`.

**One epoch, against the `galaxy-galaxy` recipe's two.** A second pass exists to let a
swarm that stopped short try again from a better starting point. Its cost is the whole
staging over again, and a multi-start descent buys the same insurance for a fraction of
it.
"""
DEFAULT_GRADIENT_DESCENT_POLISH = {
    # recipe knobs
    "stages": ["all"],
    "epochs": 1,
    "pso_iteration_scale": 0.25,
    "pso_particle_scale": 1.0,
    # optimizer settings, merged over `DEFAULT_GRADIENT_DESCENT_SETTINGS`
    "num_chains": 1,
    "sigma_scale": 0.5,
    "maxiter": 600,
    "tolerance": 1e-9,
    "warm_start": True,
}

"""Per-stage overrides for `DEFAULT_GRADIENT_DESCENT_POLISH`, overridable per lens with
`fitting.gradient_descent_schedule.polish_stages:`.

Everything that distinguishes the final `all` stage lives here: it keeps its full
swarm, gets the long descent, and gets the chains. The four restricted stages carry
only a `maxiter`, which a default run never reaches since they are not polished at all;
they are here so that adding one to `polish: stages:` gets a budget suited to a stage
that holds most parameter blocks fixed and converges in tens of iterations, rather than
the final stage's."""
DEFAULT_GRADIENT_DESCENT_POLISH_STAGES = {
    "lens_light": {"maxiter": 300},
    "source": {"maxiter": 400},
    "lens_source": {"maxiter": 600},
    "lens_source_beta": {"maxiter": 600},
    "all": {
        "pso_iteration_scale": 1.0,
        "maxiter": 3000,
        "num_chains": 5,
        "sigma_scale": 0.5,
    },
}


def recipe_uses_gradient_descent(recipe_name):
    """Whether a recipe name implies the gradient descent optimizer, independently of
    the `fitting.gradient_descent:` setting.

    :param recipe_name: recipe name passed to `Recipe.get_recipe`
    :type recipe_name: `str`
    :return: whether the recipe runs gradient descent
    :rtype: `bool`
    """
    return recipe_name in GRADIENT_DESCENT_RECIPE_NAMES


def check_gradient_descent_dependencies():
    """Check that the packages backing the gradient descent optimizer are installed.

    Only the presence of the packages is checked, they are not imported here. `jax`
    reads the `JAX_PLATFORMS` and `XLA_FLAGS` environment variables at import time, so
    importing it as a side effect of reading the settings would be surprising.

    :raises ImportError: if any of `GRADIENT_DESCENT_PACKAGES` is not installed
    :return: None
    :rtype: `None`
    """
    missing = []
    for package in GRADIENT_DESCENT_PACKAGES:
        try:
            found = find_spec(package) is not None
        except (ImportError, ValueError):
            found = False

        if not found:
            missing.append(package)

    if missing:
        raise ImportError(
            "Gradient descent requires these packages, which are not installed: "
            "{}. `fitting: gradient_descent:` runs through JAXtronomy's `optax` "
            "routine; install with `pip install jax optax numpyro` and `pip install "
            "git+https://github.com/lenstronomy/JAXtronomy.git`, or set `fitting: "
            "gradient_descent: false` in the settings file.".format(", ".join(missing))
        )


class Recipe(object):
    """This class contains methods to create fitting recipes.

    It builds an optimization workflow (using either particle-swarm optimization or
    gradient descent) to first find a good enough lens model within the total parameter
    space. Then, the sampling can be done starting from the neighborhood of this point.
    """

    def __init__(self, config, thread_count=1):
        """Initiate the class from the given settings for a lens system.

        :param config: `ModelConfig` instance
        :type config: `ModelConfig`
        :param thread_count: number of threads if `multiprocess` is used
        :type thread_count: `int`
        """
        self._config = config
        try:
            config.settings["fitting"]["pso"]
        except (NameError, KeyError):
            self.do_pso = False
        else:
            self.do_pso = deepcopy(config.settings["fitting"]["pso"])

            if self.do_pso is None:
                self.do_pso = False

        if self.do_pso:
            self._pso_num_particle = self._config.settings["fitting"]["pso_settings"][
                "num_particle"
            ]
            self._pso_num_iteration = self._config.settings["fitting"]["pso_settings"][
                "num_iteration"
            ]
            self._pso_sigma_scale = self._config.settings["fitting"][
                "pso_settings"
            ].get("sigma_scale", DEFAULT_PSO_SIGMA_SCALE)

        try:
            config.settings["fitting"]["gradient_descent"]
        except (NameError, KeyError):
            self.do_gradient_descent = False
        else:
            self.do_gradient_descent = deepcopy(
                config.settings["fitting"]["gradient_descent"]
            )

            if self.do_gradient_descent is None:
                self.do_gradient_descent = False

        # the settings are parsed whatever the flag says, because gradient descent can
        # also be requested by recipe name, which is not known until `get_recipe` runs.
        # A `gradient_descent_settings:` key left blank in the yaml loads as `None`.
        gradient_descent_settings = (
            self._config.settings["fitting"].get("gradient_descent_settings") or {}
        )
        unknown = set(gradient_descent_settings) - set(
            DEFAULT_GRADIENT_DESCENT_SETTINGS
        )
        if unknown:
            raise ValueError(
                "Unknown key(s) in `fitting: gradient_descent_settings:`: {}. "
                "Supported: {}.".format(
                    ", ".join(sorted(unknown)),
                    ", ".join(sorted(DEFAULT_GRADIENT_DESCENT_SETTINGS)),
                )
            )
        self._gradient_descent_settings = {
            **DEFAULT_GRADIENT_DESCENT_SETTINGS,
            **gradient_descent_settings,
        }

        # the staging knobs of the gradient descent recipes are kept separate from the
        # settings above, which are passed to the optimizer verbatim
        schedule = (
            self._config.settings["fitting"].get("gradient_descent_schedule") or {}
        )
        self._gradient_descent_stage_schedule = {
            stage: {
                **DEFAULT_GRADIENT_DESCENT_STAGE_SCHEDULE[stage],
                **(schedule.get("stages") or {}).get(stage, {}),
            }
            for stage in GALAXY_GALAXY_STAGES
        }
        self._gradient_descent_epoch_decay = {
            **DEFAULT_GRADIENT_DESCENT_EPOCH_DECAY,
            **(schedule.get("epoch_decay") or {}),
        }
        # Most specific wins: a built-in per-stage default overrides the built-in
        # global one, and anything the settings file says overrides both -- so
        # `polish: {maxiter: ...}` means "everywhere", as it reads.
        polish_overrides = schedule.get("polish") or {}
        polish_stage_overrides = schedule.get("polish_stages") or {}
        self._gradient_descent_polish = {
            **DEFAULT_GRADIENT_DESCENT_POLISH,
            **polish_overrides,
        }
        self._gradient_descent_polish_stages = {
            stage: {
                **DEFAULT_GRADIENT_DESCENT_POLISH,
                **DEFAULT_GRADIENT_DESCENT_POLISH_STAGES[stage],
                **polish_overrides,
                **polish_stage_overrides.get(stage, {}),
            }
            for stage in GALAXY_GALAXY_STAGES
        }

        if self.do_gradient_descent:
            check_gradient_descent_dependencies()

        try:
            config.settings["fitting"]["psf_iteration"]
        except (NameError, KeyError):
            self.reconstruct_psf = False
        else:
            self.reconstruct_psf = deepcopy(config.settings["fitting"]["psf_iteration"])

            if self.reconstruct_psf is None:
                self.reconstruct_psf = False

        try:
            config.settings["fitting"]["sampling"]
        except (NameError, KeyError):
            self.do_sampling = False
        else:
            self.do_sampling = deepcopy(config.settings["fitting"]["sampling"])

            if self.do_sampling is None:
                self.do_sampling = False

        self._thread_count = thread_count

    def _get_pso_step(self, sigma_scale=None, n_particles=None, n_iterations=None):
        """Get one PSO step for `fitting_kwargs_list`.

        :param sigma_scale: scaling of the box the swarm is seeded in relative to the
            per-parameter sigmas for this particular step. If `None`, the value
            configured in `fitting: pso_settings:` is used.
        :type sigma_scale: `float` or `None`
        :param n_particles: number of particles for this particular step. If `None`,
            the value configured in `fitting: pso_settings:` is used.
        :type n_particles: `int` or `None`
        :param n_iterations: number of iterations for this particular step. If `None`,
            the value configured in `fitting: pso_settings:` is used.
        :type n_iterations: `int` or `None`
        :return: a single fitting step, `['PSO', {...}]`
        :rtype: `list`
        """
        return [
            "PSO",
            {
                "sigma_scale": (
                    self._pso_sigma_scale if sigma_scale is None else sigma_scale
                ),
                "n_particles": (
                    self._pso_num_particle if n_particles is None else n_particles
                ),
                "n_iterations": (
                    self._pso_num_iteration if n_iterations is None else n_iterations
                ),
                "threadCount": self._thread_count,
            },
        ]

    def _get_shortened_pso_step(self, particle_scale, iteration_scale):
        """Get a PSO step whose budget is a fraction of the configured one.

        Used by the hybrid recipe, where the swarm only has to find the right basin and
        a gradient descent does the converging. Both counts are floored at 1, so a
        scale small enough to round to zero still leaves a swarm rather than removing
        the step.

        :param particle_scale: fraction of the configured particle count to use
        :type particle_scale: `float`
        :param iteration_scale: fraction of the configured iteration count to use
        :type iteration_scale: `float`
        :return: a single fitting step, `['PSO', {...}]`
        :rtype: `list`
        """
        return self._get_pso_step(
            n_particles=max(1, int(round(particle_scale * self._pso_num_particle))),
            n_iterations=max(1, int(round(iteration_scale * self._pso_num_iteration))),
        )

    def _get_gradient_descent_step(self, **overrides):
        """Get a gradient descent step for `fitting_kwargs_list`.

        With a warm start the optimizer resumes from the current parameter state rather
        than discarding it, so a recipe can run one of these per stage the way it runs
        one PSO per stage. Any of the configured settings can be overridden for a
        single step, which is how the staged recipes narrow the search as the model
        converges.

        :param overrides: settings to override for this step only, e.g. `sigma_scale`,
            `num_chains`, `maxiter`, `rng_seed`, `warm_start`, `grad_tolerance`,
            `tolerance`
        :return: a single fitting step, `['optax', {...}]`
        :rtype: `list`
        :raises ValueError: if an override is not a setting the optimizer accepts
        """
        unknown = set(overrides) - set(DEFAULT_GRADIENT_DESCENT_SETTINGS)
        if unknown:
            raise ValueError(
                "Unknown gradient descent setting(s): {}. Supported: {}.".format(
                    ", ".join(sorted(unknown)),
                    ", ".join(sorted(DEFAULT_GRADIENT_DESCENT_SETTINGS)),
                )
            )

        settings = deepcopy(self._gradient_descent_settings)
        settings.update(overrides)

        return ["optax", settings]

    def _get_gradient_descent_stage_settings(self, stage, epoch):
        """Resolve the gradient descent settings for one stage of a staged recipe.

        Because the first chain starts at the current parameter state and a chain that
        ends worse than it started is never accepted, `sigma_scale` controls only how
        far the *exploring* chains are thrown. Early stages can therefore explore
        freely: the worst case is wasted time, not a worse model. Late stages are
        already in the right basin, so a wide throw only spends chains on worse ones --
        and chains are expensive, since the optimizer runs them one after another.

        Epochs after the first start from an already-good model, so the whole schedule
        is narrowed by `epoch_decay`.

        :param stage: one of `GALAXY_GALAXY_STAGES`
        :type stage: `str`
        :param epoch: 0-based epoch index
        :type epoch: `int`
        :return: settings for `_get_gradient_descent_step`
        :rtype: `dict`
        """
        settings = dict(self._gradient_descent_stage_schedule.get(stage, {}))

        if epoch > 0:
            decay = self._gradient_descent_epoch_decay
            base = self._gradient_descent_settings
            settings["sigma_scale"] = (
                settings.get("sigma_scale", base["sigma_scale"])
                * decay["sigma_scale"] ** epoch
            )
            settings["num_chains"] = min(
                settings.get("num_chains", base["num_chains"]), decay["max_chains"]
            )

        return settings

    def _get_stage_rng_seed(self, index):
        """Get the RNG seed for the `index`-th gradient descent step of a recipe.

        Deriving every stage's seed from one configured base keeps the whole run
        reproducible from a single number while giving each stage an independent draw.
        Sharing one seed across stages would correlate the perturbations applied at
        each stage, wasting the exploration the extra chains are there to provide.

        :param index: 0-based index of the gradient descent step within the recipe
        :type index: `int`
        :return: seed for this step, or `None` if the base seed is `None`
        :rtype: `int` or `None`
        """
        base = self._gradient_descent_settings["rng_seed"]

        if base is None:
            return None

        return int(base) + index

    def get_recipe(self, kwargs_data_joint=None, recipe_name="galaxy-quasar"):
        """Get `fitting_kwargs_list` according to the requested `recipe`.

        :param kwargs_data_joint: `kwargs_data_joint` dictionary for multiple bands
        :type kwargs_data_joint: `dict` or `None`
        :param recipe_name: recipe name, one of 'galaxy-quasar', 'galaxy-galaxy',
            'galaxy-galaxy-gradient-descent', 'galaxy-galaxy-pso-gradient-descent',
            'custom', or 'skip'. The two gradient descent names run the galaxy-galaxy
            staging with a gradient descent at each stage, or with a gradient descent
            polishing each particle swarm, and require `Processor.swim(...,
            use_jax=True)`.
        :type recipe_name: `str`
        :return: fitting kwargs list
        :rtype: `list`
        """
        fitting_kwargs_list = []

        if recipe_name == "custom":
            try:
                self._config.settings["fitting_kwargs_list"]
            except (KeyError, NameError):
                raise KeyError(
                    "custom recipe_name requires fitting_kwargs_list key in yaml settings"
                )
            else:
                if self._config.settings["fitting_kwargs_list"] is not None:
                    fitting_kwargs_list += self._config.settings["fitting_kwargs_list"]
                else:
                    pass
        elif recipe_name == "galaxy-quasar":
            fitting_kwargs_list += self.get_galaxy_quasar_recipe()
        elif recipe_name == "galaxy-galaxy":
            if kwargs_data_joint is None:
                raise ValueError(
                    "kwargs_data_joint is necessary to use "
                    "galaxy-galaxy optimization recipe!"
                )
            fitting_kwargs_list += self.get_galaxy_galaxy_recipe(kwargs_data_joint)
        elif recipe_name in GRADIENT_DESCENT_RECIPE_NAMES:
            if kwargs_data_joint is None:
                raise ValueError(
                    "kwargs_data_joint is necessary to use the '{}' "
                    "optimization recipe!".format(recipe_name)
                )
            check_gradient_descent_dependencies()

            if recipe_name == "galaxy-galaxy-gradient-descent":
                fitting_kwargs_list += self.get_galaxy_galaxy_gradient_descent_recipe(
                    kwargs_data_joint
                )
            else:
                fitting_kwargs_list += (
                    self.get_galaxy_galaxy_pso_gradient_descent_recipe(
                        kwargs_data_joint
                    )
                )
        elif recipe_name == "skip":
            pass
        else:
            raise ValueError("Recipe name '{}' not recognized!!".format(recipe_name))

        fitting_kwargs_list += self.get_sampling_sequence()

        return fitting_kwargs_list

    def _get_power_law_model_index(self):
        """Get the index of the power-law model, if included in the lens model list.

        :return: index or `None`
        :rtype: `int` or `None`
        """
        lens_model_list = self._config.get_lens_model_list()

        if "SPEMD" in lens_model_list:
            index = lens_model_list.index("SPEMD")
        elif "PEMD" in lens_model_list:
            index = lens_model_list.index("PEMD")
        elif "SPEP" in lens_model_list:
            index = lens_model_list.index("SPEP")
        elif "EPL" in lens_model_list:
            index = lens_model_list.index("EPL")
        else:
            index = None

        return index

    def _get_external_shear_model_index(self):
        """Get the index of the external shear model, if included in the lens model
        list.

        :return: index or `None`
        :rtype: `int` or `None`
        """
        lens_model_list = self._config.get_lens_model_list()
        if "SHEAR_GAMMA_PSI" in lens_model_list or "SHEAR" in lens_model_list:
            if "SHEAR_GAMMA_PSI" in lens_model_list:
                index = lens_model_list.index("SHEAR_GAMMA_PSI")
            else:
                index = lens_model_list.index("SHEAR")
        else:
            index = None

        return index

    def _get_shapelet_model_index(self):
        """Get the index of the shapelets model, if included in the source model list.

        :return: index or `None`
        :rtype: `int` or `None`
        """
        source_model_list = self._config.get_source_light_model_list()
        if "SHAPELETS" in source_model_list:
            index = source_model_list.index("SHAPELETS")
        else:
            index = None

        return index

    def get_galaxy_quasar_recipe(self):
        """Get the default pre-sampling optimization routine.

        :return: fitting kwargs list
        :rtype: `list`
        """
        if self.do_gradient_descent:
            return [self._get_gradient_descent_step()]

        fitting_kwargs_list = []

        if self.do_pso:
            pso_range_multipliers = [1.0, 0.1, 0.1]

            pl_model_index = self._get_power_law_model_index()

            for epoch in range(2):
                if epoch == 0 and pl_model_index is not None:
                    fitting_kwargs_list.append(
                        [
                            "update_settings",
                            {"lens_add_fixed": [[pl_model_index, ["gamma"]]]},
                        ]
                    )
                elif pl_model_index is not None:
                    fitting_kwargs_list.append(
                        [
                            "update_settings",
                            {"lens_remove_fixed": [[pl_model_index, ["gamma"]]]},
                        ]
                    )

                for multiplier in pso_range_multipliers:
                    # if multiplier in [10., 1.]:
                    #     fitting_kwargs_list.append([
                    #         'update_settings',
                    #         {'lens_add_fixed': [[index, ['gamma']]]}
                    #
                    #     ])
                    # elif multiplier == .1:
                    #     fitting_kwargs_list.append([
                    #         'update_settings',
                    #         {'lens_remove_fixed': [[index, ['gamma']]]}
                    #     ])

                    fitting_kwargs_list.append(
                        self._get_pso_step(sigma_scale=multiplier)
                    )

                    if self.reconstruct_psf:
                        fitting_kwargs_list.append(
                            ["psf_iteration", self._config.get_kwargs_psf_iteration()]
                        )
        return fitting_kwargs_list

    def get_sampling_sequence(self):
        """Get the sampling sequence. Currently MCMC with emcee and nested sampling with
        Nautilus are supported.

        :return: a list containing the sampling sequence arguments
        :rtype: `list`
        """
        fitting_kwargs_list = []

        if self.do_sampling:
            supported_samplers = [
                "emcee",
                # "zeus",
                # "dynesty",
                # "dyPolyChord",
                # "MultiNest",
                # "nested_sampling",
                "Nautilus",
            ]
            if self._config.settings["fitting"]["sampler"] not in supported_samplers:
                raise ValueError(
                    "Sampler '{}' not supported! ".format(
                        self._config.settings["fitting"]["sampler"]
                    )
                    + "Supported ones are: {}".format(supported_samplers)
                )

            sampling_kwargs = self._config.settings["fitting"]["sampler_settings"]
            if self._config.settings["fitting"]["sampler"] in ["emcee", "Nautilus"]:
                if "threadCount" not in sampling_kwargs:
                    sampling_kwargs["threadCount"] = self._thread_count

                try:
                    self._config.settings["fitting"]["sampler_settings"]["init_samples"]
                except (NameError, KeyError):
                    pass
                else:
                    if (
                        self._config.settings["fitting"]["sampler_settings"][
                            "init_samples"
                        ]
                        is not None
                    ):
                        sampling_kwargs["init_samples"] = np.array(
                            self._config.settings["fitting"]["sampler_settings"][
                                "init_samples"
                            ]
                        )

                        sampling_kwargs["re_use_samples"] = True

            fitting_kwargs_list.append(
                [self._config.settings["fitting"]["sampler"], sampling_kwargs]
            )

        return fitting_kwargs_list

    def get_galaxy_galaxy_recipe(self, kwargs_data_joint, epochs=2):
        """Get the pre-sampling optimization routine for a galaxy-galaxy lens. PSF
        iteration is not added.

        :param kwargs_data_joint: dictionary containing joint data specifications
        :type kwargs_data_joint: `dict`
        :param epochs: number of times to repeat the fitting sequence
        :type epochs: `int`
        :return: a list containing the sequence of fitting operations
        :rtype: `list`
        """
        if self.do_gradient_descent:
            return [self._get_gradient_descent_step()]

        if not self.do_pso:
            return []

        return self._build_galaxy_galaxy_sequence(
            kwargs_data_joint,
            epochs,
            lambda stage, epoch: [self._get_pso_step()],
        )

    def get_galaxy_galaxy_gradient_descent_recipe(self, kwargs_data_joint, epochs=2):
        """Get the staged galaxy-galaxy optimization routine, run with gradient descent.

        This is the freeze/unfreeze staging of `get_galaxy_galaxy_recipe` with a
        gradient descent at each stage instead of a particle swarm. The staging matters
        just as much for a gradient descent as for a swarm: optimizing every block at
        once from a rough starting point walks into a local minimum where the source
        absorbs the lens light, and the staging is what keeps that from happening.

        The first chain of every descent starts at the current parameter state, and a
        stage never returns a state worse than the one it received, so the stages
        compose rather than each discarding the last.

        :param kwargs_data_joint: dictionary containing joint data specifications
        :type kwargs_data_joint: `dict`
        :param epochs: number of times to repeat the fitting sequence
        :type epochs: `int`
        :return: a list containing the sequence of fitting operations
        :rtype: `list`
        """
        step_count = count()

        def make_steps(stage, epoch):
            settings = self._get_gradient_descent_stage_settings(stage, epoch)
            settings["rng_seed"] = self._get_stage_rng_seed(next(step_count))
            return [self._get_gradient_descent_step(**settings)]

        return self._build_galaxy_galaxy_sequence(
            kwargs_data_joint, epochs, make_steps
        )

    def get_galaxy_galaxy_pso_gradient_descent_recipe(
        self, kwargs_data_joint, epochs=None
    ):
        """Get the staged galaxy-galaxy routine with a shortened particle swarm and a
        gradient descent at each stage.

        This is the `galaxy-galaxy` staging with each stage's optimization split
        between the two optimizers according to what each is good at: a shortened swarm
        to find the basin, and a warm-started gradient descent to converge inside it.
        See `DEFAULT_GRADIENT_DESCENT_POLISH` for why that is both a better fit and less
        wall-clock time than running the swarm alone to convergence.

        The descent warm-starts from the swarm's result and a chain that ends worse than
        it started is never accepted, so a stage can only ever hand on a state at least
        as good as the one its swarm reached.

        :param kwargs_data_joint: dictionary containing joint data specifications
        :type kwargs_data_joint: `dict`
        :param epochs: number of times to repeat the fitting sequence. If `None`, the
            value from `fitting: gradient_descent_schedule: polish: epochs:` is used.
        :type epochs: `int` or `None`
        :return: a list containing the sequence of fitting operations
        :rtype: `list`
        :raises ValueError: if the PSO is not configured
        """
        if not self.do_pso:
            raise ValueError(
                "The 'galaxy-galaxy-pso-gradient-descent' recipe runs a particle swarm "
                "at each stage, so it needs `fitting: pso: true` and "
                "`fitting: pso_settings:` in the settings file. Use "
                "'galaxy-galaxy-gradient-descent' for gradient descent alone."
            )

        if epochs is None:
            epochs = int(self._gradient_descent_polish["epochs"])

        step_count = count()

        # `stages` names which stages get a descent at all, so it is a property of the
        # recipe rather than of any one stage
        polished_stages = self._gradient_descent_polish["stages"]

        def make_steps(stage, epoch):
            polish = self._gradient_descent_polish_stages[stage]

            steps = [
                self._get_shortened_pso_step(
                    particle_scale=polish["pso_particle_scale"],
                    iteration_scale=polish["pso_iteration_scale"],
                )
            ]

            if stage in polished_stages:
                settings = {
                    key: value
                    for key, value in polish.items()
                    if key in DEFAULT_GRADIENT_DESCENT_SETTINGS
                }
                settings["rng_seed"] = self._get_stage_rng_seed(next(step_count))
                steps.append(self._get_gradient_descent_step(**settings))

            return steps

        return self._build_galaxy_galaxy_sequence(
            kwargs_data_joint, epochs, make_steps
        )

    def _build_galaxy_galaxy_sequence(self, kwargs_data_joint, epochs, make_steps):
        """Build the staged galaxy-galaxy optimization sequence.

        This holds the freeze/unfreeze staging of Shajib et al. (2021), which is
        independent of which optimizer runs at each stage. `get_galaxy_galaxy_recipe`
        and the gradient descent recipes all go through here, so the staging cannot
        drift between them.

        :param kwargs_data_joint: dictionary containing joint data specifications
        :type kwargs_data_joint: `dict`
        :param epochs: number of times to repeat the fitting sequence
        :type epochs: `int`
        :param make_steps: callable taking `(stage, epoch)`, where `stage` is one of
            `GALAXY_GALAXY_STAGES` and `epoch` is the 0-based epoch index, returning
            the list of fitting steps to run at that stage
        :type make_steps: callable
        :return: a list containing the sequence of fitting operations
        :rtype: `list`
        """
        fitting_kwargs_list = []

        arc_masks = []
        masks = self._config.get_masks()
        for i, band_item in enumerate(kwargs_data_joint["multi_band_list"]):
            mask = masks[i] if masks is not None else None
            image = band_item[0]["image_data"]
            arc_masks.append(self.get_arc_mask(image, mask=mask))

        pl_model_index = self._get_power_law_model_index()
        external_shear_model_index = self._get_external_shear_model_index()
        shapelets_index = self._get_shapelet_model_index()

        # temp_constraints = self._config.get_kwargs_constraints()
        for epoch in range(epochs):
            # first fix everything else except for lens light and use arc
            # mask to fit the lens light only. Join the centroids of lens
            # and lens light
            fitting_kwargs_list += [
                self.fix_params("lens"),
                self.fix_params("source"),
                [
                    "update_settings",
                    {"kwargs_likelihood": {"image_likelihood_mask_list": arc_masks}},
                ],
                # [
                #     "update_settings",
                #     {
                #         "kwargs_constraints": {
                #             "joint_lens_with_light": [
                #                 [0, 0, ["center_x", "center_y"]]
                #             ]
                #         }
                #     },
                # ],
            ]
            fitting_kwargs_list += make_steps("lens_light", epoch)

            # unfix the source except for beta, keep lens fixed, fix lens
            # light, use regular mask
            fitting_kwargs_list += [self.unfix_params("source")]

            # fix the shapelets beta parameter
            if shapelets_index is not None:
                fitting_kwargs_list += [
                    [
                        "update_settings",
                        {"source_add_fixed": [[shapelets_index, ["beta"], [0.1]]]},
                    ]
                ]

            fitting_kwargs_list += [
                # self.unfix_params('lens'),
                self.fix_params("lens_light"),
                [
                    "update_settings",
                    {"kwargs_likelihood": {"image_likelihood_mask_list": masks}},
                ],
            ]

            # set lens parameter values to guess values, if provided
            if "initial_guesses" in self._config.settings["lens_options"].keys():
                param_list = []
                for index, params in self._config.settings["lens_options"][
                    "initial_guesses"
                ].items():
                    param_list.append(
                        [index, list(params.keys()), list(params.values())]
                    )

                fitting_kwargs_list += [
                    ["update_settings", {"lens_add_fixed": param_list}]
                ]

            # optimize for the source only
            # self.fix_params('lens'),
            fitting_kwargs_list += make_steps("source", epoch)

            # unfix the central deflector parameters, keep beta fixed
            fitting_kwargs_list += [self.unfix_params("lens")]

            # keep the external shear pinned during the optimization, if there is one
            if external_shear_model_index is not None:
                fitting_kwargs_list += [
                    self.fix_params("lens", external_shear_model_index)
                ]

            # optimize for lens and source together, fix power-law gamma to
            # 2, as all the lens parameters are unfixed
            if pl_model_index is not None:
                fitting_kwargs_list += [
                    [
                        "update_settings",
                        {"lens_add_fixed": [[pl_model_index, ["gamma"], [2.0]]]},
                    ]
                ]

            fitting_kwargs_list += make_steps("lens_source", epoch)

            # unfix the shapelets beta parameter
            if shapelets_index is not None:
                fitting_kwargs_list += [
                    [
                        "update_settings",
                        {"source_remove_fixed": [[shapelets_index, ["beta"]]]},
                    ]
                ]

            # finally optimize with all of lens, lens light and source free
            fitting_kwargs_list += make_steps("lens_source_beta", epoch)
            fitting_kwargs_list += [self.unfix_params("lens_light")]
            fitting_kwargs_list += make_steps("all", epoch)

            # finally, relax shear parameters for MCMC later
            # disjoin lens and lens light centroids
            fitting_kwargs_list += [
                self.unfix_params("lens"),
                # ["update_settings", {"kwargs_constraints": temp_constraints}],
            ]

        # fitting_kwargs_list += self.get_default_recipe()

        return fitting_kwargs_list

    def get_theta_E_guess(self):
        """Get the initial guess for the central deflector's Einstein radius, if
        provided in the settings.

        :return: Einstein radius guess in arcsec, or `None` if not provided
        :rtype: `float` or `None`
        """
        try:
            initial_guesses = self._config.settings["lens_options"]["initial_guesses"]
            for _, params in sorted(initial_guesses.items()):
                if "theta_E" in params:
                    return float(params["theta_E"])
        except (AttributeError, KeyError, TypeError):
            pass

        return None

    def get_clear_center(self):
        """Get the radius of the central region that `get_arc_mask` keeps clear.

        The value is taken from the `mask: clear_center:` setting, if provided.
        Otherwise, it is scaled to the Einstein radius guess (capped at
        `DEFAULT_CLEAR_CENTER`, so that it can only shrink relative to the previous
        behavior, and floored at `CLEAR_CENTER_MIN_PIXELS` pixels). If no Einstein
        radius guess is provided either, it falls back to `DEFAULT_CLEAR_CENTER`.

        :return: radius of the central region to **not** mask, in arcsec
        :rtype: `float`
        """
        try:
            clear_center = self._config.settings["mask"]["clear_center"]
        except (KeyError, TypeError):
            clear_center = None

        if clear_center is not None:
            return float(clear_center)

        theta_E = self.get_theta_E_guess()

        if theta_E is None:
            return DEFAULT_CLEAR_CENTER

        return max(
            CLEAR_CENTER_MIN_PIXELS * np.max(self._config.pixel_size),
            min(DEFAULT_CLEAR_CENTER, CLEAR_CENTER_THETA_E_FACTOR * theta_E),
        )

    def get_arc_mask(self, image, clear_center=None, mask=None):
        """Create a mask for lensed galaxy arcs from the image of the lens. The lens
        galaxy is required to be close to the center (within a few pixels) of the image.

        :param image: image of the lensing system
        :type image: `numpy.ndarray`
        :param clear_center: radius of the central region to **not** mask. If `None`,
            it is taken from the settings through `get_clear_center()`, which scales
            it to the Einstein radius guess when one is provided.
        :type clear_center: `float` or `None`
        :param mask: a mask to multiply with the arc mask. If the central
            region is masked out in `mask`, then a circle with radius
            `clear_center` will be unmasked.
        :type mask: `numpy.ndarray` or `None`
        :return: mask for the lensed galaxy arcs
        :rtype: `numpy.ndarray`
        """
        if clear_center is None:
            clear_center = self.get_clear_center()

        clear_center_pixels = int(clear_center / np.max(self._config.pixel_size))

        # take x- and y- gradient of the image
        x_diff = np.diff(image, axis=1)[1:, :]
        y_diff = np.diff(image, axis=0)[:, 1:]

        w = len(x_diff) - 1
        x, y = np.meshgrid(
            np.linspace(-w / 2, w / 2, int(w + 1)),
            np.linspace(-w / 2, w / 2, int(w + 1)),
        )
        r = np.sqrt(x * x + y * y)

        # compute the radial gradient of the image
        softening = 1e-10
        radial_gradient = -(x_diff * x / (r + softening) + y_diff * y / (r + softening))

        # convert radial_gradient to binary map (+ve to 0 and -ve to 1).
        # where the arc starts when going radially outward, the gradient
        # will be +ve, so this operation marks the inner edge of the arcs
        radial_gradient[np.isnan(radial_gradient)] = 0
        radial_gradient[radial_gradient > 0] = 1
        radial_gradient[radial_gradient <= 0] = 0
        radial_gradient = 1 - radial_gradient

        # unmark any marked pixels from the central region
        radial_gradient[r < clear_center_pixels] = 0

        # remove connected regions with area less than 5 pixels to remove
        # masked regions created by noise
        structure = np.ones((3, 3))
        filtered_map = deepcopy(radial_gradient)
        id_regions, num_ids = ndimage.label(filtered_map, structure=structure)
        id_sizes = np.array(
            ndimage.sum(radial_gradient, id_regions, range(num_ids + 1))
        )
        area_mask = id_sizes < 5
        filtered_map[area_mask[id_regions]] = 0

        # dilate the binary marked-pixel map
        dilated = np.zeros_like(filtered_map)

        # create structural elements for dilating the image radially outward
        # in each of the four quadrants separately
        a2 = np.tril(np.ones((8, 8)))
        np.fill_diagonal(a2, 0)
        a2 = np.flip(a2, axis=1)  # 1's at lower than anti-diagonal
        a4 = np.flip(a2)  # 1's at upper than anti-diagonal
        a3 = np.rot90(a2)  # 1's at upper than the diagonal
        a1 = np.flip(a3)  # 1's at lower than the diagonal

        # the quadrants are taken about the deflector, which sits at the center
        split_x = int(np.searchsorted(x[0], 0, side="right"))
        split_y = int(np.searchsorted(y[:, 0], 0, side="right"))

        dilated[:split_y, :split_x] = ndimage.binary_dilation(
            filtered_map[:split_y, :split_x], a4
        )
        dilated[:split_y, split_x:] = ndimage.binary_dilation(
            filtered_map[:split_y, split_x:], a3
        )
        dilated[split_y:, :split_x] = ndimage.binary_dilation(
            filtered_map[split_y:, :split_x], a1
        )
        dilated[split_y:, split_x:] = ndimage.binary_dilation(
            filtered_map[split_y:, split_x:], a2
        )

        # increase the size by 1 along both axes to match the image size
        # the mask is the negative of the marked pixel-map
        arc_mask = 1 - np.pad(dilated, ((0, 1), (0, 1)), "minimum")

        # check for bad values
        arc_mask[arc_mask > 0] = 1
        arc_mask[arc_mask <= 0] = 0

        if mask is not None:
            arc_mask *= mask

            w = len(arc_mask) - 1
            x, y = np.meshgrid(
                np.linspace(-w / 2, w / 2, int(w + 1)),
                np.linspace(-w / 2, w / 2, int(w + 1)),
            )
            r = np.sqrt(x * x + y * y)

            arc_mask[r < clear_center_pixels] = 1

        return arc_mask

    def fix_params(self, model_component, index=None):
        """Fix all the params in `model_component` that are not fixed by settings.

        :param model_component: name of params type, e.g., 'lens', 'lens_light', or 'source'
        :type model_component: `str`
        :param index: profile indices, if `None` all will be fixed
        :type index: `list` or `int` or `None`
        :return: formatted fit-sequence code to go into `fitting_kwargs_list`
        :rtype: `list`
        """
        if model_component == "lens":
            kwargs_params = self._config.get_lens_model_params()
        # elif model_component == 'point_source':
        #    kwargs_params = self.get_point_source_params()
        elif model_component == "lens_light":
            kwargs_params = self._config.get_lens_light_model_params()
        elif model_component == "source":
            kwargs_params = self._config.get_source_light_model_params()
        else:
            raise ValueError(
                "{} not recognized! Must be lens or "
                "lens_light or source.".format(model_component)
            )

        lower_list = kwargs_params[3]
        fixed_list = kwargs_params[2]

        if index is None:
            index = [i for i, _ in enumerate(lower_list)]

        if not isinstance(index, list):
            index = [index]

        param_list_with_index = []

        for i, (sigma, fixed) in enumerate(zip(lower_list, fixed_list)):
            if i in index:
                param_list = []
                for key, value in sigma.items():
                    if key not in fixed:
                        param_list.append(key)

                param_list_with_index.append([i, param_list])

        key = "{}_add_fixed".format(model_component)

        return ["update_settings", {key: param_list_with_index}]

    def unfix_params(self, model_component, index=None):
        """Unfix all the params in `model_component` that are not fixed from settings.

        :param model_component: name of params type, e.g., 'lens', 'lens_light', or 'source'
        :type model_component: `str`
        :param index: profile indices, if `None` all will be unfixed
        :type index: `list` or `int` or `None`
        :return: formatted fit-sequence code to go into `fitting_kwargs_list`
        :rtype: `list`
        """
        code = self.fix_params(model_component, index=index)

        old_key = "{}_add_fixed".format(model_component)
        key = "{}_remove_fixed".format(model_component)

        code[1][key] = deepcopy(code[1][old_key])
        del code[1][old_key]

        return code
