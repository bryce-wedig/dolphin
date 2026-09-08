Configuration Files
===================

This document provides a detailed explanation of all the allowed options in the ``config.yaml`` files for ``dolphin``, but some of them are optional, as indicated. Check out the ``io_directory_example/settings`` `folder <https://github.com/ajshajib/dolphin/tree/main/io_directory_example/settings>`_ for some example config files.

.. contents:: Table of Contents
   :local:
   :depth: 2

Top-level information
---------------------

- ``lens_name``: The name of the lens system being modeled.

  - Type: ``string``
  - Example:

    .. code-block:: yaml

       lens_name: "DESJ0408-5354"

- ``band``: List of photometric bands used for modeling.

  - Type: ``list of strings``
  - Example:

    .. code-block:: yaml

       band: ["F475X", "F600LP"]

- ``psf_supersampled_factor``: *(Optional)* Factor by which the Point Spread Function (PSF) is supersampled. Default is 1.

  - Type: ``float``
  - Example:

    .. code-block:: yaml

       psf_supersampled_factor: 3

- ``pixel_size``: *(Optional)* Pixel size for each band. If not provided, it will be inferred from the image data.

  - Type: ``list of floats``
  - Example:

    .. code-block:: yaml

       pixel_size: [0.04, 0.04]

Model Section
-------------

- ``model``: Defines the components of the lens model.

  - Suboptions:

    - ``lens``: List of lens mass profiles. Supported models include: ``EPL``, ``SIE``, ``SIS``, ``SPEP``, ``PEMD``, ``SHEAR_GAMMA_PSI``, ``FLEXION``.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           lens: ["EPL", "SHEAR_GAMMA_PSI"]

    - ``lens_light``: List of lens light profiles. Supported models include: ``SERSIC``, ``SERSIC_ELLIPSE``, ``MGE_SET``, ``MGE_SET_ELLIPSE``, ``UNIFORM``. The list will be duplicated for each band.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           lens_light: ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"]

    - ``source_light``: List of source light profiles. Supported models include: ``SERSIC_ELLIPSE``, ``SHAPELETS``. The list will be duplicated for each band.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           source_light: ["SERSIC_ELLIPSE", "SHAPELETS"]

    - ``point_source``: *(Optional)* List of point source models. Supported models include: ``LENSED_POSITION``, ``SOURCE_POSITION``. Can be an empty list for galaxy-galaxy lenses.

      - Type: ``list of strings``
      - Example:

        .. code-block:: yaml

           point_source: ["LENSED_POSITION"]

    - ``special``: *(Optional)* String or list of special parameter types.

      - Type: ``string`` or ``list of strings``
      - Example:

        .. code-block:: yaml

           special: ["astrometric_uncertainty"]


Lens Options
------------

- ``lens_options``: Additional options for the lens model.

  - Suboptions:

    - ``centroid_init``: Initial guess for the lens centroid. This will apply to all lens model components.
      For more fine-tuned control of individual lens model positions, see ``initial_guesses`` below.

      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           centroid_init: [0.04, -0.04]
    
    - ``centroid_bound``: Half of the box width to constrain the deflector's centroid. This will apply to all lens model components.
      For more fine-tuned control of the bounds for individual lens model positions, see ``uniform_prior`` below.

      - Type: ``float``
      - Default: ``0.5``
      - Example:

        .. code-block:: yaml

           centroid_bound: 0.5

    - ``initial_guesses``: *(Optional)* Adjust ``dolphin``'s default initial lens parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           initial_guesses:
             0:
               theta_E: 1.3
               gamma: 2.0
             1:
               e1: 0.3
               e2: -0.1

    - ``gaussian_prior``: *(Optional)* Gaussian priors for lens parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           gaussian_prior:
             0: [[gamma, 2.11, 0.03], [theta_E, 1.11, 0.13]]

    - ``uniform_prior``: *(Optional)* Adjust ``dolphin``'s default lower and upper bounds for lens parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           uniform_prior:
             0: [[theta_E, 0.2, 1.4], [center_x, 0.5, 0.9]]
             1: [[gamma_ext, 0.02, 0.8]]

    - ``fix``: *(Optional)* Fix specific parameters for the lens model.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           fix:
             0:
               gamma: 2.0

    - ``limit_mass_pa_from_light``: *(Optional)* Maximum allowed difference between the position angle of the mass and light profiles.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

           limit_mass_pa_from_light: 10.0

    - ``limit_mass_q_from_light``: *(Optional)* Maximum allowed difference between the axis ratio of the mass and light profiles.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

           limit_mass_q_from_light: 0.1


Satellites Option
------------------

- ``satellites``: *(Optional)* Options for modeling satellite galaxies.

  - Suboptions:

    - ``centroid_init``: Initial guesses for the centroids of satellites.

      - Type: ``list of lists of floats``
      - Example:

        .. code-block:: yaml

           centroid_init: [[1, 1], [1.5, 1.5]]

    - ``centroid_bound``: Half of the box width to constrain the centroids of satellites.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

           centroid_bound: 0.5

    - ``is_elliptical``: Whether each satellite is elliptical.

      - Type: ``list of booleans``
      - Example:

        .. code-block:: yaml

           is_elliptical: [true, false]           
      

Lens Light Options
------------------

- ``lens_light_options``: *(Optional)* Additional options for the lens light model.

  - Suboptions:

    - ``initial_guesses``: *(Optional)* Adjust ``dolphin``'s default initial lens light parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           initial_guesses:
             0:
               R_sersic: 2.3
               n_sersic: 1
             1:
               R_sersic: 0.3
               n_sersic: 4

    - ``fix``: Fix specific parameters for the lens light profile.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           fix: {0: {"n_sersic": 4.}}

    - ``gaussian_prior``: Gaussian priors for lens light parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           gaussian_prior:
             0: 
               [[R_sersic, 0.21, 0.15]]

    - ``uniform_prior``: *(Optional)* Adjust ``dolphin``'s default lower and upper bounds for lens light parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           uniform_prior:
             0: [[R_sersic, 0.2, 7.], [n_sersic, 0.5, 4.]]

    - ``mge_config``: *(Optional)* Configuration for MGE_SET and MGE_SET_ELLIPSE light profiles. Can be used to set the number of Gaussian components.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           mge_config:
             0:
               n_comp: 20

      - Note: when an MGE profile is used, the Gaussian amplitudes are solved for with a
        non-negativity constraint, following He et al. 2024 (MNRAS 532, 2441). Without it,
        the unconstrained solver returns Gaussians with alternating large positive and
        negative amplitudes that absorb flux from the lensed source. The constraint is applied
        to the MGE amplitudes only, since the shapelet source basis needs negative
        coefficients. Pass ``use_nn_mge=False`` to ``Processor.swim()`` to use
        ``lenstronomy``'s unconstrained solver instead. This solver is not available through
        JAXtronomy, so ``use_jax=True`` is not supported for MGE lens light models.

Source Light Options
--------------------

- ``source_light_options``: *(Optional)* Additional options for the source light model.

  - Suboptions:

    - ``initial_guesses``: *(Optional)* Adjust ``dolphin``'s default initial source light parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           initial_guesses:
             0:
               R_sersic: 2.3
               n_sersic: 1
             1:
               R_sersic: 0.3
               n_sersic: 4

    - ``gaussian_prior``: Gaussian priors for source light parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           gaussian_prior:
             0: [[beta, 0.15, 0.05]]

    - ``uniform_prior``: *(Optional)* Adjust ``dolphin``'s default lower and upper bounds for source light parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           uniform_prior:
             0: [[R_sersic, 0.2, 7.], [n_sersic, 0.5, 4.]]

    - ``shapelet_scale_logarithmic_prior``: Whether to apply a logarithmic prior on the shapelet scale parameter.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           shapelet_scale_logarithmic_prior: true

    - ``n_max``: Maximum number of Shapelet profiles for each band.

      - Type: ``integer`` or ``list of integers``
      - Example:

        .. code-block:: yaml

           n_max: [2, 4]

Point Source Options
--------------------

- ``point_source_options``: *(Optional)* Options for point source models.

  - Suboptions:

    - ``ra_init``: Initial guess for point source RA positions.

      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           ra_init: [0.1, -0.1]

    - ``dec_init``: Initial guess for point source DEC positions.

      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           dec_init: [0.1, -0.1]

    - ``bound``: Bound width for searching the point source centroids.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

           bound: 0.2

    - ``gaussian_prior``: Gaussian priors for point source parameters.

      - Type: ``dictionary``
      - Example:

        .. code-block:: yaml

           gaussian_prior:
             0: [[ra_image, 0.1, 0.05]]

    - ``time_delays_measured``: Relative time delays (in days) with respect to the first point-source image. If this is provided, ``time_delays_covariance`` must also be provided to compute the time-delay likelihood, which is then added to the combined likelihood.

      - Type: ``list of floats```
      - Example:

        .. code-block:: yaml

          time_delays_measured: [3., 2., 1.]

    - ``time_delays_covariance``: Full covariance matrix of the provided time delay measurements.
  
      - Type: ``list of list of floats``
      - Example:

        .. code-block:: yaml

          time_delays_covariance: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]

Special Options
---------------

- ``special_options``: *(Optional)* Initialization of special parameters.

  - Suboptions:

    - ``delta_x_image``: Initial spread from point source centroid in the x-axis.

      - Type: ``array of floats corresponding to the number of point sources``
      - Example:

        .. code-block:: yaml
        
           delta_x_image: [0.0, 0.0]

    - ``delta_y_image``: Initial spread from point source centroid in the y-axis.

      - Type: ``array of floats corresponding to the number of point sources``
      - Example:

        .. code-block:: yaml
        
           delta_y_image: [0.0, 0.0]

    - ``delta_image_lower``: Lower bound in spread of point source centroid sampler.

      - Type: ``float``
      - Example:

        .. code-block:: yaml
        
           delta_image_lower: -0.004

    - ``delta_image_upper``: Upper bound in spread of point source centroid sampler.

      - Type: ``float``
      - Example:

        .. code-block:: yaml
        
           delta_image_upper: 0.004

    - ``cosmology``: *(Optional)* Astropy cosmology model to use for time-delay computations. Supported models: ``FlatLambdaCDM`` (default), ``LambdaCDM``, ``FlatwCDM``, ``wCDM``, ``Flatw0waCDM``, ``w0waCDM``, ``w0wzCDM``, ``Flatw0wzCDM``, ``wpwaCDM``, ``FlatwpwaCDM``.

      - Type: ``string``
      - Example:

        .. code-block:: yaml

           cosmology: "FlatLambdaCDM"

    - ``H0``: Fiducial Hubble constant in km/s/Mpc (required for all cosmologies).
  
      - Type: ``float``
      - Example:

        .. code-block:: yaml

          H0: 70.0

    - ``Om0``: Fiducial matter energy density at z=0 (required for all cosmologies).

      - Type: ``float``
      - Example:

        .. code-block:: yaml

          Om0: 0.3

    - ``Ode0``: *(Optional)* Fiducial dark energy density at z=0 (required for non-flat cosmologies like ``LambdaCDM``, ``wCDM``, ``w0waCDM``, ``w0wzCDM``, ``wpwaCDM``).

      - Type: ``float``
      - Example:

        .. code-block:: yaml

          Ode0: 0.7

    - ``w0``: *(Optional)* Dark energy equation of state parameter at z=0 (used in ``wCDM``, ``FlatwCDM``, ``w0waCDM``, ``Flatw0waCDM``, ``w0wzCDM``, ``Flatw0wzCDM``).

      - Type: ``float``
      - Example:

        .. code-block:: yaml

          w0: -1.0

    - ``wa``: *(Optional)* Dark energy equation of state parameter derivative (used in ``w0waCDM``, ``Flatw0waCDM``, ``wpwaCDM``, ``FlatwpwaCDM``).

      - Type: ``float``
      - Example:

        .. code-block:: yaml

          wa: 0.0

    - ``wz``: *(Optional)* Dark energy equation of state parameter derivative with respect to redshift (used in ``w0wzCDM``, ``Flatw0wzCDM``).

      - Type: ``float``
      - Example:

        .. code-block:: yaml

          wz: 0.1

    - ``wp``: *(Optional)* Dark energy equation of state parameter at the pivot redshift (used in ``wpwaCDM``, ``FlatwpwaCDM``).

      - Type: ``float``
      - Example:

        .. code-block:: yaml

          wp: -1.0

    - ``zp``: *(Optional)* Pivot redshift (used in ``wpwaCDM``, ``FlatwpwaCDM``).

      - Type: ``float``
      - Example:

        .. code-block:: yaml

          zp: 0.5

    - ``Tcmb0``: *(Optional)* Temperature of the CMB at z=0 in Kelvin.

      - Type: ``float``
      - Example:

        .. code-block:: yaml

          Tcmb0: 2.725

    - ``general_scaling``: Scale specific model parameters together across multiple profiles. Input should be a dictionary mapping parameter names to the masks defining which lens models are scaled together.

      - Type: ``dictionary``
      - Example: Scaling ``theta_E`` for the second and third mass profiles together, but not the first.

        .. code-block:: yaml

          general_scaling:
            theta_E:
              [False, 1, 1]

    - ``{param_name}_scale_factor``: The scaling relation factor for the specified `param_name` across multiple profiles. Instead of individually sampling the connected parameters, only this scale factor will be sampled, and all the connected parameters will be scaled based on the same factor.
  
      - Type: ``list``
      - Example:

        .. code-block:: yaml

          theta_E_scale_factor: [1]

    - ``{param_name}_scale_factor_sigma``: Initial paramater spread relative to ``{param_name}_scale_factor``.

      - Type: ``list``
      - Example:

        .. code-block:: yaml

          theta_E_scale_factor_sigma: [0.05]

    - ``{param_name}_scale_pow``: Power-law scaling factor for the specified `param_name` scaling.

      - Type: ``list``
      - Example:

        .. code-block:: yaml

          theta_E_scale_pow: [1]

    - Combining all of the above, a specified parameter, :math:`p`, is scaled from the sampled scale factor, :math:`f_p`, and sampled power-law index, :math:`\alpha_p`, as:

      .. math::

        p \rightarrow p_{\mathrm{scaled}} =
        f_p \, p^{\alpha_p}

Numeric Options
---------------

- ``numeric_options``: Numerical settings for the modeling process.

  - Suboptions:

    - ``supersampling_factor``: Supersampling factor for each band.

      - Type: ``list of integers``
      - Example:

        .. code-block:: yaml

           numeric_options:
             supersampling_factor: [2]

Fitting Options
---------------

- ``fitting``: Settings for the fitting process.

  - Suboptions:

    - ``pso``: Whether to use Particle Swarm Optimization (PSO) for fitting.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           pso: true

    - ``pso_settings``: Settings for the PSO algorithm.

      - Suboptions:

        - ``num_particle``: Number of particles in the swarm.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_particle: 50

        - ``num_iteration``: Number of iterations for PSO.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_iteration: 50

        - ``sigma_scale``: *(Optional)* Scaling of the box the particle swarm is
          seeded in, relative to the per-parameter ``sigma`` values. The swarm starts
          uniformly within ``init +/- sigma * sigma_scale`` (clipped to the hard bounds),
          and the swarm's spatial scale never grows beyond that box, so this controls
          how much of the allowed parameter space the optimization can reach. Used by
          the ``galaxy-galaxy`` recipe; the ``galaxy-quasar`` recipe sets its own
          per-stage scaling. Defaults to ``1.0``.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               sigma_scale: 1.0

    - ``gradient_descent``: *(Optional)* Whether to use gradient descent instead of
      PSO for the pre-sampling optimization. When turned on, the whole staged
      optimization sequence of the ``galaxy-quasar`` and ``galaxy-galaxy`` recipes is
      replaced by a **single** gradient descent over all free parameters. If both
      ``pso`` and ``gradient_descent`` are ``true``, gradient descent takes precedence.

      This flag is the original, unstaged way to run gradient descent, and optimizing
      every parameter block at once from a rough starting point tends to walk into a
      local minimum where the source absorbs the lens light. Two staged recipe names
      are preferred; both imply gradient descent regardless of this flag, and take
      precedence over it:

      - ``recipe_name="galaxy-galaxy-pso-gradient-descent"`` runs the ``galaxy-galaxy``
        staging with a shortened particle swarm at each stage and a multi-start
        gradient descent at the end of it. **This is the recommended way to run
        gradient descent**, and on the systems it was measured on it is the better fit
        in about half the wall clock time. The swarm chooses the basin, which is what a
        swarm is good at, and the descent converges it, which is what a swarm is bad
        at: it stops at the resolution its collapsing spatial scale allows. On the
        single-band galaxy-galaxy lens the three recipes were measured on, over runs
        repeated because the swarm is not seeded, it reached ``logL = -4901.9`` in
        113 s for a single Sersic source against the PSO recipe's ``-4910.7`` in
        209 s, beating it in every run. For a
        Sersic-plus-shapelets source it took 433 s against 906 s and reached a better
        optimum at every rank from best run to worst, though both recipes land in one
        of two nearby modes at about the same rate. See ``GALAXY_GALAXY_RECIPE.rst``
        for the full numbers.

      - ``recipe_name="galaxy-galaxy-gradient-descent"`` runs the same staging with a
        gradient descent replacing each swarm. It is fully reproducible and no longer
        diverges, but on that same system it reached only ``logL = -4940``, worse than
        either of the above, and it is the slowest of the three. That is the expected
        result rather than a defect: it has no global search at all, so each stage
        converges whichever basin its draws happen to land in. Prefer the hybrid above
        unless you are deliberately investigating gradient descent on its own.

      Gradient descent uses JAXtronomy's Optax L-BFGS minimizer, which differentiates
      the likelihood with JAX. It therefore requires ``Processor.swim(...,
      use_jax=True)`` and the ``jax``, ``jaxtronomy``, ``optax`` and ``numpyro``
      packages, which are not installed with ``dolphin``. A helpful exception is
      raised if any of them are missing. Defaults to ``false``.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           gradient_descent: true

    - ``gradient_descent_settings``: *(Optional)* Settings for the gradient descent
      optimizer. For full documentation of the parameters, refer to the ``optax``
      method of `JAXtronomy's FittingSequence
      <https://github.com/lenstronomy/JAXtronomy>`_.

      - Suboptions:

        - ``maxiter``: Maximum number of gradient descent iterations per chain.
          Defaults to ``1000``.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               maxiter: 2000

        - ``num_chains``: Number of minimization chains to run. Each chain starts
          from a different point drawn from the prior distribution, so running more
          chains costs more time but helps avoid local minima. Chains run
          sequentially, not in parallel, so cost scales roughly linearly with this
          value. Defaults to ``8``.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_chains: 8

        - ``tolerance``: Relative convergence tolerance. A chain stops when a step's
          decrease in loss (``-logL``) is smaller than ``tolerance`` times the loss's
          magnitude, and did not make the loss worse, three times in a row. Being
          relative to the loss magnitude rather than an absolute threshold, the same
          value behaves consistently across datasets of different sizes. Defaults to
          ``1e-6``.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               tolerance: 1e-6

        - ``grad_tolerance``: Gradient-norm convergence tolerance. A chain stops once
          the largest component of the gradient falls below it. This is the standard
          L-BFGS stopping test and asks the question that matters, whether the chain
          has reached a stationary point, so it is the criterion to tune rather than
          ``tolerance``. ``0`` disables it. Defaults to ``1e-5``.

          The right value depends on the dataset. Read the per-chain diagnostics
          recorded in the output file to tune it: a stage that stops after a handful of
          iterations with a large final gradient norm needs a smaller value.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               grad_tolerance: 1e-5

        - ``sigma_scale``: Scaling of the distribution the starting points are drawn
          from, relative to the per-parameter ``sigma`` values. Defaults to ``1.0``.
          With ``warm_start`` on, this affects only the *drawn* chains, not the first
          one, so it controls how widely the optimizer explores around the current fit.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               sigma_scale: 1.0

        - ``warm_start``: Whether the first chain starts at the current parameter state
          instead of at a random draw. This is what lets a descent refine the fitting
          sequence's current fit rather than discarding it, so that the staged recipes
          compose. A chain that ends worse than the state it started from is never
          accepted, so a stage can never leave the model worse than it found it.
          Defaults to ``true``.

          - Type: ``boolean``
          - Example:

            .. code-block:: yaml

               warm_start: true

        - ``rng_seed``: Seed used to draw the starting points of the chains after the
          first. Defaults to ``0``, so that a run repeats. Each gradient descent step of
          a staged recipe gets its own seed, derived as ``rng_seed + step index``, so
          the whole run is reproducible from this one number while the stages stay
          independent. Set it to ``null`` to draw a fresh random seed every run, which
          makes the run **not** reproducible.

          - Type: ``integer`` or ``null``
          - Example:

            .. code-block:: yaml

               rng_seed: 0

    - ``gradient_descent_schedule``: *(Optional)* Staging knobs for the
      ``galaxy-galaxy-gradient-descent`` and ``galaxy-galaxy-pso-gradient-descent``
      recipes. These shape the recipe rather than the optimizer, which is why they are
      separate from ``gradient_descent_settings``.

      - Suboptions:

        - ``stages``: Per-stage overrides of ``gradient_descent_settings``, keyed by
          stage name: ``lens_light``, ``source``, ``lens_source``,
          ``lens_source_beta`` and ``all``, in the order they run within an epoch.

          Because the first chain starts at the current parameter state and a chain
          that ends worse is never accepted, ``sigma_scale`` controls only how far the
          exploring chains are thrown. The default schedule therefore explores freely
          in the early stages, where the worst case is wasted time rather than a worse
          model, and refines from the warm start alone in the late stages, which are
          already in the right basin and free every block at once:

          .. code-block:: yaml

             stages:
               lens_light: {sigma_scale: 1.0, num_chains: 4}
               source: {sigma_scale: 1.0, num_chains: 4}
               lens_source: {sigma_scale: 0.5, num_chains: 2}
               lens_source_beta: {sigma_scale: 0.2, num_chains: 1}
               all: {sigma_scale: 0.1, num_chains: 1}

        - ``epoch_decay``: How the schedule is narrowed in epochs after the first,
          which start from an already-good model. ``sigma_scale`` is multiplied by this
          factor per epoch and ``num_chains`` is capped at ``max_chains``. Defaults to
          ``{sigma_scale: 0.3, max_chains: 1}``, so later epochs are fully
          deterministic.

        - ``polish``: Settings for the ``galaxy-galaxy-pso-gradient-descent`` recipe,
          which runs a particle swarm at each stage of the galaxy-galaxy staging and a
          gradient descent at the end of it. Alongside the optimizer settings
          (``maxiter``, ``num_chains``, ``tolerance``, ``sigma_scale``,
          ``warm_start``) it takes three knobs that shape the recipe:

          - ``stages``: which stages get a descent. Defaults to ``[all]``, one descent
            after the final stage. Naming all five runs a descent after every swarm.
          - ``epochs``: how many times the five-stage staging repeats. Defaults to 1,
            against the ``galaxy-galaxy`` recipe's 2.
          - ``pso_iteration_scale`` and ``pso_particle_scale``: the fraction of the
            iteration and particle counts configured in ``pso_settings`` that each
            swarm actually runs. Defaults to 0.25 of the iterations at the full
            particle count for the four restricted stages, and the whole budget for
            the final ``all`` stage. Both counts are floored at 1.

          The division of labour is that the swarm chooses the basin and the descent
          converges it. A swarm finds the right region within a few tens of iterations
          and then buys further digits very slowly, because its spatial scale collapses
          geometrically; a warm-started gradient descent buys those digits for a small
          fraction of the cost. So the restricted stages, whose swarm budget makes no
          measurable difference to the answer, are cut to a quarter, while the final
          ``all`` stage -- the only one with every parameter block free, and the one
          that decides which basin the run ends in -- keeps its whole swarm.

          The defaults were tuned by measurement for both a single Sersic source and
          a Sersic-plus-shapelets source; see ``GALAXY_GALAXY_RECIPE.rst`` for the
          measured system and the numbers. Three results are worth knowing
          before changing them:

          - ``num_chains: 5`` on the final descent is what makes it reliable, not just
            accurate. The first chain warm-starts from the swarm's answer and the rest
            from draws around it, and a chain that ends worse than the state it was
            handed is never accepted, so extra chains can only cost time. With a single
            chain the shapelets source sometimes finished several hundred in ``logL``
            short; with five it did not.
          - Adding descents at the earlier stages does not improve the fit and costs
            real time, because ``FittingSequence.fit_sequence`` rebuilds the likelihood
            and clears the JAX compilation cache before every step, so each extra
            descent pays a full XLA compilation. Worse, converging a restricted stage
            too well can trap the model: a later short swarm, seeded at that optimum,
            cannot leave it.
          - Cutting the *final* stage's swarm is what does the damage. Cutting the
            restricted stages' swarms further, or keeping more of them, both landed
            within the run-to-run scatter.

        - ``polish_stages``: Per-stage overrides of ``polish``, keyed by the same five
          stage names as ``stages`` above. Everything that distinguishes the final
          stage lives here:

          .. code-block:: yaml

             polish_stages:
               lens_light: {maxiter: 300}
               source: {maxiter: 400}
               lens_source: {maxiter: 600}
               lens_source_beta: {maxiter: 600}
               all: {pso_iteration_scale: 1.0, maxiter: 3000, num_chains: 5, sigma_scale: 0.5}

          The four ``maxiter`` values above are never reached in a default run, since
          those stages are not polished at all. They are there so that adding one to
          ``polish: stages:`` gets a budget suited to a stage that holds most parameter
          blocks fixed and converges in tens of iterations.

          Anything set under ``polish`` applies to every stage and overrides these
          built-in per-stage defaults; ``polish_stages`` overrides both.

      - Example:

        .. code-block:: yaml

           gradient_descent_schedule:
             stages:
               all: {sigma_scale: 0.05, num_chains: 2}
             epoch_decay: {sigma_scale: 0.3, max_chains: 1}
             polish: {epochs: 1, pso_iteration_scale: 0.25}
             polish_stages:
               all: {num_chains: 8}

    - ``sampling``: *(Optional)* Whether to perform sampling after optimization.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           sampling: true

    - ``sampler``: The sampler to use for sampling. Supported samplers are ``emcee`` and ``Nautilus``.

      - Type: ``string``
      - Example:

        .. code-block:: yaml

           sampler: emcee

    - ``sampler_settings``: Settings for the sampler. For full documentation of parameters, refer to the `lenstronomy sampler documentation <https://lenstronomy.readthedocs.io/en/latest/lenstronomy.Sampling.Samplers.html>`_ and the `Nautilus documentation <https://nautilus-sampler.readthedocs.io>`_.

      - Suboptions:

        **Recommended Nautilus options:**

        - ``n_live``: Number of live points used by the nested sampler.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               n_live: 2000

        - ``n_eff``: Minimum targeted effective sample size. The algorithm will sample from the shells until this is reached.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               n_eff: 10000.0

        - ``verbose``: If True, prints detailed information about the sampler's progress.

          - Type: ``boolean``
          - Example:

            .. code-block:: yaml

               verbose: true

        - ``pass_dict``: Whether to pass dictionaries to the likelihood. Usually set to ``false`` for better performance.

          - Type: ``boolean``
          - Example:

            .. code-block:: yaml

               pass_dict: false

        - ``filepath``: Path to a file (``.h5`` or ``.hdf5``) where Nautilus will save its internal state for checkpointing.

          - Type: ``string``
          - Example:

            .. code-block:: yaml

               filepath: "nautilus_checkpoint.hdf5"

        - ``resume``: Whether to resume a run from the file specified in ``filepath``.

          - Type: ``boolean``
          - Example:

            .. code-block:: yaml

               resume: true

        **Emcee-specific options:**

        - ``n_burn``: Number of burn-in steps.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               n_burn: 2

        - ``n_run``: Number of sampling steps.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               n_run: 2

        - ``walkerRatio``: Ratio of walkers to parameters.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               walkerRatio: 2
    
        - ``init_samples``: *(Optional)* Initial samples for walkers.
          
          - Type: ``list of lists of floats``

        - ``n_walkers``: *(Optional)* Number of walkers of emcee. If set, this overwrites the ``walkerRatio`` input.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               n_walkers: 100

        - ``sigma_scale``: *(Optional)* Scaling of the initial parameter spread relative to the width in the initial settings. Default is 1.0.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               sigma_scale: 1.0

        - ``threadCount``: *(Optional)* Number of CPU threads to use for multiprocessing.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               threadCount: 4

        - ``re_use_samples``: *(Optional)* If True and ``init_samples`` is provided, re-uses the samples described.

          - Type: ``boolean``

        - ``progress``: *(Optional)* If True, shows the progress bar during sampling.

          - Type: ``boolean``

        - ``backend_filename``: *(Optional)* Name of the HDF5 file where the emcee sampling state is saved for checkpointing.

          - Type: ``string``
          - Example:

            .. code-block:: yaml

               backend_filename: "mcmc_emcee_checkpoint.h5"

        - ``start_from_backend``: *(Optional)* If True, start sampling from the state saved in ``backend_filename``.

          - Type: ``boolean``

    - ``psf_iteration``: *(Optional)* Whether to perform iterative PSF fitting.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           psf_iteration: true

    - ``psf_iteration_settings``: Settings for iterative PSF fitting.

      - Suboptions:

        - ``stacking_method``: Method for stacking PSFs.

          - Type: ``string``
          - Example:

            .. code-block:: yaml

               stacking_method: "median"

        - ``num_iter``: Number of PSF iterations.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               num_iter: 20

        - ``psf_iter_factor``: Factor for PSF iteration.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               psf_iter_factor: 0.5

        - ``keep_psf_variance_map``: Whether to keep the PSF variance map.

          - Type: ``boolean``
          - Example:

            .. code-block:: yaml

               keep_psf_variance_map: true

        - ``psf_symmetry``: Symmetry of the PSF.

          - Type: ``integer``
          - Example:

            .. code-block:: yaml

               psf_symmetry: 4

        - ``block_center_neighbour``: Block center neighbour factor.

          - Type: ``float``
          - Example:

            .. code-block:: yaml

               block_center_neighbour: 0.0

- ``fitting_kwargs_list``: *(Optional)* User-provided list of fitting sequences to bypass the automated recipes in dolphin.
In order to use this list, the ``Processor.swim()`` method must be called with ``recipe_name="custom"``.

  - Type: ``list``
  - Example:

    .. code-block:: yaml

       fitting_kwargs_list:
         - ['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 50}]

Lenstronomy Arbitrary Keyword Arguments
---------------------------------------
Model Options
-------------

- ``kwargs_model``: *(Optional)* Pass any arbitrary arguments strictly allowed in `lenstronomy.LensModel`, `lenstronomy.LightModel` inside this section.
- ``model_options`` (or ``kwargs_model``): *(Optional)* Pass any arbitrary arguments strictly allowed in `lenstronomy.Util.class_creator.create_class_instances()` inside this section.

- ``kwargs_constraints``: *(Optional)* Pass any arbitrary constraints strictly allowed in `lenstronomy.Workflow.fitting_sequence` inside this section.
Constraints Options
-------------------

- ``constraints_options`` (or ``kwargs_constraints``): *(Optional)* Pass any arbitrary constraints strictly allowed in `lenstronomy.Sampling.parameters.Param()` inside this section.

  - Example:

    .. code-block:: yaml

       kwargs_constraints:
       constraints_options:
         joint_lens_with_light: [[0, 0, ['center_x', 'center_y']]]

Likelihood Options
------------------

- ``likelihood_options`` (or ``kwargs_likelihood``): *(Optional)* Pass any arbitrary likelihood arguments strictly allowed in `lenstronomy.Sampling.likelihood.Likelihood()` inside this section.

Mask Options
------------

- ``mask``: *(Optional)* Settings for masking regions of the image.

  - Suboptions:

    - ``provided``: Set to `true` to use custom `.npy` mask files from the `settings/masks/` directory instead of using analytical masking below.

      - Type: ``boolean``
      - Example:

        .. code-block:: yaml

           provided: false

    - ``centroid_offset``: Offset for the centroid of the mask.

      - Type: ``list of lists of floats``
      - Example:

        .. code-block:: yaml

           centroid_offset: [[0.0, 0.0], [0.0, 0.0]]

    - ``mask_edge_pixels``: Number of edge pixels to mask.

      - Type: ``list of integers``
      - Example:

        .. code-block:: yaml

           mask_edge_pixels: [0, 2]

    - ``radius``: Radius of the mask for each band.

      - Type: ``list of floats``
      - Example:

        .. code-block:: yaml

           radius: [20.0, 20.0]

    - ``a``, ``b``, ``angle``: Elliptical mask parameters for each band. Used when ``radius`` is not provided.
    
      - Type: ``list of floats``
      - Example:
      
        .. code-block:: yaml
        
           a: [10.0, 10.0]
           b: [5.0, 5.0]
           angle: [0.0, 0.0]

    - ``extra_regions``: List of circular regions to mask additionally. Format is ``[ra, dec, radius]``.

      - Type: ``list of lists of lists of floats``
      - Example:

        .. code-block:: yaml

           extra_regions:
             - [[1.0, -1.0, 0.5]]
