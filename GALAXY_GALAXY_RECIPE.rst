The ``galaxy-galaxy`` Fitting Recipe
====================================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

``dolphin`` optimizes a lens model before sampling it. That pre-sampling
optimization is not a single run of the optimizer; it is a *recipe*: an ordered
list of operations handed to ``lenstronomy``'s
``FittingSequence.fit_sequence()``, in which subsets of the model parameters are
released and re-frozen between successive particle-swarm optimization (PSO)
runs.

The rationale is given in the ``dolphin`` paper (A. J. Shajib et al. 2025,
Section 2.2):

    "Although a single PSO optimization can, in principle, provide a solution
    after a sufficient number of iterations, the convergence speed of the
    optimizer can be improved by using a fitting recipe that guides the
    optimization process. For example, instead of all the model parameters
    being free at the beginning, the optimization can be iteratively guided
    toward convergence by allowing subsets of the parameters to be free while
    keeping the others fixed at assumed or previously optimized values (from a
    previous iteration). The fitting recipe is a set of instructions that guides
    the optimization process to a stable solution."

and, on this specific recipe:

    "In addition to the ``"galaxy-quasar"`` fitting recipe, a purpose-built
    ``"galaxy-galaxy"`` fitting recipe is also provided for galaxy–galaxy
    lensing systems (A. J. Shajib et al. 2021; C. Y. Tan et al. 2024)."

The ``galaxy-galaxy`` recipe implemented in
`dolphin/processor/recipe.py <dolphin/processor/recipe.py>`_ is the recipe
published in A. J. Shajib et al. (2021), *MNRAS* **503**, 2380, Section 3.2 —
the analysis of 23 SLACS galaxy–galaxy lenses in which, as that paper puts it,
"We package our modelling code into the DOLPHIN pipeline, which is a wrapper for
LENSTRONOMY to uniformly model large lens samples." Everything below maps the
code to that published recipe step by step, and flags the places where the code
has since diverged from it.

Invoke it with::

    from dolphin.processor import Processor

    processor = Processor(io_directory_path)
    processor.swim(
        lens_name="system_name",
        model_id="my_model",
        recipe_name="galaxy-galaxy",
    )

Unlike ``"galaxy-quasar"``, this recipe never inserts a ``psf_iteration`` step
(``Recipe.get_galaxy_galaxy_recipe`` docstring: "PSF iteration is not added."),
because iterative PSF reconstruction is a galaxy–quasar need:

    "An iterative fitting recipe is also necessary when the PSF is required to
    be iteratively constructed as done for galaxy–quasar systems"
    (A. J. Shajib et al. 2025, Section 2.2).

Consequently ``fitting: psf_iteration:`` in the config is ignored by this
recipe.


Why the model is optimized before it is sampled
-----------------------------------------------

    "We obtain the posterior probability distributions for our model parameters
    using the Markov chain Monte Carlo (MCMC) method. If the MCMC sampling is
    started from a point close to the maxima of the posterior, the chain can
    converge with relatively less computational time. Therefore, we first
    optimize the lens model to get a point close to the maxima of the posterior.
    We use the particle swarm optimization (PSO) method for this step (Kennedy &
    Eberhart 1995)."
    — A. J. Shajib et al. (2021), Section 3.2

The recipe therefore produces only PSO steps. The sampler is appended
afterwards by ``Recipe.get_sampling_sequence``, so the full
``fitting_kwargs_list`` returned by ``Recipe.get_recipe`` is
``<galaxy-galaxy PSO sequence> + <sampler>``.


Model components the recipe expects
------------------------------------

The recipe locates profiles by name rather than by position, so it adapts to the
``model:`` block of the config. The published model is (Section 3.1):

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Component
     - Shajib et al. (2021)
     - ``dolphin`` config equivalent
   * - Deflector mass
     - "We adopt the power-law ellipsoidal mass distribution (PEMD) for the
       deflector (Barkana 1998)."
     - ``model: lens:`` — one of ``SPEMD``, ``PEMD``, ``SPEP``, ``EPL``, found
       by ``Recipe._get_power_law_model_index()``. Returns ``None`` if none is
       present, and the ``gamma``-fixing step is then skipped.
   * - External shear
     - "We also adopt an external shear profile parametrized with the shear
       magnitude :math:`\gamma_{\rm ext}` and the shear angle
       :math:`\phi_{\rm ext}`."
     - ``SHEAR_GAMMA_PSI`` or ``SHEAR``, found by
       ``Recipe._get_external_shear_model_index()``. Returns ``None`` if absent,
       and the shear-pinning step is skipped.
   * - Deflector light
     - "We adopt a double Sérsic profile for the deflector's light
       distribution, as a single Sérsic profile leaves significant residual at
       the galaxy's centre"
     - ``model: lens_light:``. The recipe treats the whole list as one block;
       it does not require two profiles.
   * - Source light
     - "We reconstruct the source galaxy's light distribution with a basis of
       shapelets and a Sérsic profile (Refregier 2003; Birrer et al. 2015)."
     - ``model: source_light: [SERSIC_ELLIPSE, SHAPELETS]``. The shapelet
       profile is found by ``Recipe._get_shapelet_model_index()``; if absent,
       the ``beta`` steps are skipped.

The shapelet order is a config setting, not a recipe setting:

    "The order parameter :math:`n_{\max}` determines the number of shapelets as
    :math:`N_{\rm shapelets} = (n_{\max} + 1)(n_{\max} + 2)/2`. The scale size of
    the shapelets is ruled by the scaling parameter :math:`\beta`."

and "We set :math:`n_{\max}` = 6 for most of the lens systems in our sample" —
which is ``source_light_options: n_max: [6]``. ``n_max`` is placed in the
``fixed`` dictionary by ``ModelConfig.get_source_light_model_params()``, so the
recipe's fix/unfix helpers skip it: a parameter that is fixed by the config is
never released by the recipe.

The same holds for the linear amplitudes. ``amp`` appears in the ``init``
dictionaries but not in the bound dictionaries that
``Recipe.fix_params`` iterates over, so amplitudes are never frozen — they
are solved analytically by ``lenstronomy`` at every step of the sequence.


Step 0: the constraint that holds for the whole run
----------------------------------------------------

Published step (1):

    "Join the deflector mass and light centroids and fix the logarithmic slope
    :math:`\gamma = 2` and shear magnitude :math:`\gamma_{\rm ext} = 0` for all
    the steps below."

The centroid join is *not* emitted by the recipe. It is a standing constraint
built by ``ModelConfig.get_joint_lens_with_light()``, which starts every model
with ``[[0, 0, ["center_x", "center_y"]]]`` — the main deflector's mass centroid
tied to its light centroid — and is passed to ``FittingSequence`` through
``kwargs_constraints`` before the recipe ever runs. ``recipe.py`` contains a
commented-out block that would have set the same constraint per step; it is
redundant with the config default.

The :math:`\gamma = 2` and shear halves of step (1) *are* emitted, but as part
of the per-epoch sequence — see steps 12 and 11 in the table below.


The emitted sequence, step by step
-----------------------------------

``Recipe.get_galaxy_galaxy_recipe`` takes an ``epochs`` argument that
defaults to ``2``; ``Recipe.get_recipe`` always uses the default. This
implements published step (6):

    "Repeat steps 2–5 with the current initial conditions from the previous
    step."

The table below is the literal sequence emitted for a model with
``lens: [EPL, SHEAR_GAMMA_PSI]``, ``lens_light: [SERSIC_ELLIPSE]``,
``source_light: [SERSIC_ELLIPSE, SHAPELETS]``, and a ``theta_E`` entry under
``lens_options: initial_guesses:``. Indices 0–17 are one epoch; index 18 closes
the epoch and index 19 restarts it. Ten PSO runs are emitted in total.

.. list-table::
   :header-rows: 1
   :widths: 5 24 45 26

   * - #
     - Operation
     - Effect
     - Published step
   * - 0
     - ``lens_add_fixed``
     - Freezes every free lens-mass parameter at its current value:
       ``EPL: theta_E, e1, e2, gamma, center_x, center_y`` and
       ``SHEAR_GAMMA_PSI: gamma_ext, psi_ext``.
     - (2) "Fix all the model parameters except for the deflector's light
       profile."
   * - 1
     - ``source_add_fixed``
     - Freezes every free source parameter:
       ``SERSIC_ELLIPSE: R_sersic, n_sersic, center_x, center_y, e1, e2`` and
       ``SHAPELETS: center_x, center_y, beta``.
     - (2), same clause. The lens light is deliberately *not* frozen — it is
       the only free block.
   * - 2
     - ``kwargs_likelihood: image_likelihood_mask_list``
     - Swaps the science masks for the **arc masks** built by
       ``Recipe.get_arc_mask`` (one per band).
     - (2) "Create a mask for the lensed arcs."
   * - 3
     - **PSO**
     - Deflector light only, arcs masked out of the likelihood.
     - (2) "Optimize the deflector light parameters masking the lensed arcs."
   * - 4
     - ``source_remove_fixed``
     - Releases all source parameters frozen at step 1.
     - (3) "Optimize only the remaining source light parameters"
   * - 5
     - ``source_add_fixed`` with a value
     - Re-freezes the shapelet ``beta`` **at the explicit value 0.1**.
     - (3) "Fix the shapelet scale parameter :math:`\beta` = 0.1 arcsec."
   * - 6
     - ``lens_light_add_fixed``
     - Freezes the deflector light at its step-3 solution.
     - (3) "fix the lens model parameters and the deflector light parameters"
   * - 7
     - ``kwargs_likelihood: image_likelihood_mask_list``
     - Restores the science masks from ``ModelConfig.get_masks()``
       (``None`` if the config has no ``mask:`` block).
     - (3) "Note, in this step the lensed arcs are not masked."
   * - 8
     - ``lens_add_fixed`` with values
     - Overwrites the frozen lens-mass values with everything listed under
       ``lens_options: initial_guesses:`` — e.g. ``theta_E``, ``e1``, ``e2``.
     - (3) "Fix the Einstein radius :math:`R_{\rm E}` and ellipticity
       parameters :math:`\{q_{\rm m}, {\rm PA}_{\rm m}\}` to the values
       measured by the SLACS analysis (Auger et al. 2009). If such
       pre-determined values are not available, :math:`R_{\rm E}` can be fixed
       to an approximate guess and the ellipticity parameters can be set to the
       values for the circular case."
   * - 9
     - **PSO**
     - Source only, full mask, lens pinned at the guesses, ``beta`` = 0.1.
     - (3) "to find an approximate position of the source on the source plane"
   * - 10
     - ``lens_remove_fixed``
     - Releases the whole lens-mass model.
     - (4) "optimize for the source parameters and the PEMD parameters
       together"
   * - 11
     - ``lens_add_fixed``
     - Immediately re-freezes ``gamma_ext, psi_ext`` at their current values.
       Emitted only when a shear profile exists.
     - (1) "fix ... shear magnitude :math:`\gamma_{\rm ext} = 0` for all the
       steps below" — see *Deviations* below.
   * - 12
     - ``lens_add_fixed`` with a value
     - Freezes the power-law slope at the explicit value ``gamma = 2.0``.
       Emitted only when a power-law profile exists.
     - (1) "fix the logarithmic slope :math:`\gamma = 2` ... for all the steps
       below"
   * - 13
     - **PSO**
     - Lens mass + source together; deflector light fixed; ``beta`` = 0.1.
     - (4) "Keep the deflector light parameters fixed and optimize for the
       source parameters and the PEMD parameters together. Keep
       :math:`\beta` = 0.1 arcsec fixed in this step."
   * - 14
     - ``source_remove_fixed``
     - Releases the shapelet ``beta``.
     - (5) "Free :math:`\beta`"
   * - 15
     - **PSO**
     - Lens mass + source, ``beta`` free, deflector light still fixed.
     - (5) "and optimize all the non-fixed parameters together"
   * - 16
     - ``lens_light_remove_fixed``
     - Releases the deflector light.
     - *Extension* — not in the published recipe.
   * - 17
     - **PSO**
     - Everything free except ``gamma`` (= 2) and the shear.
     - *Extension* — not in the published recipe.
   * - 18
     - ``lens_remove_fixed``
     - Releases the whole lens-mass model, including ``gamma`` and the shear.
       No PSO follows inside the epoch, so this is a hand-off, not an
       optimization step.
     - "During the sampling, we free the parameters :math:`\gamma` and
       :math:`\gamma_{\rm ext}` that were fixed in the optimization step."
   * - 19+
     - —
     - Epoch 2 repeats indices 0–18 from the current state.
     - (6) "Repeat steps 2–5 with the current initial conditions from the
       previous step."

How the fix/unfix helpers behave
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Recipe.fix_params`` and ``Recipe.unfix_params`` build the parameter
name lists by reading the *bound* dictionaries returned by ``ModelConfig``
(``get_lens_model_params()``, ``get_lens_light_model_params()``,
``get_source_light_model_params()``) and skipping anything already present in
that component's ``fixed`` dictionary. Two consequences worth knowing:

* Anything you fix in the config (``lens_options: fix:``,
  ``lens_light_options: fix:``, ``source_light_options: fix:``) stays fixed for
  the entire run — the recipe will never release it. With
  ``lens_light_options: fix: {0: {n_sersic: 4.}}``, for instance, ``n_sersic``
  is absent from the ``lens_light_add_fixed`` list at step 6.
* When ``lenstronomy``'s ``update_settings`` receives an entry without an
  explicit value list, it fixes the parameter *at its current model value*
  (``UpdateManager._add_fixed``). Only steps 5, 8 and 12 supply values —
  ``beta`` = 0.1, the ``initial_guesses``, and ``gamma`` = 2.0 respectively.


The arc mask
------------

Step 2 masks the lensed arcs so that the deflector light can be fitted without
the arcs pulling on it. Published description:

    "Create a mask for the lensed arcs. We provide the algorithm to
    automatically make the mask for the arcs in Appendix B."

and, in the Figure B1 caption:

    "We use this mask to robustly fit the lens light profile only at one
    particular step within our model fitting procedure."

``Recipe.get_arc_mask`` implements Appendix B's six illustrated steps. It
requires the deflector to be near the image centre — the docstring states "The
lens galaxy is required to be close to the center (within a few pixels) of the
image" — because the radial gradient is measured about the centre of the array.

.. list-table::
   :header-rows: 1
   :widths: 6 44 50

   * - #
     - Figure B1 (Shajib et al. 2021)
     - Implementation in ``get_arc_mask``
   * - 1
     - "Take pixel-wise radial gradient outward from the center of the
       deflector. The pixels with positive gradient (red) mark the inner edge
       of the arcs."
     - ``x_diff``/``y_diff`` are first differences along each axis; they are
       projected onto the radial unit vector of a centred coordinate grid,
       with a ``1e-10`` softening in the denominator to avoid dividing by
       :math:`r = 0`.
   * - 2
     - "Make a binary image of the gradient with positive values set to 1
       (white) and negative values set to 0 (black). Set all the pixels within
       the inner 0.4 arcsecond to 0."
     - Thresholded and inverted so that pixels with a positive *outward*
       gradient end up at 1. Pixels inside ``clear_center`` are then zeroed;
       the module constant is ``DEFAULT_CLEAR_CENTER = 0.4``, matching the
       published 0.4 arcsec — but it is now scaled to the Einstein radius, see
       below.
   * - 3
     - "Set connected white regions with area below 5 pixels to zero. This
       procedure removes small white regions near the arc that were created due
       to noise."
     - ``scipy.ndimage.label`` with a 3×3 connectivity structure, then
       ``id_sizes < 5`` zeroes the small regions. The threshold is literally
       ``5``.
   * - 4
     - "Dilate the binary image separately in the four quadrants with these
       structural elements. The size of the structural element matrices is
       7×7."
     - Four triangular structuring elements ``a1``–``a4`` built from
       ``np.tril(np.ones((8, 8)))`` with a zeroed diagonal. The arrays are
       8×8, but their first row and column are all zero, so the effective
       footprint is 7×7 as published.
   * - 5
     - "The dilation grows the white regions radially outward. Thus, the white
       regions can span over the whole width of the lensed arcs."
     - Each quadrant is dilated with the element oriented outward for that
       quadrant, which is why the four elements differ by flips and rotations.
   * - 6
     - "Invert the binary image to create the mask for the lensed arcs."
     - ``arc_mask = 1 - np.pad(dilated, ((0, 1), (0, 1)), "minimum")``. The pad
       restores the pixel lost to the first difference, so the mask matches the
       image shape. Convention: 1 = used in the likelihood, 0 = masked.

Two implementation details are not in the paper:

**The quadrant split follows the array centre.** Steps 4–5 dilate each quadrant
of the gradient map separately, so the boundary between the quadrants has to
fall on the deflector. The split index is derived from the same centred
coordinate grid the radial gradient is measured on, so it tracks the cutout
size. It was previously hard-coded at pixel 50, which is the centre only for the
100×100 cutouts of the published analysis; at any other size the split fell
off-centre and marked pixels on the wrong side of it were dilated away from the
deflector's true radial direction, leaving the mask's outer edge less faithful
to the arc width. Masks for 100×100 cutouts are unchanged by the fix.

**The centre is re-opened when a science mask is supplied.** If ``get_arc_mask``
receives a ``mask``, the arc mask is multiplied by it and then a circle of
radius ``clear_center`` is forced back to 1. So even a config that masks the
deflector core out of the science likelihood still fits the deflector light to
its central pixels during step 3. This is a deliberate departure from the
paper's treatment of the core; see *Deviations*.

Scaling the clear centre to the Einstein radius
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fixed 0.4 arcsec clear centre is safe for the SLACS sample — Table 1 of
Shajib et al. (2021) lists Einstein radii from :math:`R_{\rm E}` = 0.951 to
1.788 arcsec — but it swallows a small lens whole. ``Recipe.get_clear_center``
therefore computes

.. code-block:: text

    clear_center = max(CLEAR_CENTER_MIN_PIXELS * np.max(pixel_size),
                       min(DEFAULT_CLEAR_CENTER, CLEAR_CENTER_THETA_E_FACTOR * theta_E))

with ``DEFAULT_CLEAR_CENTER = 0.4``, ``CLEAR_CENTER_THETA_E_FACTOR = 0.5`` and
``CLEAR_CENTER_MIN_PIXELS = 2``. The ``min`` means the value can only *shrink*
relative to the published 0.4 arcsec, so behaviour for SLACS-like lenses is
unchanged; the ``max`` keeps at least two pixels clear, "below which the arc
finder starts marking the deflector's own light as arc" (module comment).

``theta_E`` is read from ``lens_options: initial_guesses:`` by
``Recipe.get_theta_E_guess``. With a 0.04 arcsec pixel scale:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - ``theta_E`` guess (arcsec)
     - ``clear_center`` (arcsec)
     - in pixels
   * - none provided
     - 0.40
     - 10.0
   * - 1.25
     - 0.40
     - 10.0
   * - 0.80
     - 0.40
     - 10.0
   * - 0.50
     - 0.25
     - 6.2
   * - 0.30
     - 0.15
     - 3.7
   * - 0.15
     - 0.08
     - 2.0 (floor)

An explicit ``mask: clear_center:`` in the config overrides the whole
calculation and is used verbatim.


PSO settings
------------

Every PSO step in this recipe uses the same three values from
``fitting: pso_settings:``::

    fitting:
      pso: true
      pso_settings:
        num_particle: 100
        num_iteration: 100
        sigma_scale: 1.0     # optional

* ``num_particle`` → ``n_particles``, ``num_iteration`` → ``n_iterations``.
* ``sigma_scale`` sets the width of the box the swarm is seeded in, relative to
  the per-parameter ``sigma`` values: the swarm starts uniformly within
  ``init ± sigma * sigma_scale`` and its spatial scale never grows beyond that
  box. It defaults to ``DEFAULT_PSO_SIGMA_SCALE = 1.0`` and is applied
  identically to all ten PSO runs. (The ``galaxy-quasar`` recipe instead hard-codes
  a per-stage schedule of ``[1.0, 0.1, 0.1]``.)
* If ``fitting: pso:`` is false or absent, ``get_galaxy_galaxy_recipe`` returns
  an empty list and only the sampler runs.

``thread_count`` is passed through from ``Processor.swim(thread_count=...)``.

Running the staging with gradient descent
------------------------------------------

The staging above is a property of the *problem*, not of the optimizer: every step it
emits is an ``update_settings`` operation on the fitting sequence's parameter state,
which any optimizer reads. It is therefore built once, by
``Recipe._build_galaxy_galaxy_sequence``, and consumed by three recipes, which differ
only in what they run at each of the five optimization stages:

.. list-table::
   :header-rows: 1

   * - ``recipe_name``
     - What runs at each stage
   * - ``galaxy-galaxy``
     - one PSO
   * - ``galaxy-galaxy-gradient-descent``
     - one gradient descent
   * - ``galaxy-galaxy-pso-gradient-descent``
     - one shortened PSO, then a gradient descent converging it

The five stages are named ``lens_light``, ``source``, ``lens_source``,
``lens_source_beta`` and ``all``, corresponding to the emitted-sequence indices 3, 9,
13, 15 and 17 tabulated above. A regression test pins the emitted ``galaxy-galaxy``
PSO sequence to a golden file, so that changes to the shared staging cannot alter the
PSO path unnoticed.

The staging matters just as much for a gradient descent as for a swarm: freeing every
parameter block at once from a rough starting point walks into a local minimum where
the source absorbs the lens light. What makes the staging usable by a gradient descent
is the warm start: the first chain of each descent begins at the current parameter
state rather than at a draw around it, and a chain that ends worse than it started is
never accepted. Stages therefore compose, and a stage can never leave the model worse
than it found it.

Per-stage settings come from ``fitting: gradient_descent_schedule:``; see
``CONFIG_OPTIONS.rst``. Because the first chain is the warm start, ``sigma_scale``
controls only how far the *exploring* chains are thrown, so the default schedule
explores widely in the early stages and refines from the warm start alone in the late
ones.


How the hybrid recipe divides the work
---------------------------------------

The hybrid recipe does not simply add a descent to each swarm -- that could only ever
cost more than the ``galaxy-galaxy`` recipe it has to beat. It gives each optimizer the
job it is good at, and takes the budget for the descent out of the swarm:

* **The swarm chooses the basin.** It finds the right region within a few tens of
  iterations and then buys further digits very slowly, because its spatial scale
  collapses geometrically. The four restricted stages are cut to a quarter of the
  configured iterations (``pso_iteration_scale: 0.25``) at the full particle count, so
  each swarm keeps its coverage of its subspace and loses only its slow tail. The final
  ``all`` stage keeps its whole budget: it is the only stage with every parameter block
  free, and it is the one that decides which basin the run ends in.
* **One gradient descent converges it**, after the final stage, with five chains. The
  first warm-starts from the swarm's answer; the rest start from draws around it. A
  chain that ends worse than the state it was handed is never accepted, so the chains
  can only cost time, never quality -- and they are what let the descent leave a
  mediocre basin. In one measured run the swarm finished at ``logL = -4457`` and the
  chains found ``-4206``.
* **One epoch, not two.** A second pass through the staging exists to let a swarm that
  stopped short try again from a better starting point. Its cost is the whole staging
  over again; the extra descent chains buy the same insurance for a fraction of it.

A descent at *every* stage was measured and is not worth it. It does not improve the
fit -- the final stage re-optimizes whatever an earlier one converged -- and it costs
real time, because ``FittingSequence.fit_sequence`` rebuilds the likelihood and clears
the JAX compilation cache before every step, so each extra descent pays a full XLA
compilation. On a GPU that compilation, not the iterations, is most of what an
intermediate descent costs. It can also actively hurt: converging a restricted stage
too well traps the model, because the next stage's shortened swarm is seeded at that
optimum and cannot leave it. One measured run spent 66 s on a descent that improved
``logL`` by 0.001 and then a swarm that improved it by 0.000.


Measured against the ``galaxy-galaxy`` recipe
----------------------------------------------

All three recipes were measured on one single-band galaxy–galaxy lens with
``lens: [EPL, SHEAR_GAMMA_PSI]`` and ``lens_light: [SERSIC_ELLIPSE]``, on one
NVIDIA RTX A6000, with ``pso_settings: {num_particle: 100, num_iteration: 100}``
and sampling off. The source light is the one thing varied between the two tables
below. Every ``logL`` is recomputed from the returned
``kwargs_result`` through one pristine likelihood, so runs that ended with different
parameters fixed are on one scale. The swarm is not seeded, so each recipe was run
several times; the spread is the point, not noise to be averaged away.

Single Sersic source:

.. list-table::
   :header-rows: 1

   * - ``recipe_name``
     - seconds (mean)
     - ``logL``, best to worst
   * - ``galaxy-galaxy``
     - 209.1 (n = 3)
     - -4906.07, -4909.41, -4916.56
   * - ``galaxy-galaxy-pso-gradient-descent``
     - 113.4 (n = 5)
     - -4901.93, -4901.93, -4901.93, -4901.93, -4901.95

Sersic plus shapelets source (``n_max: 6``):

.. list-table::
   :header-rows: 1

   * - ``recipe_name``
     - seconds (mean)
     - ``logL``, best to worst
   * - ``galaxy-galaxy``
     - 906.1 (n = 5)
     - -4207.12, -4207.79, -4211.75, -4278.09, -4293.24
   * - ``galaxy-galaxy-pso-gradient-descent``
     - 433.0 (n = 7)
     - -4206.22, -4206.22, -4206.22, -4206.22, -4275.75, -4275.75, -4275.76

On the Sersic source the hybrid is 1.8 times faster and the better fit in every run, and
it is very nearly deterministic: a spread of 0.02 in ``logL`` against the swarm's 10.5,
because the descent converges the same basin every time whatever the swarm hands it.

With shapelets the picture is different and worth stating precisely. That likelihood
has two deep optima close together, near ``-4206`` and ``-4276``, and **both recipes
land in either, at about the same rate** -- three runs in five for the swarm, four in
seven for the hybrid. Adding swarm iterations at the deciding stage does not shift that
rate: doubling the ``all`` stage's swarm gave two in five. So the hybrid is *not* more
likely to find the better mode. What it does is take half the wall clock and reach a
better optimum *within* whichever mode it lands in: ``-4206.22`` every time against
``-4207.12`` to ``-4211.75``, and ``-4275.8`` every time against ``-4278.1`` to
``-4293.2``. Rank the runs of each recipe from best to worst and the hybrid wins at
every position.

Two smaller findings from the same measurements, in case they are useful:

* The ``galaxy-galaxy`` recipe's second epoch is not reliably an improvement. The
  ``lens_add_fixed`` that applies ``lens_options: initial_guesses:`` at the start of
  every ``source`` stage resets the deflector to its initial guess, and the epoch does
  not always recover: one measured Sersic run ended epoch 1 at ``logL = -4906`` and
  finished at ``-4913``.
* Sub-stage budgets barely matter. Running the restricted stages' swarms at 0.25, 0.5
  or the full budget all landed inside the run-to-run scatter. Only the final stage's
  swarm and the descent's chain count changed anything.

None of this depended on tuning the recipe alone. Two defects in the JAXtronomy fork's
optimizer had to be fixed first, and a third made gradient descent uncompetitive by
construction:

* ``EPLMajorAxis`` differentiated a fixed 100-term hypergeometric series regardless of
  the deflector's flattening, where the forward pass adaptively picks 10 to 20 terms,
  and reverse mode through that series stacked one residual per iteration. A gradient
  cost 24.6 ms against 2.1 ms for a value. Switching between pre-differentiated series
  and returning materialized partials brings it to 4.1 ms -- twice a value, not twelve
  times it. That is what makes a descent of a few thousand iterations affordable at
  all.
* A chain returned its *final* L-BFGS iterate rather than the best one it visited.
* A run of stalled steps ended a chain rather than restarting its L-BFGS memory,
  stopping chains after 43 to 400 iterations, hundreds in ``logL`` short of the
  optimum. See ``pure_gradient_descent_divergence.md``.


Every *optimizer* step of every one of the three recipes is guarded against
regression: a PSO is seeded with the current parameter state as its initial global
best, and a gradient descent chain that ends worse than it started is never accepted.
The sequence as a whole is still not monotone, because the staging also contains steps
that set parameters rather than optimize them. The ``lens_add_fixed`` that applies
``lens_options: initial_guesses:`` at the start of every ``source`` stage is the one
that matters: in a second epoch it discards the deflector the first epoch found and
resets it to the initial guess. On the system measured above that costs the
``galaxy-galaxy`` recipe several thousand in ``logL``, and its second epoch does not
always recover: one measured run ended epoch 1 at ``logL = -4906`` and finished at
``-4913``.

The older ``fitting: gradient_descent:`` flag remains supported and keeps its original
meaning: with the ``galaxy-quasar`` or ``galaxy-galaxy`` recipe names it replaces the
whole staged sequence with a single ``optax`` step over all free parameters. The recipe
names above are preferred.

Gradient descent runs through JAXtronomy, so all of these require
``Processor.swim(..., use_jax=True)``.


What happens after the recipe
------------------------------

``Recipe.get_recipe`` appends ``Recipe.get_sampling_sequence`` to the
optimization steps, matching:

    "After the pre-sampling optimization, we then initiate the MCMC sampling
    from the optimized lens model after step 6. During the sampling, we free
    the parameters :math:`\gamma` and :math:`\gamma_{\rm ext}` that were fixed
    in the optimization step."

The final ``lens_remove_fixed`` of the last epoch (index 18 above) is exactly
that release of ``gamma`` and the shear, which is why it carries the code
comment "finally, relax shear parameters for MCMC later". Sampling itself is
configured separately (``fitting: sampling:``, ``sampler:``,
``sampler_settings:``); ``emcee`` and ``Nautilus`` are supported. As the
``dolphin`` paper notes, "As a forward modeling pipeline, dolphin will still
require running sampling methods, such as the MCMC, to obtain the best-fit lens
model parameters and their uncertainties."


Deviations from the published recipe
-------------------------------------

These are real differences between the code and Shajib et al. (2021), not
simplifications of the description above.

**External shear is pinned at its initial value, not at zero.** Published step
(1) says "fix ... shear magnitude :math:`\gamma_{\rm ext} = 0` for all the steps
below". The code instead calls ``fix_params("lens", external_shear_model_index)``
with no value, which freezes ``gamma_ext``/``psi_ext`` at whatever they
currently are. Since ``ModelConfig.get_lens_model_params()`` initializes
``SHEAR_GAMMA_PSI`` with ``gamma_ext = 0.05, psi_ext = 0.0``, and no PSO ever
runs with the shear free, the shear is held at 0.05 throughout the whole
optimization. To reproduce the published behaviour, set the initial value
yourself::

    lens_options:
      initial_guesses:
        1: {gamma_ext: 0.0}

That also pins the shear to 0.0 at recipe step 8, and — because
``initial_guesses`` only changes the starting point — leaves it free for
sampling. Use ``lens_options: fix:`` instead only if you want the shear fixed
during sampling too.

**The deflector centroid is not disjoined for sampling.** The paper adds "We
also independently sample the centroids of the deflector mass and light
distributions." ``dolphin`` keeps ``joint_lens_with_light`` in force through
sampling; ``recipe.py`` contains commented-out code that would have restored the
unconstrained ``kwargs_constraints`` at the end of each epoch, but it is
inactive. Override it per lens with a ``kwargs_constraints:`` block in the
config if you want the published behaviour.

**Two PSO runs per epoch instead of one at published step (5).** The paper's
step (5) frees :math:`\beta` and optimizes the non-fixed parameters, with the
deflector light still fixed from step (4). The code runs that (index 15), then
additionally releases the deflector light and runs a further all-free PSO
(indices 16–17). Each epoch therefore contains five PSO runs, and a two-epoch
recipe emits ten.

**The 0.4 arcsec central mask is not applied automatically.** The paper reports
a second pass: "To avoid bias in the model from this poor fitting of the
deflector light profile at the centre, we mask out the central 0.4 arcsec and
rerun the whole fitting procedure." That central mask is a *science* mask, not
the arc mask's ``clear_center``, and ``dolphin`` does not add it for you. Request
it explicitly with a central entry under ``mask: extra_regions:``. Note the
interaction described above: ``get_arc_mask`` will re-open that central circle
for the deflector-light step.

**The arc-mask clear centre now scales with the Einstein radius**, where the
paper uses a flat 0.4 arcsec. See the table above; the value only ever shrinks
below 0.4 arcsec.


Configuration keys that affect this recipe
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Key
     - Effect on the recipe
   * - ``model: lens:``
     - Presence of a power-law profile enables the ``gamma`` = 2 step; presence
       of a shear profile enables the shear-pinning step.
   * - ``model: source_light:``
     - Presence of ``SHAPELETS`` enables the ``beta`` = 0.1 fix/free steps.
   * - ``lens_options: initial_guesses:``
     - Every listed parameter is pinned at its guess value during the
       source-only PSO (recipe step 8). A ``theta_E`` entry additionally drives
       the arc mask's ``clear_center``.
   * - ``*_options: fix:``
     - Fixed for the entire run; the recipe never releases these.
   * - ``mask:`` / ``mask: provided:``
     - Supplies the science masks restored at recipe step 7, and the mask that
       ``get_arc_mask`` multiplies into the arc mask.
   * - ``mask: clear_center:``
     - Overrides the arc mask's clear-centre radius, in arcsec.
   * - ``fitting: pso:``, ``fitting: pso_settings:``
     - Enable the recipe and set every PSO's particle count, iteration count
       and seeding-box scale.
   * - ``fitting: psf_iteration:``
     - **Ignored** by this recipe.
   * - ``fitting: sampling:``, ``sampler:``, ``sampler_settings:``
     - Appended after the recipe by ``get_sampling_sequence()``.

See `CONFIG_OPTIONS.rst <CONFIG_OPTIONS.rst>`_ for the full configuration
reference.


References
----------

* A. J. Shajib, T. Treu, S. Birrer & A. Sonnenfeld, 2021, *MNRAS* **503**, 2380,
  "Dark matter haloes of massive elliptical galaxies at z ∼ 0.2 are well
  described by the Navarro–Frenk–White profile" — Section 3.2 (the recipe) and
  Appendix B (the arc mask). doi:10.1093/mnras/stab536
* A. J. Shajib et al., 2025, "dolphin: A fully automated forward modeling
  pipeline powered by artificial intelligence for galaxy-scale strong lenses" —
  Sections 2.1–2.2. arXiv:2503.22657
* C. Y. Tan et al., 2024, *MNRAS* **530**, 1474 — the multi-band application of
  this recipe. doi:10.1093/mnras/stae884
* J. Kennedy & R. Eberhart, 1995, *Proceedings of ICNN'95* — particle swarm
  optimization. doi:10.1109/icnn.1995.488968
