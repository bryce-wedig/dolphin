# Pure staged gradient descent used to diverge: the warm-start chain ended worse than it started

**Status: fixed 2026-09-02.** The divergence described below is gone. Two changes to the
JAXtronomy fork's `OptaxMinimizer.run_single` did it:

1. **A chain returns its best iterate, not its last one.** L-BFGS is monotone only while its
   line search succeeds; when the zoom search exhausts `max_linesearch_steps` without meeting
   the Wolfe conditions, optax applies the last trial step anyway and the loss can jump by
   orders of magnitude in one iteration. Returning the final iterate threw away everything the
   chain had achieved before that step. This is mechanism 1 below, and it was the whole of the
   reported symptom.
2. **A run of stalled steps restarts the L-BFGS memory instead of ending the chain.** A stalled
   line search is a failure of the quadratic model -- the stored curvature pairs no longer
   describe the local geometry -- not a stationary point. Discarding the memory turns the next
   step into steepest descent and the approximation rebuilds. Without this, chains stopped
   after 43 to 400 iterations, hundreds in `logL` short of the optimum they were heading for.

Re-measured on `lens_system1` (single Sersic source, PSO 100x100, one A6000, `maxiter=1000`):

| `recipe_name` | logL | seconds |
|---|---|---|
| `galaxy-galaxy-pso-gradient-descent` | **-4901.9** | 171 |
| `galaxy-galaxy` (PSO) | -4910.7 (mean of 3) | 209 |
| `galaxy-galaxy-gradient-descent` | -4939.6 | 301 |

No stage now ends worse than it started through its optimizer. The one remaining drop in the
pure-descent trace (`-981` at step 25) is the `lens_add_fixed` that applies
`lens_options: initial_guesses:` at the start of every `source` stage, which resets the
deflector to its initial guess -- a property of the shared staging, not of the optimizer, and
one the PSO recipe pays too.

Pure gradient descent is still the worst of the three, and that is now the expected result
rather than a bug: it has no global search at all, so each stage converges whichever basin its
draws happen to land in. Use `galaxy-galaxy-pso-gradient-descent`, where a particle swarm
chooses the basin and the descent converges it.

The rest of this note is the original diagnosis, kept because mechanisms 2 to 4 were **not**
fixed and remain worth understanding.

## What was measured (original, on 4 CPU cores)

Identical config (`lens_system1`, `maxiter=500`, sampling off, `epochs=2`), differing only
in `recipe_name`. `logL` recomputed from each `kwargs_result` through the likelihood so all
three are on one scale:

| `recipe_name` | logL |
|---|---|
| `galaxy-galaxy-pso-gradient-descent` | **-4904.03** |
| `galaxy-galaxy` (PSO) | -4967.33 |
| `galaxy-galaxy-gradient-descent` | **-5305.55** |

Reproducibility is not the problem: two runs in separate processes returned bitwise-identical
results, all 24 parameters and `logL = -5305.550647688584`.

## The decisive observation

Per-stage diagnostics from the run (`fit_output[i][2]`, see "Reproducing" below). Only the
four stages where every chain was rejected are shown:

```
--- stage 2: ALL CHAINS REJECTED, start_loss=6270.6 ---
   WARM iters= 500 loss=       23674.9 grad=  6.826e+02 finite=True
   draw iters=   3 loss=   100019359.1 grad=  1.719e+02 finite=True
--- stage 4: ALL CHAINS REJECTED, start_loss=5449.1 ---
   WARM iters=  31 loss=      963058.1 grad=  2.055e-01 finite=True
--- stage 7: ALL CHAINS REJECTED, start_loss=860085.7 ---
   WARM iters=  44 loss=     3603570.2 grad=  4.301e+08 finite=True
--- stage 9: ALL CHAINS REJECTED, start_loss=5305.6 ---
   WARM iters=  27 loss=      211334.3 grad=  0.000e+00 finite=True
```

The warm chain starts at exactly `start_loss` and ends far above it. **This is not a schedule
problem** — it is not the random draws misbehaving, so narrowing `sigma_scale` or changing
`num_chains` cannot fix it. It is also not premature stopping: the chains exit with gradient
norms of 10² to 10⁸, nowhere near the `grad_tolerance = 1e-5` that would indicate a stationary
point. Raising `maxiter` or lowering `grad_tolerance` will not help either; stage 2 already
ran to `maxiter = 500` and diverged anyway.

A correctly line-searched L-BFGS is monotonically non-increasing. That this one is not is the
core anomaly.

## Mechanisms identified

### 1. `run_single` returns the *final* iterate, not the *best* iterate -- FIXED

`/grad/bwedig/JAXtronomy/jaxtronomy/Sampling/Samplers/optax.py`, `run_single`: the
`jax.lax.while_loop` carries `(params, state, tol_hit, grad_norm, finite)` and returns
`final_params`. Nothing tracks the lowest loss seen along the way. If the trajectory wanders
uphill — for whatever reason — the returned point is wherever it happened to stop, which can be
arbitrarily worse than both the start and the best point visited.

This is the cheapest thing to fix and converts several of these stages from "rejected" to
"kept a real improvement": add `best_params` / `best_value` to the carry and return those.
It treats the symptom rather than the cause, but it is strictly correct behaviour for a
minimizer and is worth doing regardless of what else is found.

**Done.** `best_params` and `best_value` are carried through the `while_loop`, updated from
the loss the line search has already computed, so this costs no extra likelihood
evaluations. `best_value` starts at infinity rather than at the starting loss, so a chain
that never improves returns exactly where it began.

### 2. The likelihood has a hard, zero-gradient cliff on bound violation

`/grad/bwedig/JAXtronomy/jaxtronomy/Sampling/likelihood.py:310-318`:

```python
if self._check_bounds is True:
    penalty, bound_hit = self.check_bounds(
        args, self._lower_limit, self._upper_limit, verbose=verbose
    )

    def true_fun(*args, **kwargs):
        return -(10.0**18)

    logL = lax.cond(bound_hit, true_fun, self.log_likelihood, kwargs_return)
```

Outside the bounds the likelihood is the **constant** `-1e18`. A constant has zero gradient, so
once a line-search trial step crosses a bound there is no gradient information pointing back
inside — the descent direction is undefined and the optimizer cannot recover. This is a
fundamental incompatibility between a hard-penalty likelihood and a gradient-based optimizer.
PSO is unaffected because it is derivative-free and simply sees a bad fitness value.

Note `check_bounds` is on in dolphin's likelihood settings, and note the related dead code at
`likelihood.py:397` (`penalty = jnp.where(bound_hit, 10.0**5, 0.0)`) — the returned `penalty`
is computed but the `lax.cond` above uses the `-1e18` constant instead, so the `1e5` value is
never applied. Worth checking whether that divergence is intentional.

The plausible fix is a **soft** penalty that grows with the distance outside the bound, so the
gradient points back into the feasible region, e.g. `-1e18` replaced by a smooth quadratic in
the constraint violation. That is a behavioural change to JAXtronomy's likelihood and would
affect the samplers too, so it needs care — possibly gate it on a flag used only by `optax`.

### 3. A flat attractor at exactly `211334.31083053577`

Stage 9's warm chain terminates there with gradient **exactly** `0.0`. The same value, to all
17 digits, appears three times in an earlier unstaged 8-chain run recorded in
`notebooks/Basic example.ipynb` (chains 1, 5 and 6). A value that repeats bitwise from
different starting points, with an exactly zero gradient, is a constant region of the
likelihood, not a local minimum of the fit.

Because the gradient is exactly zero there, the gradient-norm stopping criterion
(`grad_norm > grad_tol` in `run_single`) reads it as convergence and stops. So the flat region
is both an attractor and a place the optimizer declares success. It is not the `-1e18` cliff —
the value is far too small — so it is a *separate* flat region that has not yet been
identified.

Three chains in the same run also ended near `1.0002e8` (`100018942.80`, `100018948.29`,
`100019359.11`), i.e. `1e8` plus a varying few-thousand offset, which looks like a different
additive penalty of `1e8`. That constant has not been located in the source either. **Finding
what produces `211334.31083053577` and the `1e8` offset is the highest-value next step**: both
are almost certainly penalty or degenerate-solve regimes, and both trap the optimizer.

Suggested approach: evaluate `OptaxMinimizer._loss` at the stage-9 returned parameters, then
perturb each parameter individually and watch which coordinates leave the plateau. Also
instrument the individual likelihood terms (`image_likelihood`, `source_position_likelihood`,
`check_positive_flux`) to see which one is saturating.

### 4. A NaN gradient appears in stage 0

The first chain of stage 0 reports `grad = nan` while `finite=True` and the stage still
improves overall. Non-fatal, but it means at least one gradient evaluation produced NaN, which
is worth understanding — it may share a cause with 2 and 3.

## Already ruled out

- **Not non-determinism.** Two separate processes gave bitwise-identical output.
- **Not the random draws / schedule.** The deterministic warm-start chain diverges on its own.
- **Not premature stopping.** Chains exit with gradient norms of 10²–10⁸, and one ran the full
  `maxiter=500` before diverging.
- **Not the staging.** A test asserts the recipe's emitted `update_settings` sequence is
  identical to the PSO recipe's, and the PSO path is pinned to a golden file, so both recipes
  provably see the same freeze/unfreeze sequence.
- **Not the previously-fixed bugs.** The bound-clipping, finiteness guard, relative tolerance
  and warm start are all in place and tested (see "Context" below); these diverging chains are
  finite, in-bounds at the start, and running to completion.

## Reproducing

```bash
# JAXtronomy must be the editable fork, not a site-packages copy -- see Context.
python -c "import jaxtronomy; print(jaxtronomy.__file__)"

cd /grad/bwedig/dolphin
python - <<'PY'
from dolphin.util import select_jax_device, enable_jax_x64
select_jax_device("cpu", host_device_count=4); enable_jax_x64()

from dolphin.processor import Processor
from dolphin.processor.core import Processor as P

p = Processor("io_directory_example")
config = p.get_lens_config("lens_system1")
config.settings["fitting"]["sampling"] = False
config.settings["fitting"]["gradient_descent_settings"] = {"maxiter": 500, "rng_seed": 1}
P.get_lens_config = lambda self, name: config

p.swim("lens_system1", "gd_debug", log=False, use_jax=True,
       recipe_name="galaxy-galaxy-gradient-descent")
PY
```

Takes roughly 25 minutes on 4 CPU cores. Then read the per-stage diagnostics, which are saved
into the output file by `FileSystem.save_output`:

```python
out = p.file_system.load_output("lens_system1", "gd_debug")
for step in out["fit_output"]:
    if step[0] != "optax":
        continue
    start, chains = step[2][0], step[2][1:]
    for c in chains:
        print("WARM" if c["warm_start"] else "draw", c["num_iterations"],
              c["loss"], c["grad_norm"], c["finite"], c["accepted"])
```

Each `optax` entry in `fit_output` is `[fitting_type, kwargs_result, chain_diagnostics]`.
`chain_diagnostics[0]` is the starting state (`chain == -1`); the rest are one per chain.

## Context: what is already fixed, so it is not re-investigated

Fixed and tested on 2026-09-01/02, in the JAXtronomy fork
(`/grad/bwedig/JAXtronomy/jaxtronomy/Sampling/Samplers/optax.py`):

- **Bound `±inf` trap.** numpyro's transform for a bounded parameter is a logit, so a value
  sitting exactly on a bound unconstrains to `±inf` and one outside to `NaN`. `_clip_inside_bounds`
  nudges starting points inside by `1e-9` of the range (floored at a few ULPs, since in float32
  `1.0 - 1e-9` rounds back to `1.0`).
- **Non-finite chains never terminating.** A `NaN` diff makes both halves of the stall test
  False, so the counter reset every iteration and the chain burned all of `maxiter`. A
  finiteness flag is now carried in the `while_loop` carry; such a chain stops at iteration 0.
- **Gradient-norm stopping.** Added, carried in the loop carry — reading it off the optimizer
  state does not work, since `opt.init` leaves `grad = 0` and `value = inf`, which would end
  the loop before its first iteration.
- **Relative loss tolerance**, and worsening steps no longer counting toward convergence.
- **Regression guard.** A chain that ends worse than the starting state is never accepted. This
  is what keeps the current divergence from being catastrophic — without it the recipe would
  return the `211334` plateau instead of `5305`.
- **Reproducibility.** `rng_seed` defaults to `0` with per-stage derived seeds.

Also relevant: the `kuina` conda env used to contain a **stale non-editable copy** of
JAXtronomy in `site-packages` that shadowed the fork, so runs executed older code than the
checkout. It was removed and the fork installed editable on 2026-09-02. Any result predating
that should be treated as suspect.

## Related notes

- [mge_lens_light_negative_source_amplitudes.md](mge_lens_light_negative_source_amplitudes.md)
  and `jaxtronomy_check_positive_flux_inert.md` — `check_positive_flux` is inert under
  JAXtronomy, which is worth bearing in mind when auditing which likelihood terms can saturate.
- [GALAXY_GALAXY_RECIPE.rst](GALAXY_GALAXY_RECIPE.rst) — the shared staging and the three
  recipes that consume it.
- [CONFIG_OPTIONS.rst](CONFIG_OPTIONS.rst) — `gradient_descent_settings` and
  `gradient_descent_schedule`.
