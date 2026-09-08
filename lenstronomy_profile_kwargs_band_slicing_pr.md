# lenstronomy PR: `create_class_instances` does not band-slice `*_profile_kwargs_list`

## Summary

`lenstronomy.Util.class_creator.create_class_instances` slices every model list by
`index_*_model_list[band_index]` to build the per-band model, but passes the corresponding
`*_profile_kwargs_list` through **unsliced**. Profile keyword arguments are then paired with
the wrong profiles.

All four are affected:

| model list (sliced) | profile kwargs (NOT sliced) | consumer |
|---|---|---|
| `lens_model_list` | `lens_profile_kwargs_list` | `LensModel` |
| `source_light_model_list` | `source_light_profile_kwargs_list` | `LightModel` |
| `lens_light_model_list` | `lens_light_profile_kwargs_list` | `LightModel` |
| `optical_depth_model_list` | `optical_depth_profile_kwargs_list` | `DifferentialExtinction` |

The failure is **silent** whenever the misassigned kwargs happen to be valid for the profile
they land on, because `LightModelBase.__init__` pairs the two lists with `zip`, which truncates
without complaint. It is a `TypeError` when they are not.

Present on `main` as of this writing (verified against the current
`lenstronomy/Util/class_creator.py`) and in the released 1.14.1.

## Severity

Multi-band fitting with per-band models is the only affected configuration, but there it either
silently fits the wrong model or crashes. This surfaced downstream in
[dolphin](https://github.com/ajshajib/dolphin), where a two-band model with satellite galaxies
and an `MGE_SET_ELLIPSE` deflector cannot be constructed at all.

## Reproducers

All three run against a clean install; no data files needed.

### 1. Silent misassignment (light)

```python
from lenstronomy.Util.class_creator import create_class_instances

kwargs = dict(
    lens_model_list=["EPL"],
    lens_light_model_list=["MGE_SET", "MGE_SET"],
    lens_light_profile_kwargs_list=[{"n_comp": 10}, {"n_comp": 30}],
    index_lens_light_model_list=[[0], [1]],
)
for band in (0, 1):
    _, _, lens_light, _, _ = create_class_instances(band_index=band, **kwargs)
    print(band, [f.num_linear for f in lens_light.func_list])

# actual:   0 [10]   /   1 [10]      <- band 1 silently gets band 0's n_comp
# expected: 0 [10]   /   1 [30]
```

Same shape with `source_light_profile_kwargs_list` + `index_source_light_model_list`.

### 2. Silent misassignment (mass)

```python
kwargs = dict(
    lens_model_list=["NFW", "NFW"],
    lens_light_model_list=[],
    lens_profile_kwargs_list=[{"interpol": True}, {"interpol": False}],
    index_lens_model_list=[[0], [1]],
)
for band in (0, 1):
    lens_model, _, _, _, _ = create_class_instances(band_index=band, **kwargs)
    print(band, lens_model.lens_model.func_list[0]._interpol)

# actual:   0 True / 1 True         <- band 1 silently gets band 0's interpol
# expected: 0 True / 1 False
```

### 3. Hard failure

```python
kwargs = dict(
    lens_model_list=["EPL"],
    lens_light_model_list=["SERSIC_ELLIPSE", "MGE_SET"],
    lens_light_profile_kwargs_list=[{}, {"n_comp": 20}],
    index_lens_light_model_list=[[0], [1]],
)
create_class_instances(band_index=0, **kwargs)   # ok
create_class_instances(band_index=1, **kwargs)
# TypeError: MGESet.__init__() missing 1 required positional argument: 'n_comp'
```

Band 1's model list is `["MGE_SET"]` but it is zipped against `[{}, {"n_comp": 20}]`, so the
MGE is constructed from `{}`. The mirror image (a `{"n_comp": ...}` landing on a Sersic) gives
`TypeError: SersicUtil.__init__() got an unexpected keyword argument 'n_comp'`.

## Root cause

`lenstronomy/Util/class_creator.py`, `create_class_instances`. The lens light block is
representative:

```python
if index_lens_light_model_list is None or all_models is True:
    lens_light_model_list_i = lens_light_model_list
else:
    lens_light_model_list_i = [
        lens_light_model_list[k] for k in index_lens_light_model_list[band_index]
    ]
lens_light_model_class = LightModel(
    light_model_list=lens_light_model_list_i,
    profile_kwargs_list=lens_light_profile_kwargs_list,   # <-- full list, not sliced
)
```

The lens, source light, and optical depth blocks have the same shape.

Two things then hide the error:

- `LightModelBase.__init__` (`lenstronomy/LightModel/light_model_base.py`) iterates
  `zip(light_model_list, profile_kwargs_list)`. `zip` stops at the shorter list, so a
  length mismatch never raises — it just pairs the band's profiles with the *first* N kwargs.
- `ProfileListBase.__init__` (`lenstronomy/LensModel/profile_list_base.py`) indexes
  `profile_kwargs_list[i]` with `i` running over the *sliced* model list, so it reads the wrong
  entry. It also does, for `NFW_MC` / `NFW_MC_ELLIPSE_POTENTIAL`:

  ```python
  profile_kwargs_list[i]["z_lens"] = lens_redshift_list[i]
  profile_kwargs_list[i]["z_source"] = z_source_convention
  ```

  which mutates the caller's dicts in place at the wrong indices. Since `create_class_instances`
  is called once per band from `SingleBandMultiModel.__init__`, repeated construction can
  progressively corrupt the user's `kwargs_model`.

## Proposed fix

Two parts. The second is what stops this class of bug recurring.

**1. Slice each `*_profile_kwargs_list` alongside its model list.** For each of the four blocks:

```python
if index_lens_light_model_list is None or all_models is True:
    lens_light_model_list_i = lens_light_model_list
    lens_light_profile_kwargs_list_i = lens_light_profile_kwargs_list
else:
    lens_light_model_list_i = [
        lens_light_model_list[k] for k in index_lens_light_model_list[band_index]
    ]
    if lens_light_profile_kwargs_list is None:
        lens_light_profile_kwargs_list_i = None
    else:
        lens_light_profile_kwargs_list_i = [
            lens_light_profile_kwargs_list[k]
            for k in index_lens_light_model_list[band_index]
        ]
```

then pass `..._i` to the constructor. `None` must stay `None` (it means "defaults for every
profile"), so guard the comprehension rather than slicing unconditionally.

**2. Validate the lengths where they are consumed**, turning any residual mismatch into a loud
error instead of a silent truncation:

- `LightModelBase.__init__`: after the `None` default, raise `ValueError` if
  `len(profile_kwargs_list) != len(light_model_list)`, and keep the `zip`.
- `ProfileListBase.__init__`: the same check against `lens_model_list`.
- `DifferentialExtinction`: same, if it defaults the list the same way.

The message should name both lengths, e.g.
`"profile_kwargs_list has length 4 but light_model_list has length 2; they must match."`

## Tests

`test/test_Util/test_class_creator.py` — one test per affected list, asserting that band 1 gets
band 1's kwargs. Reproducers 1 and 2 above are directly usable; assert on
`func_list[0].num_linear` for MGE and on the constructor flag for NFW. Add reproducer 3 as a
`pytest.raises`-free positive test (after the fix it must succeed and produce
`MGESet(n_comp=20)` for band 1).

`test/test_LightModel/test_light_model.py` and `test/test_LensModel/test_profile_list_base.py`
— `pytest.raises(ValueError)` on a mismatched `profile_kwargs_list` length.

## Notes for the reviewer

- `all_models=True` must keep passing the full list through unsliced; the fix preserves that.
- Nothing changes for single-band or for `index_*_model_list=None`, which is the overwhelming
  majority of use, so the blast radius is small.
- The `NFW_MC` in-place mutation of `profile_kwargs_list[i]` is a separate latent problem
  (it writes into the caller's `kwargs_model`). The slicing fix makes it write to the *right*
  dict, but it still mutates caller state. Worth a follow-up that copies the dict before
  injecting `z_lens` / `z_source`; mention it in the PR body rather than expanding this change.
