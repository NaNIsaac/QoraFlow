# Synthetic input scenarios — QoraFlow redesign baseline

This is a **contract test harness**, not a substitute for measured PAL-XFEL data.

```sh
uv run python examples/synthetic_loader_cases.py --output /tmp/qoraflow-scenarios
uv run pytest -q tests/test_loader_scenarios.py
```

The generator creates tiny HDF5 inputs (8 frames, 3×4 pixels) in the existing
`PalXFELLoader` layout and a matching JSON truth file per case.

| Scenario | Perturbation | Expected result |
| --- | --- | --- |
| normal | All eight shots present | Eight correctly paired frames |
| shuffled_metadata | Metadata table rows rearranged | Original detector frame order recovered |
| missing_pump_flags | Two pump flags missing | Missing shots dropped, not classified as on |
| partial_metadata | Metadata for only half the shots | Only matched shots retained |
| no_qbpm_timestamp_overlap | QBPM timestamps disjoint | Explicit no-matching-data error |

The expected frame index, pump state, QBPM value and delay are written to
per-scenario JSON for independent checking. No private data are included.

## Architectural direction

A format-neutral *shot contract* should eventually make the integration code
independent of HDF5 and Zarr. Candidate logical fields:

- `images`: (shots, height, width) numeric array
- `timestamp`: shot identifier; alignment policy must be explicit
- `pump_state`: bool or missing, **never** silently coerce NaN to True
- `qbpm`: per-shot finite scalar or missing, with explicit handling policy
- `delay`: per-shot or per-file with declared unit, never inferred silently

This matrix currently tests **legacy HDF5 only**. It deliberately does not
invent a PAL Zarr schema: Zarr is a storage format with arbitrary array/group
paths, and real beamline data may use different layouts. Before developing
a Zarr reader, capture an anonymized tree of actual Zarr group/array paths,
shapes, dtypes, chunk shapes, compressor/codec metadata and shot identifiers.

## Next synthetic cases

Duplicate timestamps, QBPM channel mismatches, detector NaN/Inf and bad pixels,
nonmonotonic delays, pump-state imbalance, multi-file chunk boundaries, and
large logical arrays. Define an expected outcome (success, drop, or failure)
for each before changing the loader or integrator.
