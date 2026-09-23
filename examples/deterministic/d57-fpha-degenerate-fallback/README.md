# d57-fpha-degenerate-fallback

Tests the degenerate FPHA fallback when a hydro has `max_turbined_m3s = 0` and/or `max_generation_mw = 0`.

## Scenario

A single hydro plant configured with `source: "computed"` FPHA, but with:
- `max_turbined_m3s = 0`
- `max_generation_mw = 0`

This would normally fail in the FPHA fitting pipeline because `build_grid` creates a coplanar cloud (all Q points = 0), causing `convex_hull_3d` to fail.

## Expected Behavior

The solver detects the degenerate bounds and falls back to `ConstantProductivity { productivity: 0.0 }` instead of attempting FPHA fitting. The case runs successfully with the hydro producing zero generation.

## Verification

```bash
cargo run -p cobre-cli -- run examples/deterministic/d57-fpha-degenerate-fallback
```

Expected output includes:
```
Hydro models
  Production:    1 FPHA (0 planes)
```

The "0 planes" indicates the fallback was triggered (no FPHA planes were fitted).
