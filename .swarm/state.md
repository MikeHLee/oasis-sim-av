# State — oasis-sim-av

**Focus:** SIM-001..004 all complete; 52/52 tests pass in ~1.2s. Repo at
github.com/MikeHLee/oasis-sim-av has three embedded demos:
  - `docs/demo_baseline.gif`          fusion recovers (max p_fused = 0.977)
  - `docs/demo_heavy_rain.gif`        fusion collapses (max p_fused = 0.137)
  - `docs/demo_curved_road.gif`       Bezier pursuit + cast shadows

**Last agent:** opencode (SIM-001/002/003/004 round)
**Last update:** 2026-04-26

**Blockers:** None.

**Next touchable items:**
- SIM-005 BVH / uniform grid scene acceleration. Needed once building count
  > ~50; current scenes have ≤ 6. Defer until a scenario demands it.
- SIM-006 `.las` export via laspy. Pure plumbing; add when a downstream
  tool needs ASPRS-compliant clouds.

**Do not touch:**
- `oasis-firmware/simulation/` — different domain, different deps.
- Detection threshold + alpha in `FusionConfig` are tuned against the
  baseline scenario. Don't re-tune defaults without updating the
  `test_baseline_tape_stays_below_threshold` test.
- `LIGHT_DIR` and `AMBIENT` constants in `camera.py` are used by both the
  shader and `test_shadow_mask_*` geometry expectations. If you re-tune,
  update the tests' pole / light geometry to match.
