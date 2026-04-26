# State — oasis-sim-av

**Focus:** SIM-001..004 all complete; 52/52 tests pass in ~1.2s. Repo at
github.com/MikeHLee/oasis-sim-av has three embedded demos:
  - `docs/demo_baseline.gif`          fusion recovers (max p_fused = 0.977)
  - `docs/demo_heavy_rain.gif`        fusion collapses (max p_fused = 0.137)
  - `docs/demo_curved_road.gif`       Bezier pursuit + cast shadows

SIM-005 and SIM-006 marked DEFERRED in `queue.md` (2026-04-26). No open
work items remain against the MVP. Module is at a natural pause point
until a new scenario / downstream consumer changes the calculus.

**Last agent:** opencode (SIM-005/006 triage + defer)
**Last update:** 2026-04-26

**Blockers:** None.

**Next touchable items:**
- None currently. Un-defer SIM-005 only when a scenario with > ~50
  buildings lands, or a profile shows AABB-intersection cost dominating
  frame time. Un-defer SIM-006 only when an actual .las consumer appears.

**Do not touch:**
- `oasis-firmware/simulation/` — different domain, different deps.
- Detection threshold + alpha in `FusionConfig` are tuned against the
  baseline scenario. Don't re-tune defaults without updating the
  `test_baseline_tape_stays_below_threshold` test.
- `LIGHT_DIR` and `AMBIENT` constants in `camera.py` are used by both the
  shader and `test_shadow_mask_*` geometry expectations. If you re-tune,
  update the tests' pole / light geometry to match.
- Don't speculatively implement SIM-005 or SIM-006 — they are deferred
  by design, not by oversight. See queue.md notes before un-deferring.
