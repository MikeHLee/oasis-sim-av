# State — oasis-sim-av

**Focus:** SIM-001 (render-video CLI) and SIM-002 (complementary fusion filter)
complete; 43/43 tests pass. End-to-end baseline confirms the fusion failure:
`max_p_fused = 0.433` with detect threshold `0.5` (`frac_detected = 0`).
Public GitHub repo initialised at github.com/MikeHLee/oasis-sim-av.

**Last agent:** opencode (SIM-001/002 round)
**Last update:** 2026-04-26

**Blockers:** None.

**Next touchable items:**
- SIM-003 Bezier curved-lane (validates bicycle on non-degenerate path).
- SIM-004 Shadow rays for directional light.
- SIM-005 BVH / uniform grid scene acceleration (needed > ~50 buildings).
- SIM-006 `.las` export via laspy.

**Do not touch:**
- `oasis-firmware/simulation/` — different domain, different deps.
- Renderer still has no shadow pass or specular (deliberate MVP cut; SIM-004
  tracks the follow-up).
- Detection threshold + alpha in `FusionConfig` are tuned against the
  baseline scenario. Don't re-tune defaults without updating the
  `test_baseline_tape_stays_below_threshold` test.
