# State — oasis-sim-av

**Focus:** Handoff point. SIM-001..004 shipped; SIM-005/006 deferred;
SIM-007/008/009 spec **frozen in queue.md** but **not yet implemented**.
Next agent should open `.swarm/queue.md` and begin with SIM-007.

52/52 tests pass in ~1.2s. Three baseline demos embedded in `docs/`
(single-panel — to be superseded by grid by SIM-007).

**Last agent:** opencode (2026-04-26 spec-freeze session)
**Last update:** 2026-04-26

**Blockers:** None. All spec decisions resolved with user (see memory.md
entry "SIM-007/008/009 multi-view demo spec" for the 7 decisions made).

## Handoff notes for next agent

1. **Read first**: `.swarm/memory.md` — SIM-007/008/009 entry. It records
   7 design decisions that were resolved with the user; do not re-litigate
   them without fresh user input:
     - Layout is 5×2 (not 5×3)
     - BEV is world-fixed orthographic, per-scenario anchored
     - Detector is oracle with **condition-dependent** noise (not iid)
     - Rain clutter is **visual-only**, clean scans untouched
     - Droplets are advected, not uncorrelated
     - Fusion posterior stays as panel 5 (top-right)
     - curved_road demo is NOT re-rendered in this round

2. **Implementation order** (each PR-sized, each independently testable):
     - SIM-007 first (layout infra + BEVRenderer + reprojection helpers)
     - SIM-008 on top (oracle detector)
     - SIM-009 last (rain clutter)

3. **Critical regressions to preserve**:
     - `test_baseline_tape_stays_below_threshold` — fusion tuning
     - `FusionConfig` defaults (alpha, detect_threshold, lidar_peak)
     - Published `max_p_fused` numbers in README (0.977 / 0.137)
     - Clean `.ply` scan contents (no rain points in files)
     - `LIGHT_DIR` / `AMBIENT` constants (shadow test dependencies)

4. **New modules to add** (per spec in queue.md):
     - `src/oasis_sim_av/bev.py`       — BEVRenderer
     - `src/oasis_sim_av/overlays.py`  — reprojection + rasterisers
     - `src/oasis_sim_av/detect.py`    — OracleDetector (SIM-008)
     - `src/oasis_sim_av/rain.py`      — RainField (SIM-009)

5. **Existing modules to extend**:
     - `config.py`          add `BEVConfig`, `RainClutterConfig`
     - `run.py`             write `bev/NNNNNN.png` alongside frames
     - `render_video.py`    add `--layout grid5x2`
     - Three scenario YAMLs add `bev:` and (where relevant) `rain_clutter:`

6. **Bootstrap check for next session**:
     ```
     cd oasis-sim-av
     git status                  # should be clean
     pytest -v                   # baseline must be 52/52 green
     cat .swarm/queue.md         # read SIM-007 spec in full
     cat .swarm/memory.md | head -80   # read 7 decisions
     ```

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
- Don't re-open the 7 spec decisions listed in memory.md without new
  user input. They were each discussed and resolved deliberately.
- Don't merge rain clutter into clean scan output. Clean scan
  bit-identity with pre-SIM-009 is a load-bearing property — it is how
  we avoid re-tuning all the fusion baselines.
