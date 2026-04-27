# State — oasis-sim-av

**Focus:** SIM-012 through SIM-019 (except SIM-015 deferred) shipped. 91/91 tests pass.

**Last agent:** opencode (2026-04-27 SIM-012/013/014/016/017/018/019)
**Last update:** 2026-04-27

**Blockers:** None.

## Summary of this session

### SIM-012: Cautious-mode demo scenario
- Created `scenarios/police_tape_cautious.yaml` with bezier_pursuit + cautious=true
- Identical world to heavy_rain but wires up safety behaviour via CLI
- README "9.95 m → 1.08 m" comparison now one command away

### SIM-013: Abstention reason taxonomy
- Expanded abstain.jsonl reasons from single to prioritized taxonomy:
  1. cloth_velocity_excessive (RMS > 3 m/s)
  2. lidar_dropout_rate_high (> 2x baseline + 0.1)
  3. n_detections_flicker (detector dropped vs prior frame)
  4. p_fused_below_threshold (catch-all)
- Added `_classify_abstain_reason()` helper in run.py

### SIM-014: Percept-aware bezier_pursuit
- New `_wrap_cautious_bezier()` tightens max_delta at low p_fused
- Delta scale = 0.6 + 0.4 * (p/p_thresh) when below threshold
- Registered in `make_controller()` for bezier_pursuit + cautious

### SIM-015: EKF/UKF on p_fused (DEFERRED)
- Documented in memory.md Decision 8 as SIM-v2 milestone
- ComplementaryFilter remains first-order low-pass per CONTEXT.md

### SIM-016: grid5x2 macro-block fix
- `compose_grid5x2()` now pads total_h to be divisible by 16
- Eliminates imageio "resizing from (1600, 572) to (1600, 576)" warning

### SIM-017: Empty abstain.jsonl documentation
- Already documented in memory.md Decision 7 — confirmed intentional

### SIM-018: max_detection_score in percept
- Added to percept dict each frame
- Enables distinguishing "oracle sees tape but fusion disagrees"

### SIM-019: Curved road grid5x2 re-render
- Generated docs/demo_curved_road_grid5x2.mp4 (20 frames at 10 fps)
- Shows bezier_pursuit path trail nicely in panel 7

## Next steps

- Add test for SIM-013 abstention taxonomy (classify_abstain_reason)
- Add test for SIM-014 cautious_bezier max_delta scaling
- Consider: add test for max_detection_score in percept

## Do not touch (from prior sessions)

- Detection threshold + alpha in `FusionConfig` are tuned against the
  baseline scenario (tests/test_fusion.py).
- `LIGHT_DIR` and `AMBIENT` constants in `camera.py`.
- Clean `.ply` scan contents — must not contain rain points (Decision 4).
- The pre-SIM-011 controller callable signature `(t, state) -> (v, d)`
  is preserved internally in `_make_base_controller`; external callers
  should use the wrapped callable via `make_controller` and the new
  3-arg form `(t, state, percept=...)`.
