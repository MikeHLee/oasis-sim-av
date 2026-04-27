# State — oasis-sim-av

**Focus:** SIM-007, SIM-008, SIM-009 shipped. Multi-view demo grid5x2
infrastructure complete. 78/78 tests pass.

**Last agent:** opencode (2026-04-26 multi-view demo implementation)
**Last update:** 2026-04-26

**Blockers:** None.

## Summary of completed work

### SIM-007: Multi-view 5×2 grid renderer + BEV
- Added `BEVConfig` to config.py and `BEVRenderer` in new `bev.py` module
- Created `overlays.py` with `reproject_points_to_camera`, `rasterise_lidar_bev`,
  `draw_bboxes`, `draw_fusion_strip`, and `compose_grid5x2` helpers
- Extended `run.py` to write `bev/NNNNNN.png` alongside frames when BEV configured
- Added `--layout grid5x2` to `render_video.py` for multi-view composition
- Added `bev:` config blocks to all three scenario YAMLs
- 16 tests in test_bev.py and test_overlays.py

### SIM-008: Oracle-projection detector
- Created `detect.py` with `OracleDetector` class
- Condition-dependent noise based on range, rain_dropout_prob, cloth velocity
- Detections recorded in state.jsonl per frame
- 5 tests in test_detect.py

### SIM-009: Rain field (advected droplets)
- Created `rain.py` with `RainField` class
- Added `RainClutterConfig` to config.py
- Droplets advect downward, recycle at ground
- 5 tests in test_rain.py

## New modules

| Module | Purpose |
|--------|---------|
| `src/oasis_sim_av/bev.py` | BEVRenderer - world-fixed orthographic top-down |
| `src/oasis_sim_av/overlays.py` | Reprojection, rasterisation, bbox, fusion strip, grid composition |
| `src/oasis_sim_av/detect.py` | OracleDetector - condition-modulated projection detector |
| `src/oasis_sim_av/rain.py` | RainField - advected droplet field for visual-only clutter |

## New config blocks

```yaml
bev:
  center: [15.0, 0.0]
  extent_m: 50.0
  size_px: 256
  show_vehicle_marker: true
  show_road: true

rain_clutter:
  enabled: false
  n_droplets: 200
  spawn_box: [10.0, -5.0, 0.0, 20.0, 5.0, 3.0]
  fall_velocity_m_s: 5.0
  jitter_std_m_s: 0.3
  droplet_radius_m: 0.02
```

## Tests

78 tests pass (was 52, added 26 new tests across 4 new test files).

## Next steps

- Re-render baseline and heavy_rain demos with `--layout grid5x2`
- Optionally integrate rain clutter into LiDAR scan for visualization
- Update README with new demo GIFs once re-rendered

## Do not touch (from prior session)

- Detection threshold + alpha in `FusionConfig` are tuned against the
  baseline scenario
- `LIGHT_DIR` and `AMBIENT` constants in `camera.py`
- Clean `.ply` scan contents (no rain points in files)
