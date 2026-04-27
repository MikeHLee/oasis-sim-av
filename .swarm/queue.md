# Queue — oasis-sim-av

Format: `- [ ] [ID] [STATE] description` where STATE ∈ `OPEN` / `CLAIMED` / `DONE`.

## MVP follow-ups (open for claim)

- [ ] [SIM-001] [DONE 2026-04-26] Add `oasis-sim-av render-video <run_dir>` CLI that
      stitches the PNG frames into an mp4 with a sidecar overlay of the tape
      return count per frame.
      priority: medium | project: oasis-sim-av
      notes: entry point `oasis-sim-av-render-video`, HUD via Pillow with
      pure-numpy fallback, mp4 via imageio-ffmpeg with gif fallback.

- [ ] [SIM-002] [DONE 2026-04-26] Add a minimal 1D complementary filter that fuses LiDAR
      range with camera template-matching. Even a toy filter makes the
      "fusion failure" point tangible.
      priority: medium | project: oasis-sim-av
      notes: entry point `oasis-sim-av-fuse`. Yellow-box heuristic as the
      camera sensor. Baseline scenario produces max_p_fused < 0.5 (threshold).

- [ ] [SIM-003] [DONE 2026-04-26] Swap the straight-line road grid for a single curved
      lane defined by a Bezier centerline. Validates the bicycle model on a
      non-degenerate path.
      priority: low
      notes: Implemented as `bezier_pursuit` controller type (pure-pursuit
      on de-Casteljau-sampled cubic). Demo: `scenarios/curved_road.yaml`.

- [ ] [SIM-004] [DONE 2026-04-26] Add shadow rays for the directional light. Currently
      the shading is ambient + Lambert with no occlusion, so tape doesn't cast
      shadow on the road. Low visual value, medium physical-realism value.
      priority: low
      notes: `camera.shadow_rays: true` flag. ~2x render cost, default off.

- [ ] [SIM-005] [DEFERRED 2026-04-26] BVH / uniform grid over the AABB scene
      to drop per-ray-per-box cost from O(N*M) to O(N*log M). Worth it once
      the building count > ~50.
      priority: low
      notes: Deferred — current scenarios have ≤ 3 buildings (max 6 across
      the repo). O(N*M) with M ≤ 6 is not the bottleneck; camera ray count
      dominates. Revisit when a scenario adds > ~50 buildings, or when a
      profile shows building-intersection cost > 20% of frame time.

- [ ] [SIM-006] [DEFERRED 2026-04-26] Optional `.las` export via `laspy` for
      interop with commercial point-cloud tooling.
      priority: low
      notes: Deferred — pure plumbing on top of the existing `.ply` writer.
      Add when a concrete downstream consumer (e.g. a CloudCompare / QGIS
      / PDAL workflow) actually requires ASPRS-compliant output. Until
      then the extra `laspy` dependency is not justified.

## Multi-view demo upgrade (next session — spec frozen 2026-04-26)

Overarching goal: replace the current single-panel demo GIFs (camera
only) with a **5×2 panel grid** showing, per frame, vehicle-camera vs
world-BEV perspectives across 5 modalities (raw, detections,
reprojected LiDAR, fused, HUD/posterior). Scope split into three items
below. Tackle in order — SIM-007 must land first; SIM-008 and SIM-009
each stand alone on top of it.

- [x] [SIM-007] [DONE 2026-04-26] Multi-view 5×2 grid renderer + BEV camera + LiDAR reprojection
      priority: high | project: oasis-sim-av
      notes: Implemented BEVRenderer in bev.py, overlays.py with reprojection
      and rasterisation helpers, extended render_video.py with --layout grid5x2,
      added BEV config blocks to all three scenarios. 26 new tests pass.

- [x] [SIM-008] [DONE 2026-04-26] Oracle-projection detector with condition-dependent noise
      priority: high | project: oasis-sim-av
      notes: Implemented OracleDetector in detect.py with condition-dependent
      noise based on range, rain_dropout_prob, and cloth velocity. Detections
      recorded in state.jsonl. 5 tests pass.

- [x] [SIM-009] [DONE 2026-04-26] Advected droplet field for visual-only LiDAR rain clutter
      priority: medium | project: oasis-sim-av
      notes: Implemented RainField in rain.py with advected droplets, recycling
      at ground. RainClutterConfig added to config.py. 5 tests pass including
      clean scan unchanged regression. Visual-only clutter pass NOT yet integrated
      into lidar.py scan output (left for future when actually needed for viz).

        Tests:
          - test_rain_clutter_points_have_kind_3
          - test_clean_scan_unchanged_when_rain_enabled
          - test_rain_advection_loops_at_ground (after N steps no droplet
            below ground_z)
          - Regression: existing baseline fusion test still produces
            max_p_fused > threshold unchanged.

## Multi-view demo follow-ups + safety-aware control (active 2026-04-27)

- [x] [SIM-010] [DONE 2026-04-27] Wire up stub grid5x2 panels (3 camera+LiDAR
      reprojection, 4 fused, 8 LiDAR BEV, 9 fused BEV) that were previously
      placeholders. Integrated rain clutter into the visualisation pipeline
      via `lidar_viz/NNNNNN.npz` sidecar (clean `.ply` untouched per
      Decision 4). `lidar/NNNNNN.npz` added as lossless sidecar for kind
      preservation. 2 new integration tests: panels 3/4/8/9 assert
      non-trivial pixel content; rain-augmented viz scan contains kind=3
      points while clean PLY remains rain-free.

- [x] [SIM-011] [DONE 2026-04-27] Promoted fusion filter to in-loop
      (step-by-step). Controllers now accept `percept` kwarg carrying
      `p_fused`, `p_lidar`, `p_camera`. Added `cautious` flag + three
      params (`cautious_p_threshold`, `cautious_min_v_frac`,
      `abstain_p_threshold`) to `VehicleControllerConfig`. Any base
      controller can run cautiously — velocity scales linearly with
      `p_fused / p_threshold`, clipped to `min_v_frac`. Steering is
      untouched. Active-learning queue: frames with `p_fused <
      abstain_p_threshold` are serialized to `<run_dir>/abstain.jsonl`.
      11 new tests in `test_cautious_control.py`. End-to-end: heavy-rain
      cautious run travels 1.08 m vs 9.95 m aggressive in 2 s (9× less
      distance when fusion collapses).

## MVP clean-ups and enhancements (done 2026-04-27)

- [x] [SIM-012] [DONE 2026-04-27] Add `scenarios/police_tape_cautious.yaml`
      with bezier_pursuit + cautious=true against heavy-rain world. Makes
      the README's "9.95 m → 1.08 m" comparison one CLI command away.
      priority: high | project: oasis-sim-av

- [x] [SIM-013] [DONE 2026-04-27] Abstention reason taxonomy. Entries in
      abstain.jsonl now carry prioritized reasons: cloth_velocity_excessive,
      lidar_dropout_rate_high, n_detections_flicker, p_fused_below_threshold.
      Useful for downstream labeller stratifying frames by failure mode.
      priority: high | project: oasis-sim-av

- [x] [SIM-014] [DONE 2026-04-27] Percept-aware bezier_pursuit. New
      `_wrap_cautious_bezier` tightens max_delta at low confidence to prevent
      aggressive steering under uncertainty.
      priority: medium | project: oasis-sim-av

- [ ] [SIM-015] [DEFERRED] EKF/UKF on p_fused. First-order ComplementaryFilter
      is out of scope per CONTEXT.md; flagged for SIM-v2 milestone.
      priority: low
      notes: See memory.md Decision 8.

- [x] [SIM-016] [DONE 2026-04-27] Fix grid5x2 macro-block warning. Pad total_h
      to be divisible by 16 in `compose_grid5x2`.
      priority: medium | project: oasis-sim-av

- [x] [SIM-017] [DONE 2026-04-27] Empty abstain.jsonl behaviour documented in
      memory.md Decision 7 — intentional for simpler tooling contract.
      priority: low

- [x] [SIM-018] [DONE 2026-04-27] Add max_detection_score to percept dict.
      Enables controllers to distinguish "oracle detector sees tape but
      fusion disagrees" from "detector itself is uncertain".
      priority: medium | project: oasis-sim-av

- [x] [SIM-019] [DONE 2026-04-27] Re-render curved_road.yaml with grid5x2.
      ~2 min render, produces docs/demo_curved_road_grid5x2.mp4.
      priority: low

## Done

- [x] [SIM-019] [DONE 2026-04-27] Curved road grid5x2 re-render.
- [x] [SIM-018] [DONE 2026-04-27] max_detection_score in percept.
- [x] [SIM-016] [DONE 2026-04-27] grid5x2 macro-block fix.
- [x] [SIM-014] [DONE 2026-04-27] Percept-aware bezier_pursuit.
- [x] [SIM-013] [DONE 2026-04-27] Abstention reason taxonomy.
- [x] [SIM-012] [DONE 2026-04-27] Cautious-mode demo scenario.
- [x] [SIM-011] [DONE 2026-04-27] In-loop fusion + cautious controller.
- [x] [SIM-010] [DONE 2026-04-27] Grid5x2 panel wiring.
- [x] [SIM-009] [DONE 2026-04-26] Rain field for visual-only clutter.
- [x] [SIM-008] [DONE 2026-04-26] Oracle-projection detector.
- [x] [SIM-007] [DONE 2026-04-26] Multi-view 5×2 grid renderer.
- [x] [SIM-004] [DONE 2026-04-26] Shadow rays (`camera.shadow_rays`).
- [x] [SIM-003] [DONE 2026-04-26] Bezier-pursuit controller (`vehicle.py`).
- [x] [SIM-002] [DONE 2026-04-26] Complementary-filter fusion (`fusion.py`).
- [x] [SIM-001] [DONE 2026-04-26] Video rendering CLI (`render_video.py`).
- [x] [SIM-000] [DONE 2026-04-26] Initial scaffold: geometry / cloth /
      vehicle / lidar / camera / run / viz / tests / baseline scenario.
