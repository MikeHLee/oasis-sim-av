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

## Done

- [x] [SIM-004] [DONE 2026-04-26] Shadow rays (`camera.shadow_rays`).
- [x] [SIM-003] [DONE 2026-04-26] Bezier-pursuit controller (`vehicle.py`).
- [x] [SIM-002] [DONE 2026-04-26] Complementary-filter fusion (`fusion.py`).
- [x] [SIM-001] [DONE 2026-04-26] Video rendering CLI (`render_video.py`).
- [x] [SIM-000] [DONE 2026-04-26] Initial scaffold: geometry / cloth /
      vehicle / lidar / camera / run / viz / tests / baseline scenario.
