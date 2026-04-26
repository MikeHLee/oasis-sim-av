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

- [ ] [SIM-007] [OPEN] Multi-view 5×2 grid renderer + BEV camera + LiDAR reprojection
      priority: high | project: oasis-sim-av
      depends: none
      notes:
        Layout (10 tiles per frame):

          TOP ROW — forward-facing vehicle camera:
            1. Camera RGB (raw forward view — existing frames/*.png)
            2. Camera + 2D bboxes (oracle detector, from SIM-008)
            3. Camera + reprojected LiDAR points (per-point range-coloured)
            4. Fused: camera + bboxes + reprojected LiDAR + rain clutter (SIM-009)
            5. Fusion posterior strip (running p_fused over time from fusion.py)

          BOTTOM ROW — world-fixed bird's-eye orthographic:
            6. BEV ground truth (geometry + vehicle marker + tape + road)
            7. BEV + driven-path trail
            8. LiDAR BEV, colour-coded by kind (ground/building/tape/rain)
            9. Fused BEV: ground-truth silhouettes + LiDAR + rain clutter
            10. Legend / HUD (t, speed, p_fused, detected-box count)

        Work items for SIM-007:
          a. Add `BEVRenderer` (world-fixed orthographic top-down) and
             `bev` config block to ScenarioConfig:
               bev:
                 center: [x, y]        # world-frame anchor
                 extent_m: 40.0        # side length of square viewport
                 size_px: 256
                 show_vehicle_marker: true
                 show_road: true
             BEV re-uses the existing ray tracer by casting orthographic
             rays straight down onto world.boxes, world.ground_z, road
             polygons, and cloth triangles.
          b. Integrate BEV into run.py — alongside `frames/NNNNNN.png`,
             write `bev/NNNNNN.png` at the same sensor cadence.
          c. Add two helpers in a new `overlays.py`:
               - `reproject_points_to_camera(points_xyz, camera, veh_origin,
                  veh_R) -> (u, v, mask)` applying the same pinhole
                  geometry as camera._primary_rays.
               - `rasterise_lidar_bev(points, kinds, bev_cfg) -> HxWx3 uint8`
                 ground=grey, building=blue, tape=yellow, rain=cyan.
          d. Extend render_video.py with `--layout grid5x2` that composes
             the 10 panels per frame into one image, adds a thin title
             strip per tile (≈18 px) and a single-line footer with t/
             speed/p_fused/n_boxes. Reuse existing HUD font helpers.

        Per-scenario BEV config is required — the three existing
        scenarios need `bev:` blocks added. Suggested defaults:
          police_tape_rain / heavy_rain:  center=[15, 0], extent=50
          curved_road:                    center=[17.5, 0], extent=60

        Tests:
          - test_bev_renders_nonzero_scene (building + tape visible)
          - test_reprojection_round_trip  (world pt -> pixel -> ray hits
            within 1 px)
          - test_grid_layout_shape        (composed frame has the right
            HxW dims for the configured panel size)

- [ ] [SIM-008] [OPEN] Oracle-projection detector with condition-dependent noise
      priority: high | project: oasis-sim-av
      depends: SIM-007 (uses reprojection helper from overlays.py)
      notes:
        Not a real learned detector — deliberately chosen so the demo
        decouples "detector quality" from "camera visibility under
        rain / sub-pixel tape". New module `detect.py`:

          OracleDetector.detect(world, cloth, vehicle, camera, conditions)
            -> list[BBox(xmin, ymin, xmax, ymax, score, class)]

        Algorithm:
          1. Compute the 8 corners of the tape's bounding region in world
             frame (from cloth.positions convex hull or min/max corners).
          2. Reproject all 8 corners into camera image space using
             overlays.reproject_points_to_camera. Drop behind-camera.
          3. Take axis-aligned (u, v) bbox of remaining corners, clip to
             image bounds.
          4. Reject if resulting bbox area < 4 px  OR  if majority of
             corners are behind camera  →  detection drops out.
          5. Apply condition-dependent noise to the surviving bbox:
               - xy jitter σ = max(1, bbox_width * 0.05 + range_m * 0.02) px
               - size jitter σ = max(2, bbox_width * 0.1) px
               - score = clip(base_score - 0.15*rain_dropout_prob -
                             0.05*cloth_rms_velocity - 0.1*range_norm, 0, 1)
               - drop entirely with prob = (1 - score) * 0.5
          6. Write detections into state.jsonl as:
               "detections": [{
                 "class": "tape", "bbox": [xmin,ymin,xmax,ymax],
                 "score": 0.82
               }, ...]

        Why condition-dependent: per the discussion, "always-correct"
        oracle boxes would make panel 2 look healthy while the real
        camera is blind — misleading. The noise model makes the oracle
        degrade with the same physical conditions that degrade the real
        camera (range, motion-blur, rain) so the panel still tells the
        visibility story without implementing YOLO.

        Tests:
          - test_oracle_detects_tape_in_baseline (non-empty detections
            in first half of scenario where tape is in view)
          - test_oracle_drops_in_heavy_rain_far_range (empty or low-score
            detections past t=2s in police_tape_heavy_rain)
          - test_detection_bbox_inside_image_bounds

- [ ] [SIM-009] [OPEN] Advected droplet field for visual-only LiDAR rain clutter
      priority: medium | project: oasis-sim-av
      depends: SIM-007 (rain points rendered in BEV and fused panels)
      notes:
        Adds a second, visual-only LiDAR pass so the fused panels show
        "LiDAR bouncing off rain." Clean scans (passed to fusion.py,
        written to .ply) stay untouched — this is critical to preserve
        the existing test_baseline_tape_stays_below_threshold and the
        published fusion numbers.

        New config block in lidar YAML:
          rain_clutter:
            enabled: true
            n_droplets: 200
            spawn_box: [10.0, -5.0, 0.0, 20.0, 5.0, 3.0]  # xyz min/max
            fall_velocity_m_s: 5.0
            jitter_std_m_s: 0.3
            droplet_radius_m: 0.02

        Runtime:
          - `RainField` class in `rain.py`. Holds droplet positions (Nx3)
            and velocities. `step(dt)` advects downward, recycles any
            droplet whose z < ground_z back to top of spawn_box with
            fresh xy jitter.
          - Each sensor fire: do a second LiDAR pass over *only* the
            droplet AABBs (each droplet = 2*radius AABB, O(N_rays * N_drops)
            — 200 drops × 16k rays = 3.2M checks, acceptable in numpy).
            Tag hits with kind=3.
          - Merge rain points into a separate `ScanWithClutter` object
            used only by the visualisation layer; the canonical `scan`
            fed to fusion.py stays the clean one.

        Cost budget: ~20–30% added per sensor fire. Still well under 1s
        per frame at default resolution.

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
