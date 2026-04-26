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

## Done

- [x] [SIM-004] [DONE 2026-04-26] Shadow rays (`camera.shadow_rays`).
- [x] [SIM-003] [DONE 2026-04-26] Bezier-pursuit controller (`vehicle.py`).
- [x] [SIM-002] [DONE 2026-04-26] Complementary-filter fusion (`fusion.py`).
- [x] [SIM-001] [DONE 2026-04-26] Video rendering CLI (`render_video.py`).
- [x] [SIM-000] [DONE 2026-04-26] Initial scaffold: geometry / cloth /
      vehicle / lidar / camera / run / viz / tests / baseline scenario.
