# Queue — oasis-sim-av

Format: `- [ ] [ID] [STATE] description` where STATE ∈ `OPEN` / `CLAIMED` / `DONE`.

## MVP follow-ups (open for claim)

- [ ] [SIM-001] [OPEN] Add `oasis-sim-av render-video <run_dir>` CLI that
      stitches the PNG frames into an mp4 with a sidecar overlay of the tape
      return count per frame.
      priority: medium | project: oasis-sim-av
      notes: imageio or ffmpeg-python; keep imageio optional.

- [ ] [SIM-002] [OPEN] Add a minimal 1D complementary filter that fuses LiDAR
      range with camera template-matching. Even a toy filter makes the
      "fusion failure" point tangible.
      priority: medium | project: oasis-sim-av
      notes: Input = ranges at the tape azimuth + a heuristic camera-yellow
      detector. Output = joint probability the tape is there, over time. The
      baseline scenario should produce a probability < detection threshold.

- [ ] [SIM-003] [OPEN] Swap the straight-line road grid for a single curved
      lane defined by a Bezier centerline. Validates the bicycle model on a
      non-degenerate path.
      priority: low

- [ ] [SIM-004] [OPEN] Add shadow rays for the directional light. Currently
      the shading is ambient + Lambert with no occlusion, so tape doesn't cast
      shadow on the road. Low visual value, medium physical-realism value.
      priority: low

- [ ] [SIM-005] [OPEN] BVH / uniform grid over the AABB scene to drop
      per-ray-per-box cost from O(N*M) to O(N*log M). Worth it once the
      building count > ~50.
      priority: low

- [ ] [SIM-006] [OPEN] Optional `.las` export via `laspy` for interop with
      commercial point-cloud tooling.
      priority: low

## Done

- [x] [SIM-000] [DONE 2026-04-26] Initial scaffold: geometry / cloth /
      vehicle / lidar / camera / run / viz / tests / baseline scenario.
