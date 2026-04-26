# Memory — oasis-sim-av

Append-only record of non-obvious decisions. Newest first.

## 2026-04-26 — SIM-007/008/009 multi-view demo spec (frozen, not yet implemented)

User requested that the existing single-panel demo GIFs be replaced
with a multi-modal sensor-fusion visualisation. Full spec lives in
`queue.md` SIM-007/008/009; this entry records the non-obvious design
decisions made during the pre-coding conversation so the next agent
does not need to re-litigate them.

### Decision 1 — Layout is 5×2, NOT 5×3
User initially asked for "5×3 square grid". I pushed back: we only
have 4-5 genuinely distinct modalities (camera, LiDAR, detections,
fused, posterior/HUD) and two perspectives (vehicle-forward, world
BEV) that make sense. A 5×3 = 15-panel grid would force redundancy.
User agreed to 5×2: top row = vehicle camera views, bottom row = BEV.

### Decision 2 — BEV is world-fixed orthographic, per-scenario anchored
Not a chase camera, not a vehicle-following BEV. The camera is anchored
at a fixed world position per scenario (via `bev.center` + `extent_m`
in YAML) pointed at the decision intersection — the place where the
vehicle either continues through the tape or turns. This is the most
information-dense framing for "sensor fusion failed at this specific
moment"; a chase camera would keep the vehicle centred and lose the
fixed spatial reference. BEV re-uses the existing ray tracer with
orthographic primary rays (all rays parallel, pointing down), so no
new intersection code.

### Decision 3 — Detector is an oracle with CONDITION-DEPENDENT noise
User chose "oracle projection with noise" over colour-threshold or
scipy.ndimage.label. I flagged a risk: a naive oracle always produces
correct bboxes, which would make the detections panel look healthy
while the real camera is blind (sub-pixel tape, heavy rain). That
would be misleading — the whole point of the demo is that the camera
sees less than it should.

Resolution: noise is NOT iid Gaussian. It depends on the same physical
conditions that degrade the camera:
  - jitter ∝ range + bbox_width  (far, small tape → wiggly boxes)
  - score ∝ f(rain_dropout, cloth_velocity, range)
  - detection drop probability = (1 - score) * 0.5

This makes the oracle degrade in lockstep with what the camera would
"really" struggle with, without implementing a learned detector. It's
a deliberate stand-in. The docstring and README must call this out —
do not describe this as "object detection" full stop; it is
"condition-modulated projection."

### Decision 4 — Rain clutter is VISUAL-ONLY, clean scans untouched
User chose advected droplet field over uncorrelated short-range returns
or camera-only streaks. I flagged that integrating rain into the scan
stream would:
  (a) break test_baseline_tape_stays_below_threshold's tuning,
  (b) change the published max_p_fused numbers in the README,
  (c) require re-tuning FusionConfig defaults.

Resolution: two scan objects per sensor fire:
  - `scan_clean`  → fusion.py input, written to .ply, used by tests.
  - `scan_with_clutter` → ONLY used by the BEV / fused panels in the
                          renderer. Never persisted, never fed to
                          fusion.
Clean scans stay bit-identical to pre-SIM-009 output. Rain points tagged
`kind=3` so the BEV rasteriser can colour them cyan without touching
existing kind-0/1/2 code paths.

### Decision 5 — Advected droplets, not uncorrelated returns
User chose the more realistic option (advected field over uncorrelated
spawns). This costs ~20-30% extra per sensor fire at 200 droplets ×
16k rays = 3.2M AABB checks. Acceptable in numpy vectorisation.
Worth it because uncorrelated rain points would jitter between frames
and not look like rain — advected droplets produce visible streaks.

### Decision 6 — Fusion posterior stays in panel 5 (top-right)
We keep the existing `fusion.py` filter untouched. Panel 5 of the top
row plots the running `p_fused` time series as a growing strip chart,
so viewers can see both the current frame's sensor input AND the
integrated belief state. This keeps SIM-002 in the story rather than
burying it.

### Decision 7 — Do NOT re-render curved_road demo in same PR
SIM-007/008/009 re-generate the baseline and heavy-rain demos (those
are the fusion-failure stories). The curved_road demo stays on its
existing single-panel GIF because its purpose is to validate the
bezier-pursuit + shadow-ray features, not sensor fusion. Re-rendering
it with the grid layout would dilute the message. Can revisit later.

## 2026-04-26 — SIM-003 bezier_pursuit approximation

The `bezier_pursuit` controller uses a **simplified** pure-pursuit: it
returns `delta = clip(alpha, -max_delta, max_delta)` where `alpha` is the
heading error to the lookahead point. A physically exact pure-pursuit
law is `delta = atan(2*L*sin(alpha) / Ld)` with `L` = wheelbase. That
form requires the controller to know L, which is vehicle state, not
controller state. Passing L down would tighten coupling between
`VehicleControllerConfig` and `KinematicBicycle`.

The current form is correct in sign, tracks well at low speeds, and
respects `bezier_max_delta_rad`. At high speed/low radius you will see
understeer — the integration test
`test_bezier_pursuit_runs_on_bicycle` uses 4 m/s and a gentle S-curve to
stay inside the regime where this approximation holds (error < 6 m from
target endpoint after 8 s of closed-loop tracking).

Follow-up SIM-003-v2 would thread wheelbase through `make_controller`,
or alternatively move the pursuit logic onto a closed-loop
`PathFollower(bike, path)` object that has access to both.

## 2026-04-26 — SIM-004 shadow rays: ambient-only in shadow

The shadow shader reduces shaded-pixel output to `AMBIENT * base_color`
rather than fully clamping to black. This preserves colour continuity
across the shadow terminator so the road/ground/building all read as
the same material in shadowed vs lit regions, which is what we want for
sensor-domain realism. Tape (cloth) also drops to AMBIENT but keeps its
yellow colour — important because otherwise the camera's yellow-pixel
fusion signal would flicker as the tape enters/leaves shadow.

Shadow rays are off by default (`CameraConfig.shadow_rays = False`)
because they roughly double camera render time (one extra ray per hit
pixel). `scenarios/curved_road.yaml` sets them on, with
`motion_blur_samples: 2` (down from the baseline's 4) to offset.

## 2026-04-26 — SIM-002 complementary-filter design

The "1D complementary filter" in `fusion.py` is, strictly speaking, a
first-order low-pass over a weighted-sum measurement, not a classical
complementary filter (which splits a signal into high-pass / low-pass
components and sums them). We use the name deliberately because that is
what the brief requested, and the spirit is the same: combine two sensors
with different failure modes into one smoothed posterior.

Implementation notes that are not obvious from the code:

1. `alpha * dt >= 1` short-circuits the filter so that measurements pass
   through unchanged. This is needed because at 10 Hz sensor cadence (dt =
   0.1 s) with alpha = 6, `k = 0.6` — already quite responsive. The unit
   test `test_weighted_fusion_respects_bias` pins this behaviour with
   `alpha = 1000`.
2. Yellow detection uses a fixed chromaticity box (`R >= 120`, `G >= 90`,
   `B <= 110`, `R >= G`, `R - B >= 60`). It is deliberately looser than the
   tape colour because at oblique angles the Lambert shade drops to
   `AMBIENT = 0.22` and a strict colour match misses those pixels. Do not
   tighten without re-running `test_yellow_pixel_count_*`.
3. `lidar_peak = 6.0` is the "definitely detected" normalisation for
   `p_lidar`. This was picked from observed peaks in baseline runs (6–7
   rays). Scenarios with taller/closer tape will exceed this and clip
   `p_lidar` to 1.0, which is correct.
4. `detect_threshold = 0.5` is the cutoff used in
   `test_baseline_tape_stays_below_threshold`. Changing the default breaks
   the test — that is intentional: the default must keep the baseline
   sub-threshold.

## 2026-04-26 — SIM-001 render-video dependency posture

`render_video.py` degrades cleanly across three dependency tiers:

1. With `imageio[ffmpeg] + Pillow` → mp4 + text HUD.
2. With `imageio` only → gif fallback + Pillow-less numeric-block HUD.
3. With neither → per-frame annotated PNGs via matplotlib, or raw `.npy`
   if even matplotlib is missing.

`pyproject.toml` pins `imageio-ffmpeg` + `pillow` in the `[viz]` extra so
the happy path works out-of-the-box for anyone installing with
`pip install -e ".[viz]"`.

## 2026-04-26 — Initial scaffold

See `SIM_RATIONALE.md` for the full audit. Short version: `oasis-firmware`
simulation is a scalar signal-bus IoT sim at 100 ms tick granularity with
no geometry, sensors-in-world, or dynamics. `swarm-city` is a markdown-file
agent orchestration CLI. Neither is a viable base for AV sensor-fusion
simulation, so `oasis-sim-av` is a fresh sibling module. The only code
reused is a 12-line Gaussian noise pattern from
`oasis-firmware/simulation/behavioral/runtime.py:227-238`, now vectorised
in `src/oasis_sim_av/noise.py`.
