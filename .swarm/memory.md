# Memory — oasis-sim-av

Append-only record of non-obvious decisions. Newest first.

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
