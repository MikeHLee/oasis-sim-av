# Memory — oasis-sim-av

Append-only record of non-obvious decisions. Newest first.

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
