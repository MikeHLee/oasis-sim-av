# State — oasis-sim-av

**Focus:** Initial scaffold complete; MVP demonstrates LiDAR ring-skip +
camera motion-blur failure mode on thin fluttering cloth.

**Last agent:** opencode (initial scaffold)
**Last update:** 2026-04-26

**Blockers:** None.

**Next touchable items:**
- Dial in `scenarios/police_tape_rain.yaml` so the failure mode is visually
  unmistakable when played back as a video montage.
- Add a simple motion-gated tape-hit counter to verify the failure mode is
  reproducible in state.jsonl.
- Optional follow-up: bolt on a minimal 1D Kalman filter that fuses LiDAR range
  with camera template matching, to show *fusion* (not just sensing) breaking.

**Do not touch:**
- `oasis-firmware/simulation/` for this work — it's a different domain.
- Renderer does not yet use shadow rays / specular — that is a deliberate MVP
  cut, not an oversight. Don't add a shadow pass without discussion.
