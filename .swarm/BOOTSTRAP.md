See parent `oasis-x/.swarm/BOOTSTRAP.md` for the full protocol.

Division-local notes:
- This repo is `oasis-sim-av`, a Python simulator for AV sensor-fusion edge
  cases. It is a new sibling of `oasis-firmware` / `swarm-city` etc.
- Local dev: `pip install -e .[viz,dev]` then `pytest -v`.
- Smoke run: `oasis-sim-av scenarios/police_tape_rain.yaml --duration 1.0`.
