# Why `oasis-sim-av` is a fresh sibling repo, not an extension of oasis-firmware

Audit date: 2026-04-26.  Summarised from the in-flight concentrated audit.

## Asked the question
Could an autonomous-driving sensor-fusion edge-case simulator (3D city,
kinematic bicycle, mass-spring cloth, LiDAR + camera ray tracers, Gaussian
noise, crime-scene-tape failure mode) be built inside existing oasis-x
repositories?

## Found
| Candidate repo | What it actually is | Reusable content |
|---|---|---|
| `oasis-firmware/simulation/` | Scalar signal-bus IoT sensor behavioural sim, 100 ms ticks, YAML-configured components, Gaussian / uniform / drift noise, fault injection. Pure Python stdlib + `pyyaml`. No geometry, no 3D, no rendering, no dynamics. | **12-line Gaussian noise function** (`behavioral/runtime.py:227-238`), copied + vectorised to `src/oasis_sim_av/noise.py`. |
| `swarm-city/` | Markdown-file agent orchestration CLI (click + MCP). "Swarm" is developer-agent metaphor, "city" is metaphor for sibling repos. Zero physics. | None — it's a developer tool, not a simulator. |
| `oasis-weather/`, `oasis-cloud/`, `oasis-home/`, etc. | Product domains (aviation weather dashboard, cloud backend, home IoT app). | Irrelevant to an AV sim. |

## Decision
Build `oasis-sim-av/` as a **new sibling** at `oasis-x/oasis-sim-av/`.

Rationale:
1. **Substrate mismatch.** `BehavioralRuntime` is dict-of-scalars on a coarse
   tick. Retrofitting geometry + integrators + ray tracers onto it would
   distort both codebases.
2. **Dependency posture.** `oasis-firmware` is a firmware codegen project.
   Dragging `numpy`, `matplotlib`, `imageio` into its dependency surface is
   dead weight for its actual users.
3. **Domain boundary.** Firmware sim = "what would this chip read?". AV sim =
   "what does the world look like from a moving vehicle?". Different
   abstractions, different consumers.
4. **Clean tests and release cadence.** A new module can have its own pypi
   entry, its own CI, its own version line.

## Cost of the decision
~2000 LoC fresh + zero code deleted from other repos + one new line in the
root `.swarm/context.md` (division list).

## What we explicitly did NOT do
- Did not modify `oasis-firmware/simulation/`.
- Did not modify `swarm-city/`.
- Did not touch any file under `oasis-x/.swarm/` (root agent orchestration).
- Did not invent a new MCP server; the initial MVP is CLI + library only.
  See `.swarm/queue.md::SIM-002` for a suggested MCP hookup follow-up if the
  sim grows into an AI-driven scenario laboratory.
