# VLA Forge

Type "have a robot do X". A team of Fetch.ai uAgents turns the sentence into a task, builds the scene,
collects expert demonstrations, fine-tunes a vision-language-action policy, measures it on held-out scenes,
reads the failures, retrains if there is time, and publishes the result as a new callable skill agent.
All of it runs locally on one Mac (M5 Pro, 64 GB), inside a 3 minute budget per prompt.

```
uv sync
HF_HUB_OFFLINE=1 uv run vla-forge          # http://127.0.0.1:8100
```

The first start needs the base policy (`models/base.pt`, see "Pretraining" below) and the two Hugging Face
encoders in the local cache (`facebook/dinov2-small`, `google/siglip2-base-patch16-224`). The OpenAI key is
read from the environment or the macOS Keychain (`OPENAI_API_KEY`); it is never written to disk.

## What happens to one prompt

```
browser ── POST /api/prompt ──> gateway agent
                                   │ ChatMessage (uAgents chat protocol)
                                   v
 task agent ── TaskReady ──> scene agent ── SceneReady ──> data agent ── DemosReady ──> trainer agent
   ^  (LLM: sentence -> task spec       (MuJoCo SO-101,      (scripted expert,          (fine-tune the VLA
   |   + success predicates)             20 held-out layouts)  domain randomization,     from the nearest
   |                                                           DART noise)               skill or the base)
   |                                                                                          │ PolicyTrained
   |                                                                                          v
   |  SkillPublished      registry agent <── Publish ── critic agent <── EvalReport ── eval agent
   └──────────────────── (saves the skill,              (LLM reads failures:         (20 rollouts of the
                          starts skill-<id> agent)       retrain near failure         trained policy,
                                                         spots, or publish)           success measured)
```

Every arrow is a real uAgents message between agents in one `Bureau` (`src/vla_forge/agents.py`). The heavy
work behind each agent lives in `src/vla_forge/pipeline.py` and runs in worker threads, so the agents' event
loop stays responsive. Each stage streams its logs, demo counts, loss and evaluation to the page over SSE.

**The recursion.** Every published skill becomes its own chat-protocol agent (`skill-<id>`), callable from the
page ("Call skill agent") or by any other agent. The next prompt starts from the nearest existing skill (SigLIP text
similarity of the instructions plus a same-motion bonus) and falls back to the multi-task base policy, so the
library grows from itself. Every fine-tune mixes in replay of the base data so older skills are not forgotten.

## The VLA

Inputs: two camera images (front and wrist, 168 px), the instruction, the six joint positions.
Output: a chunk of the next 10 joint targets (1 s at 10 Hz), executed with temporal ensembling.

| part | what | trainable |
|---|---|---|
| vision | DINOv2-small, frozen, fp16 on MPS; 37 tokens x 384 per camera (CLS + 6x6 pooled patches) | no |
| language | SigLIP2 base text tower, frozen, 768-d | no |
| action head | FiLM language conditioning of the vision tokens, Perceiver read-out (8 latents per camera), 4-layer d=256 transformer, 10 action queries + 1 auxiliary query (target block and goal position) | 4.93 M params |

Training: L1 on normalised joint chunks plus 0.5 x L1 on the auxiliary positions, AdamW, bf16 autocast on MPS.
Image features are computed once per frame as demos arrive, so training only runs the head (~22 steps/s at
batch 256).

## Pretraining (once, ahead of time)

```
VLA_FORGE_TIMED=1 HF_HUB_OFFLINE=1 scripts/heavy.sh uv run python -u scripts/pretrain_base.py --episodes 2400 --seconds 2400
```

Expert demos for random (skill, colour, target) tasks with domain randomization and DART noise, encoded into
feature shards under `models/base_data/`, then the head is trained and measured per skill (`models/base.json`).
This is the root of the skill library and the replay buffer for every later fine-tune.

## The scaling ladder

Prompts come in three tiers, and the page tags each one.

- Tier 1, one motion: reach, push, lift.
- Tier 2, carry a block somewhere: put it in a bowl, stack it on another block.
- Tier 3, composed tasks: several tier 1-2 steps in one scene ("sort the blue and green blocks into the bowls
  of the same color").

A tier 3 prompt goes to the planner agent instead of the single-skill chain:

```
task agent ── ComposeTask ──> planner agent ── SkillQuery ──> skill-<id> agents (one per step that exists)
                                 │    <── SkillOffer (measured success rate) ──┘
                                 │ step with no skill: a child run through the normal one-prompt pipeline
                                 │    <── StepTrained (from the task agent when the child publishes)
                                 v
                              executor agent: runs every step's own policy back to back in 20 held-out
                                 composed scenes, then Publish -> registry (the composite becomes a skill too)
```

The planner (OpenAI) splits the sentence into steps (`compose.py` checks that the scene can hold them). Each
step either reuses a published skill agent or is trained on the fly in under 3 minutes, starting from the
nearest skill. A step trained for a composed task collects its demos and held-out checks in that task's own
scenes (`compose.step_layout`): the other bowls and squares are on the table, and half the time the earlier
steps are already done. Without that, the green bowl skill below scored 18/20 on single-skill tables and 5/20
inside the sort task, and the whole task went 0/20; with it, the same prompt went to 10/20.

In the composed run each step's policy runs until its own success check holds, then homes, and the next
step's policy takes over. A rollout counts as a success only if every step's check is true at the end.

## Measured results

Mac mini (M5 Pro, 64 GB), timed from the click to the published skill agent, success on 20 held-out scenes.
Runs in order, so later prompts can start from earlier skills (`runs/ladder.jsonl`, `scripts/ladder.py`).

| tier | prompt | started from | rounds | time | success |
|---|---|---|---|---|---|
| 1 | push the green block onto the blue square | base | 9/20, retrain, 15/20 | 164.0 s | 15/20 |
| 1 | pick up the yellow block and hold it up | base | 19/20 | 79.6 s | 19/20 |
| 2 | put the blue block in the blue bowl | base | 15/20, retrain, 17/20 | 165.5 s | 17/20 |
| 2 | put the red block in the white bowl | skill "put the blue block in the blue bowl" | 14/20, retrain, 16/20 | 166.4 s | 16/20 |
| 2 | stack the orange block on the red block | base | 12/20, retrain, 14/20 | 171.6 s | 14/20 |
| 3 | put the red block in the white bowl and push the green block onto the blue square | reused 2 skill agents | step 1 17/20, step 2 11/20 | 26.4 s | 9/20 |
| 3 | sort the blue and green blocks into the bowls of the same color | reused blue, trained green (88.8 s) | step 1 13/20, step 2 14/20 | 114.3 s | 10/20 |
| 3 | push the red block onto the orange square and stack the yellow block on it | trained both (166.1 s, 168.1 s) | step 1 14/20, step 2 3/20 | 360.0 s | 2/20 |

Earlier skills speed up later prompts: the tier 3 task made only of existing skills finished in 26 s, the one
that reused one skill and trained one took 114 s, and the one that had to train both took 360 s. The weak spot
is stacking onto a block that an earlier step just pushed (3/20).

The demo recording (`demo/vla-forge-demo-v2.mp4`) is a separate live take of two prompts: "touch the red
block" from the base, 20/20 in 70.9 s, then the sort task again, reusing blue and training green (17/20 on its
own, 96.8 s), 10/20 end to end in 122.4 s. Two other takes of the sort task after the scene fix measured 10/20
and 6/20.

Base policy v2 (pretrained once: 3200 expert attempts, 2783 kept, 252,754 frames, 66,419 steps in 3000 s,
final loss 0.039), measured on held-out layouts per motion:

| motion | v2 | v1 |
|---|---|---|
| reach | 20/20 | n/a |
| push | 12/20 | 14/20 |
| lift | 19/20 | 18/20 |
| pick and place | 14/20 | 18/20 |
| stack | 25/40 (62.5%) | 7/20 |

**What moved stacking.** v1 stacked 7/20 from the base and stalled at 50% after a 3 minute fine-tune. Three
changes: the expert's DART noise now fades out ("calm") while it squeezes the block and while it places it,
and it holds 0.3 s before letting go (the expert itself went from 14/30 to 24/30 stacks, so the demos stopped
teaching sloppy placement); stack demos are weighted 0.32 in pretraining; and stack scenes carry distractor
fixtures like the other motions. Base stacking went from 7/20 to 25/40, and the per-prompt stack skill from
50% to 14/20 (70%).

Ablation (v1, pick and place, 356 demos, 300 s training, 30 held-out layouts): with a grounding token 24/30,
without 25/30, so the grounding token is off by default.

## Interface

The page is modelled on the public website of HeyPCB (heypcb.ai): top bar with breadcrumb and segmented tabs,
chat on the left with tool status pills, floating tool groups on the canvas, a stage rail, a bottom tab bar for
rollouts, an inspector on the right, zinc palette with one safety-orange accent, Geist and Geist Mono only. HeyPCB's GitHub repositories
hold hardware (CERN-OHL-S) and an MIT MCP server but no interface code, so nothing was copied; the look was
rebuilt with this project's own HTML, CSS and JavaScript, with no HeyPCB name or logo in the page. Not
affiliated. Details and other credits in `NOTICE`.

## Custom robots (the Robot tab)

The Robot tab builds the embodiment before a skill is trained, in the spirit of Etch's part picker: pick an arm,
a wrist option and an end effector, see each part as a 3D preview and the assembled robot live, read the checks,
then press "Train the next skill on this robot". Every part is a published open-source MuJoCo model; sources,
licences and the pinned Menagerie commit are in `SOURCES.md`. Code: `src/vla_forge/sim/robots.py`.

- **Robot id.** `<arm>.<wrist>.<effector>`, for example `vx300s.native.native` or `wx250s.native.robotiq_2f85`.
  The stock arm is plain `so101`. Each published skill stores its robot in `Skill.robot` (default `"so101"`), and
  nearest-skill lookup only fine-tunes from skills trained on the same robot.
- **Assembly.** `robot_spec(id)` composes the source MJCF with `MjSpec.attach_body`: the arm's own jaw is removed
  when another effector replaces it, the effector is mounted on the last wrist link, a locked wrist drops its roll
  joint and actuator, and a `tcp` site marks the tool centre. `calibrate(id)` then measures in sim: finger opening
  at the fingertips (mm), gripper command range, a top-down home pose.
- **Checks** (`report(id)`, shown in the tab): the model compiles; every arm joint has sane limits with home
  inside; the action size; fingers open wider than a 3 cm block and close below it; top-down reach over the table
  workspace (7 x 9 grid, r 0.125-0.285 m, +/-38 deg) at grasp, push and hover height; gravity torque with a 30 g
  block at full reach against the source model's actuator limits; no self-collision at home.
- **Training on it.** Demos come from the same scripted expert, solving IK on the new arm. If the action size
  matches the base policy (6), every layer is kept and fine-tuned. If it differs (7 for the Trossen arms, 5 with
  a locked wrist on a 5-joint arm), `VLA.for_embodiment` keeps the frozen encoders and every head weight whose
  shape still fits, re-initialises the state input and action output layers, and resets the action
  normalisation from the new robot's demos; the run's `init_from.reason` says which. SO-101 replay data is not
  mixed into a custom robot's training.

Measured 2026-11-06 on the ViperX 300 (`vx300s.native.native`, 7-D actions, base policy carried over with 83 of
86 weight tensors kept), one prompt each, 20 held-out scenes, under `taskpolicy -b`: "push the green block onto
the blue square" 8/20 then 14/20 (70%) in 2:46; "pick up the red block and hold it up" 2/20 then 4/20 (20%) in
2:37. Both below the 80% target: two short rounds are not enough to relearn the action layer for grasping. The
scripted expert itself lifts on this arm in 6 of 6 seeds, so the gap is training time, not the robot.

### Router hook: `robot_feasibility`

```python
from vla_forge.sim.robots import robot_feasibility

ok, reason = robot_feasibility("koch.native.native", "pick_place")
# (False, "it reaches only 57% of the table workspace at grasp height pointing down (needs 90%); ...")
```

`task` may be a skill name (`"reach"`, `"push"`, `"lift"`, `"pick_place"`, `"stack"`), a `TaskSpec`, or a dict
with a `"skill"` key. It returns `(ok, reason)` with a plain sentence either way. In order it checks: the parts
compose; for grasping skills, the fingers open at least 38 mm and close below 28 mm; at least 90% of the workspace
is reachable pointing down at the height the skill works at, and at hover height; holding torque is within the
actuator limits. Results are cached per robot (the first call on a new robot takes about 4-8 s).

The library router calls it for the active robot on every skill a prompt needs (the `robot` check). A refusal
carries its reason, for example a pick on the WidowX 250: "the fingers open only 25 mm; a 3 cm block needs at
least 38 mm". Reuse and remix only match tiles trained on the same robot, and each tile shows its robot.

## Factory lines (the Line tab)

A company describes its line in words; the planner splits it into stations along one conveyor; the router checks
each station; the agents train every station that needs a skill; then the whole line runs, part by part.

- **Stations.** One arm beside the belt with one tote. A station is either a trained VLA (its own skill, fine-tuned
  from the nearest library skill on that station's own conveyor cell) or a scripted cell (the expert with
  privileged state). The page labels which. A line can mix them, e.g. VLA, VLA, then a scripted end-of-line cell.
- **Planner.** `factory.plan_line`: one OpenAI call with a strict JSON schema (keyword planner offline). Only
  conveyor sort / kitting lines are supported: each station takes one part colour (or, last, every part).
- **Router.** `factory.route_station`: the robot's own feasibility check, an IK reach check over the photo-eye
  stop and the tote in sampled cells, then reuse (exact skill in the library), train (with an estimate from past
  runs), scripted, or refused with the reason.
- **Cells.** `sim/line.py`: the conveyor (MuJoCo maintainers' constant-speed belt construction) and the totes
  (robosuite's bin, scaled), both sourced, see SOURCES.md. The belt stops when a part reaches the station's
  photo-eye; the station moves it into its tote; parts for later stations ride through.
- **Line run.** `factory.run_line`: every zone runs in its own MuJoCo cell; VLA stations are the policy's own
  closed-loop rollouts, batched; after a VLA pick the arm is driven home by a scripted move. The zone schedule
  (one part per zone, parts released every 3 s) gives the line time and throughput. Times are simulated time.

Measured (2026-11-06, simulation, line "Red and Blue Sorting Line": SO-101 VLA red, SO-101 VLA blue, scripted
ViperX 300 end of line):

| Station | Controller | Trained in | Held-out success | On the line (30 parts, seed 11) | Mean cycle |
|---|---|---|---|---|---|
| 1 red to red tote | VLA, fine-tuned from "put the red block in the white bowl" | 91 s | 17/20 (85%) | 9/9 | 8.0 s |
| 2 blue to blue tote | VLA, fine-tuned from station 1's skill | 163 s | 17/20 (85%) | 11/12 | 8.0 s |
| 3 everything else to gray | scripted (ViperX 300) | none | n/a | 9/9 | 10.3 s |

End to end: 29/30 parts in the right tote (97%), 252 s of line time, 6.9 parts per simulated minute. A first run
of 12 parts (seed 7) put 12/12 in the right tote, 6.6 parts per minute. The one miss was a blue part dropped on
the way by station 2.

## What is real and what is scaffolded

Real:
- The SO-101 is the open-source MJCF from TheRobotStudio's SO-ARM100 repository (`vendor/so101`, Apache-2.0,
  generated from the Onshape CAD); the browser draws its actual STL meshes at the poses MuJoCo computed.
- Every rollout on the page is the trained policy's own closed-loop rollout from that round, recorded body pose
  by body pose; success is checked by the task's predicates on the simulator state.
- The agents are real uAgents exchanging real messages; skill agents answer the chat protocol.
- Task parsing and the critic call OpenAI (falls back to a keyword parser / rule critic offline; the log shows
  which one ran).

Scaffolded or limited:
- Simulation only. No real robot was driven.
- The demos come from a scripted expert with privileged state, not from human teleoperation.
- Factory lines: each station is its own MuJoCo cell and the zones are chained by a schedule, not one shared
  simulation of the whole belt; parts are the same 3 cm cubes; the arm's return home after a VLA pick is scripted.
- The per-prompt 3 minute run is a fine-tune of a base policy pretrained ahead of time on the same five
  motion primitives; it does not learn a new motion family from nothing in 3 minutes.
- Agentverse: registration code is in `agents.register_on_agentverse`. It runs only when
  `AGENTVERSE_API_KEY` and a public `VLA_FORGE_PUBLIC_URL` (tunnel to the bureau on port 8102) are set; no
  account or key was created, so the agents are local only right now.
- Meta: `VLA_FORGE_LLM_PROVIDER=meta` routes parsing and the critic to the Llama API (`META_API_KEY`); not used
  in the measured runs.
- Timing: heavy jobs run under `scripts/heavy.sh` at `nice -n 19`. Timed runs set `VLA_FORGE_TIMED=1`, which drops
  `taskpolicy -b`, because background QoS slows MuJoCo 3-4x on this machine.

## Layout

```
src/vla_forge/
  agents.py      uAgents: gateway, task, planner, executor, scene, data, trainer, eval, critic, registry, skill-<id>
  pipeline.py    the work behind each agent, run state, event bus
  compose.py     tier 3: steps, composed scenes, per-step training scenes
  server.py      FastAPI: page, JSON API, SSE, skill calls through the gateway agent
  sim/           MuJoCo scene (SO-101 + blocks, bowls, tape squares), tasks + success predicates, scripted expert
  datagen.py     expert demos in worker processes
  encoders.py    frozen DINOv2 + SigLIP2
  policy.py      the action head
  learn.py       datasets, training loop
  rollout.py     closed-loop evaluation, failure classification
  llm.py         task parser and critic (OpenAI, optional Llama)
  registry.py    skill library, nearest-skill lookup
  factory.py     factory lines: planner, station router, building (training stations), line runs and schedule
  sim/line.py    conveyor and totes for a station cell (sourced: see SOURCES.md)
  viz.py         scene export for three.js
  sim/robots.py  robot builder: part catalog, MJCF assembly, calibration, checks, robot_feasibility
web/             one page: three.js renderer, chat, stage rail, loss, success, ladder, skill library, robot builder (builder.js), factory line (line.js)
scripts/         pretrain_base.py, experiment.py, ladder.py, record_demo.mjs, narrate.py, edit_demo.py, heavy.sh
```
