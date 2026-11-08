# Dojo

Turning one sentence into a trained robot skill takes models that plan the task, critique the failures and
optimize the next round. Here two models do it side by side, each on what it is built for: Meta's Muse models
(Muse Spark for the task, the line plan and the critic; Muse Wonder for optimizer proposals) write, and Subquadratic's
SubQ Snap reads everything at once (every trajectory, every past round, the whole skill library, up to 384K tokens) and
picks. In each optimizer round both propose a change at the same moment and both are trained and scored in simulation;
the round records whose change won. `docs/meta-subq.md` has the split and the measurements.

Type "have a robot do X". A plain in-process pipeline turns the sentence into a task, builds the scene,
collects expert demonstrations, fine-tunes a vision-language-action policy, measures it on held-out scenes,
reads the failures, retrains if there is time, and saves the result as a new skill tile.
All of it runs locally on one Mac (M5 Pro, 64 GB), inside a 3 minute budget per prompt.

```
uv sync
HF_HUB_OFFLINE=1 uv run vla-forge          # http://127.0.0.1:8100
```

The first start needs the base policy (`models/base.pt`, see "Pretraining" below) and the two Hugging Face
encoders in the local cache (`facebook/dinov2-small`, `google/siglip2-base-patch16-224`). The OpenAI key is
read from the environment or the macOS Keychain (`OPENAI_API_KEY`); it is never written to disk.

## Live Demo

Try the interactive demo at [dojo-psi-two.vercel.app/app](https://dojo-psi-two.vercel.app/app)

🎯 **Prompt** — Enter a task in natural language  
📊 **Pipeline** — Watch real-time loss, model calls, and success metrics  
🤖 **Scenes** — MuJoCo simulation with SO-101 robot  
📈 **Optimizer** — Muse vs SubQ rounds, duel outcomes  
📚 **Library** — Browse trained skills, remix and reuse

## What happens to one prompt

```
browser ── POST /api/prompt ──> task ── router ── scene ── data ── trainer ── eval ── critic ──┐
                                 │       (reuse a tile,  (MuJoCo  (scripted   (fine-tune  (20     │ retrain near
                                 │        train, chain   SO-101,  expert,     the VLA)    held-   │ failures: back
                                 │        or refuse)     layouts) DART noise)             out)    │ to data
                                 │                                                                v
                                 │                                                 registry (saves the tile)
   task: Muse Spark parses the sentence  ║  SubQ reads the whole library and picks the tile that already does it
   critic: Muse Spark writes the plan    ║  SubQ reads every rollout of the run: P(another round helps)
```

Each stage is a function call in `src/vla_forge/agents.py`; the heavy work lives in `src/vla_forge/pipeline.py` and
runs in worker threads. The two model calls at the task and critic stages run concurrently (`lanes.both`). Each stage
streams its logs, demo counts, loss and evaluation to the page over SSE, and every model call streams as an `llm_call`
event (lane, purpose, backend that really answered, tokens, latency) for the Muse + SubQ view.

**The recursion.** Every saved skill is a tile that can be run on a fresh scene from the page. The next prompt starts from the nearest existing skill (SigLIP text
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

## The scaling ladder (the RSI tab)

Prompts come in three tiers, and the page tags each one.

- Tier 1, one motion: reach, push, lift.
- Tier 2, carry a block somewhere: put it in a bowl, stack it on another block.
- Tier 3, composed tasks: several tier 1-2 steps in one scene ("sort the blue and green blocks into the bowls
  of the same color").

A tier 3 prompt goes to the planner instead of the single-skill chain:

```
task ── router ── planner: one saved skill per step that exists
                     │ step with no skill: a child run through the normal one-prompt pipeline
                     v
                  executor: runs every step's own policy back to back in 20 held-out composed scenes,
                     then registry (the composite becomes a skill too)
```

The task parser (Muse Spark) splits the sentence into steps (`compose.py` checks that the scene can hold them). Each
step either reuses a saved skill or is trained on the fly in under 3 minutes, starting from the
nearest skill. A step trained for a composed task collects its demos and held-out checks in that task's own
scenes (`compose.step_layout`): the other bowls and squares are on the table, and half the time the earlier
steps are already done. Without that, the green bowl skill below scored 18/20 on single-skill tables and 5/20
inside the sort task, and the whole task went 0/20; with it, the same prompt went to 10/20.

In the composed run each step's policy runs until its own success check holds, then homes, and the next
step's policy takes over. A rollout counts as a success only if every step's check is true at the end.

## Measured results

Mac mini (M5 Pro, 64 GB), timed from the click to the saved skill, success on 20 held-out scenes.
Runs in order, so later prompts can start from earlier skills (`runs/ladder.jsonl`, `scripts/ladder.py`).

| tier | prompt | started from | rounds | time | success |
|---|---|---|---|---|---|
| 1 | push the green block onto the blue square | base | 9/20, retrain, 15/20 | 164.0 s | 15/20 |
| 1 | pick up the yellow block and hold it up | base | 19/20 | 79.6 s | 19/20 |
| 2 | put the blue block in the blue bowl | base | 15/20, retrain, 17/20 | 165.5 s | 17/20 |
| 2 | put the red block in the white bowl | skill "put the blue block in the blue bowl" | 14/20, retrain, 16/20 | 166.4 s | 16/20 |
| 2 | stack the orange block on the red block | base | 12/20, retrain, 14/20 | 171.6 s | 14/20 |
| 3 | put the red block in the white bowl and push the green block onto the blue square | reused 2 saved skills | step 1 17/20, step 2 11/20 | 26.4 s | 9/20 |
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

The interface is ported from Etch, the author's own earlier project: top bar with breadcrumb and live status,
a toolbar with segmented view tabs, the chat panel on the left, an inspector on the right, white floating cards
over the stage, a dark dock for playback, serif headings with an italic accent word, Figtree and Averia Serif
Libre. Etch's blue accent is swapped for one safety orange. The mascot is an original monochrome robot drawn as
one SVG in `web/mascot.js`. HeyPCB was checked as a source and has no open interface code, so nothing of
theirs is used. Details and other credits in `NOTICE`.

## Custom robots (the robot chip in the prompt)

The robot chip in the prompt window opens the builder inline. It builds the embodiment before a skill is trained, in the spirit of Etch's part picker: pick an arm,
a wrist option and an end effector, see each part as a 3D preview and the assembled robot live, read the checks,
then press "Use this robot"; the next run is trained on it (`/api/robot/use` sets the robot every run copies). Every part is a published open-source MuJoCo model; sources,
licences and the pinned Menagerie commit are in `SOURCES.md`. Code: `src/vla_forge/sim/robots.py`.

- **Robot id.** `<arm>.<wrist>.<effector>`, for example `vx300s.native.native` or `wx250s.native.robotiq_2f85`.
  The stock arm is plain `so101`. Each published skill stores its robot in `Skill.robot` (default `"so101"`), and
  nearest-skill lookup only fine-tunes from skills trained on the same robot.
- **Assembly.** `robot_spec(id)` composes the source MJCF with `MjSpec.attach_body`: the arm's own jaw is removed
  when another effector replaces it, the effector is mounted on the last wrist link, a locked wrist drops its roll
  joint and actuator, and a `tcp` site marks the tool centre. `calibrate(id)` then measures in sim: finger opening
  at the fingertips (mm), gripper command range, a top-down home pose.
- **Checks** (`report(id)`, shown in the prompt window): the model compiles; every arm joint has sane limits with home
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

## Several arms (same chat and canvas)

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

The line in demo v4 ("Blue and Red Kitting Line", typed and trained live in the recording, 2026-11-06):

| Station | Controller | Trained in | Held-out success | On the line (30 parts, seed 7) | Mean cycle |
|---|---|---|---|---|---|
| 1 blue to yellow tote | VLA, fine-tuned from "put the yellow block in the yellow bin" | 93 s, 1 round | 17/20 (85%) | 10/10 | 7.4 s |
| 2 red to green tote | VLA, fine-tuned from "put the green block in the green bin" | 167 s, 2 rounds (15/20 first) | 17/20 (85%) | 8/8 | 7.8 s |
| 3 everything else to gray | scripted (ViperX 300) | none | n/a | 12/12 | 10.3 s |

End to end: 30/30 in the right tote, 249 s of line time, 7.2 parts per simulated minute. From typing the line to
the finished run took 4.9 minutes of wall time.

## What is real and what is scaffolded

Real:
- The SO-101 is the open-source MJCF from TheRobotStudio's SO-ARM100 repository (`vendor/so101`, Apache-2.0,
  generated from the Onshape CAD); the browser draws its actual STL meshes at the poses MuJoCo computed.
- Every rollout on the page is the trained policy's own closed-loop rollout from that round, recorded body pose
  by body pose; success is checked by the task's predicates on the simulator state.
- Task parsing and the critic call Muse Spark, with SubQ answering alongside (falls back to OpenAI, then to a keyword
  parser / rule critic; the log and the Muse + SubQ view show which one ran).

Scaffolded or limited:
- Simulation only. No real robot was driven.
- The demos come from a scripted expert with privileged state, not from human teleoperation.
- Factory lines: each station is its own MuJoCo cell and the zones are chained by a schedule, not one shared
  simulation of the whole belt; parts are the same 3 cm cubes; the arm's return home after a VLA pick is scripted.
- The per-prompt 3 minute run is a fine-tune of a base policy pretrained ahead of time on the same five
  motion primitives; it does not learn a new motion family from nothing in 3 minutes.
- Meta: every Meta call goes through `src/vla_forge/llama.py` to Meta's Model API (`https://api.meta.ai/v1`,
  OpenAI-compatible chat completions, Bearer `META_API_KEY` from the environment or Keychain, `response_format`
  json_schema). Task, line plan and critic use `muse-spark-1.3` (`VLA_FORGE_META_LLM`); optimizer proposals use
  `muse-wonder-1.0` (`VLA_FORGE_META_OPT_LLM`); never a contributor model. Calls run only with
  `VLA_FORGE_META_CREDITS=1` and stop when the spend estimate (`runs/meta-spend.json`, Meta's list prices per model,
  reasoning tokens billed as output) reaches $80. If Meta is off or refuses the key, the call goes to OpenAI behind the
  same interface and the page says Muse is unavailable and names what answered; each optimizer round records the
  backend in `backend` (`meta:<model>`, `openai:gpt-5.5`, `rules` or `stub`).
- SubQ: `src/vla_forge/subq.py` calls SubQ Snap 1.1 through the Decisions API (`/v1/systemone`; the Snap models do not
  take chat completions). It answers `choice` and `noul` questions over a state of up to 384K tokens. Calls stop at
  80M tokens of the 100M free grant (`runs/subq-spend.json`).
- Optimizer: every round both models are asked at once. Muse Wonder gets per-failure evidence (tag, start position,
  contact stats), rendered frames and the full knob menu; SubQ gets every trajectory of every past round and every past
  change with its score, and picks from the same menu. Neither may repeat a change that was just reverted. Both
  changes are trained from the same parent with the same budget and scored on the same SELECT scenes; the better one
  faces the parent (`duel` in each round's record). Scenes come from three
  disjoint seed sets: TRAIN (demonstrations), SELECT (100 scenes, keep or revert each round) and TEST (100 scenes, never
  seen by the optimizer, scored only at R0 and on the final policy; Wilson CIs, McNemar exact and paired bootstrap on
  the same scenes). A control arm runs the same number of rounds with a random valid change (same keep rule) so the
  gain is attributed to the optimizer and not to extra training. Near-failure demonstrations are placed around SELECT
  failure positions (jittered 1.5 cm), so SELECT scores are optimistic for that knob; TEST is the number to read.
- Blender montage: physics is MuJoCo. Blender (`scripts/blender_worker.py`, `src/vla_forge/blender_montage.py`) only
  renders the recorded body poses: the compiled SO-101 scene (its STL meshes, black and white arm, props) is exported
  once per layout structure, split over N headless `blender -b` processes (EEVEE, 2x tiles averaged down), one render
  pass per rollout and moment, and packed into the sprite sheets the Optimize panel plays. The control arm is scored
  without Blender. A scene is built once per layout structure per worker and re-posed per rollout.
- Timing: heavy jobs run under `scripts/heavy.sh` at `nice -n 19`. Timed runs set `VLA_FORGE_TIMED=1`, which drops
  `taskpolicy -b`, because background QoS slows MuJoCo 3-4x on this machine.

## Layout

```
src/vla_forge/
  agents.py      the pipeline: task, router, scene, data, trainer, eval, critic, registry, planner, executor
  lanes.py       the Meta and SubQ lanes: concurrent calls, llm_call events, totals, optimizer duels
  llama.py       Meta's Model API client (Muse Spark, Muse Wonder), OpenAI and stub fallbacks, the $80 guard
  subq.py        SubQ Snap through the Decisions API, the 80M-token guard
  pipeline.py    the work behind each stage, run state, event bus
  compose.py     tier 3: steps, composed scenes, per-step training scenes
  server.py      FastAPI: page, JSON API, SSE
  sim/           MuJoCo scene (SO-101 + blocks, bowls, tape squares), tasks + success predicates, scripted expert
  datagen.py     expert demos in worker processes
  encoders.py    frozen DINOv2 + SigLIP2
  policy.py      the action head
  learn.py       datasets, training loop
  rollout.py     closed-loop evaluation, failure classification
  llm.py         task parser and critic prompts and schemas
  registry.py    skill library, nearest-skill lookup
  factory.py     factory lines: planner, station router, building (training stations), line runs and schedule
  sim/line.py    conveyor and totes for a station cell (sourced: see SOURCES.md)
  viz.py         scene export for three.js
  sim/robots.py  robot builder: part catalog, MJCF assembly, calibration, checks, robot_feasibility
web/             one page: three.js renderer, docked prompt bar and conversation, stage rail, loss, success, RSI, skill library, robot builder (builder.js, inline in the prompt window), build plan (plan.js, inline when a reference is attached), multi-arm cells (arms.js)
scripts/         pretrain_base.py, experiment.py, ladder.py, record_demo.mjs, narrate.py, edit_demo.py, heavy.sh
```
