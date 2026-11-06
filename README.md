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
rollouts, an inspector on the right, zinc palette, Geist and Instrument Serif. HeyPCB's GitHub repositories
hold hardware (CERN-OHL-S) and an MIT MCP server but no interface code, so nothing was copied; the look was
rebuilt with this project's own HTML, CSS and JavaScript, with no HeyPCB name or logo in the page. Not
affiliated. Details and other credits in `NOTICE`.

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
  viz.py         scene export for three.js
web/             one page: three.js renderer, chat, stage rail, loss, success, ladder, skill library
scripts/         pretrain_base.py, experiment.py, ladder.py, record_demo.mjs, narrate.py, edit_demo.py, heavy.sh
```
