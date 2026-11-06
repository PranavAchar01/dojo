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
page ("Call agent") or by any other agent. The next prompt starts from the nearest existing skill (SigLIP text
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

## Measured results

RESULTS_TABLE

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
  agents.py      uAgents: gateway, task, scene, data, trainer, eval, critic, registry, skill-<id>
  pipeline.py    the work behind each agent, run state, event bus
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
web/             one page: three.js renderer, pipeline view, loss, success, skill library
scripts/         pretrain_base.py, experiment.py, record_demo.mjs, heavy.sh
```
