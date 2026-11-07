# Dojo on Meta Muse and Subquadratic

Fetch.ai is gone. Two models do the thinking, side by side, each on the job it is built for. Everything here runs
in simulation.

## What each model is (measured 2026-11-07)

***Muse Spark 1.3 and Muse Wonder 1.0** (Meta's Model API, `https://api.meta.ai/v1`, OpenAI-compatible chat
completions). They write: free text, a diagnosis, a reason, a structured plan. Replies are constrained by
`response_format` json_schema. Both are reasoning models (`reasoning_effort` low; "none" is refused), so reasoning tokens
count in `completion_tokens` and are billed as output; max_tokens is 6000. Measured on muse-spark-1.3 with a schema: 4.8 s
and 621 completion tokens (603 reasoning) at low effort, 1.6 s at minimal. Rate limit 3000 requests and 4M tokens per
minute.
- Muse Spark (`muse-spark-1.3`, `VLA_FORGE_META_LLM`): task parse, line plan, critic. $1.25 in, $0.15 cached, $4.25
  out per 1M tokens.
- Muse Wonder (`muse-wonder-1.0`, `VLA_FORGE_META_OPT_LLM`): the optimizer proposer that goes head to head with SubQ.
  $4.00 in, $0.20 cached, $20.00 out per 1M tokens. Unknown models are charged at Wonder prices. No contributor models.
- Guard: calls are off until `VLA_FORGE_META_CREDITS=1` and stop once the running estimate in `runs/meta-spend.json`
  reaches $80. The key (`META_API_KEY`, environment or Keychain) is read once per process, so a new key is picked up on
  restart.
- Honest fallback: if Meta refuses the key or the credit is off, the Meta lane says "Muse Spark unavailable" with the
  reason, and when OpenAI answers in its place the label is "OpenAI gpt-5.5 (Meta unavailable)". Each optimizer round
  records the backend that actually answered (`meta:muse-wonder-1.0`, `openai:gpt-5.5`, `rules`, `stub`).

*SubQ Snap 1.1** (Subquadratic, `https://api.subq.ai/v1`). Researched on docs.subq.ai and checked against the API with
our key:
- It is a *decision* model, not a chat model. `/v1/chat/completions` answers 400 "does not support completions" for
  subq-snap-1.1, subq-snap-latest and subq-snap-preview; the chat model (subq-preview) is early access and our key gets
  403 model_not_allowed. The model is called through the Decisions API, `POST /v1/systemone`: a `state` (any JSON or
  text) plus up to 128 named questions of three types: `choice` (2 to 255 named options, returns the pick plus a
  probability for every option), `noul` (yes/no probability) and `score` (an ordered scale).
- Context window 393,216 tokens (384K). Input $0.04 per 1M tokens, output free (always 0 tokens).
- Speed: a 269-token request answered in 0.34 s; a 189,392-token state (30 rounds x 100 scenes of trajectory stats)
  answered in 2.43 s. Long context is its strength: it reads everything at almost no extra latency.
- Free grant 100M tokens. Guard: `runs/subq-spend.json` counts the `input_tokens` the API reports; calls stop at 80M.
  Each call's state is trimmed to fit 300K tokens.

## The split

Muse writes; SubQ decides over everything. Neither gets the other's answer, and both are asked at the same moment
on threads, so the two lanes are visibly busy together.

| Moment | Muse (writes, short context) | SubQ Snap (decides, whole history) |
| --- | --- | --- |
| Task parse | Muse Spark: sentence to task spec and success check | reads the whole skill library and picks which saved skill, if any, already does this sentence (`choice`, with `none`) |
| Line plan | Muse Spark: splits a line description into stations | (not used) |
| Critic, each training round | Muse Spark: the next round's plan and a plain rationale | reads every rollout of every round of this run and gives P(another round raises success) (`noul`) |
| Optimizer round | Muse Wonder: proposes one change from the menu, with diagnosis, reason and scene notes, from the failure report and the Blender picture | reads every trajectory of every past round, every past change and its score, and picks one change from the same menu (`choice`, every option with a probability) |

In the optimizer both proposals are real changes from the same validated menu (reverted changes are barred for both).
Each is applied to the same parent, trained with the same budget and scored on the same 100 SELECT scenes, one after
the other. The round records both changes, both SELECT scores and the winner (`duel`: `meta`, `subq` or `tie`;
when both pick the identical change it is trained once and recorded as `agreed`). The winner's child then faces the
parent exactly as before: admitted only if it beats the parent on SELECT. TEST stays unseen by both. A round takes about
twice as long because two children train.

Where SubQ answers in the pipeline, the existing rule-based code still makes the decision (the router decides reuse,
the critic plan decides retrain), and SubQ's answer is shown next to it with whether they agree. Only in the optimizer
do SubQ's choices change what trains, because there both are scored.

## Orchestration

`src/vla_forge/agents.py` is now a plain in-process pipeline (asyncio, heavy stages on threads): task, router, scene,
data, trainer, eval, critic, registry, plus planner and executor for composed tasks. It emits the same events the page
already uses (`log`, `thought`, `task`, `route`, `scene`, `eval`, `critic`, `published`, `reused`, `rejected`, `plan`,
`child`, `offer`, `error`). `agent_msg` is gone with uAgents. New: `llm_call` (phase start or end, lane `meta` or
`subq`, what the call is for, backend that answered, tokens, latency in ms, error) and `duel` (one per optimizer
round: both changes, both scores, the winner).

## The Muse Spark + SubQ view (web/agents.js)

Two lanes side by side. Each lane: the calls in flight (spinner, what they are for, elapsed time), the finished calls
(backend, tokens, latency), totals for the session, and the guard (Meta $ spent of $80, SubQ tokens of 80M). Under the
lanes, the duel table: one row per optimizer round, both changes, both SELECT scores, the winner, and a running tally.
Every number comes from `llm_call` and `duel` events or `/api/llm`; nothing is drawn before it happens.

## Removed

uagents and uagents-core, the Bureau and its agents, Agentverse mailbox registration, the ASI:One chat path, the
forge-optimize protocol, the message bus (agentbus.py), `/api/agents/messages` and `/api/agents/manifests`,
tests/test_agentverse.py and tests/test_agentbus.py, docs/fetch-ai-plan.md.

## Not used

sam-3.1 on sim frames was considered for a frame check and left out: the sim already knows every contact and object
pose exactly, so a segmentation model on its renders would only restate ground truth at $2.50 per 1000 images.

## Measured (2026-11-07, live keys)

- Task parse, "insert the green block into the socket": both lanes started at the same instant. Muse Spark
  (`meta:muse-spark-1.3`) 1,241 tokens in, 606 out, 4.1 s; SubQ Snap read all 50 saved skills (8,316 tokens) in 0.69 s.
- One optimizer proposal step on saved run opt-091548-bba5 (3 rounds, 249 KB of rounds, scenes and trajectories): both
  started at the same instant. SubQ (`subq:subq-snap-1.1`) read 231,438 tokens in 2.8 s and picked "add 30
  demonstrations at the failed starts" (p 0.41). Muse Wonder (`meta:muse-wonder-1.0`) 2,012 in, 1,293 out, 14.5 s,
  proposed "learning rate 5e-05". Different changes, so both children would train. A full round (two children trained
  and scored) did not fit the 8 GB test-server cap next to the live server, so the training half was not run here.
