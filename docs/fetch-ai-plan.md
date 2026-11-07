# Fetch.ai plan: skills and the optimizer as agents other agents can hire

Status: a plan. Nothing here has been registered, and no Agentverse account or key has been used.

## What exists

The Forge already runs a team of uAgents in one Bureau (gateway, task, planner, executor, scene, data, trainer,
eval, critic, registry). Every published skill gets its own agent that answers the uAgents chat protocol.
`agents.register_on_agentverse` can register the task agent and the skill agents, and only runs when an
`AGENTVERSE_API_KEY` and a public endpoint are set. The optimizer (`optimizer.py`) is a plain function with a
JSON record of every round.

## Proposal

1. **Optimizer agent.** Wrap `optimizer.run` as one more agent, `optimizer`, speaking the chat protocol plus a
   small typed protocol of its own:
   - `OptimizeRequest { skill_id | checkpoint_url, rounds, train_seconds }`
   - `OptimizeProgress { round, change, kept, success, n, ci }` (one message per round)
   - `OptimizeResult { run_id, start_success, final_success, ci, checkpoint_url }`
   The agent describes itself as "improves a simulated manipulation policy, one change per round, scored on 100
   held-out scenes". Another agent can hire it by address, watch the progress messages and receive the better
   checkpoint, or a note that nothing was kept.
2. **Skill agents as published services.** Register each skill agent with a description that states the
   instruction, the robot and its measured success in simulation. The registry agent answers "which skill does
   this sentence need" from the library, so a planning agent elsewhere on Agentverse can ask instead of training.
3. **Pay per job, later.** Fetch's payment protocol would let the optimizer agent quote a price per run
   (compute seconds). Out of scope until there is an account and a decision to take money; start free.

## Steps, in order

1. Local only: put the optimizer agent in the existing Bureau and call it from the gateway over the chat protocol
   (no Agentverse). Test that a second agent can request a run and receive progress and the result.
2. Add the typed protocol and publish its manifest next to the skill agents' chat protocol.
3. With his key and a public URL (tunnel to the Bureau port), call `register_on_agentverse` for `optimizer` and the
   skill agents; check discovery from a fresh agent that knows only the name.
4. Add the payment protocol only if he wants to charge for runs.

## Limits to state plainly

- One run uses the machine's single training slot for about 15 minutes; the agent must refuse or queue a second
  request, not run two.
- Results are simulation results on one skill and one robot. The agent's description says so.
- A hired optimizer improves a checkpoint it is given; it does not see the other agent's data.
