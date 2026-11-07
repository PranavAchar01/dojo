# Optimizer framework

The first optimizer was a hand-written loop: a model proposes one change, retrain, score on SELECT, keep or revert, one
chain of states. It could not keep a population, could not resume, and ended after a fixed number of rounds. The ask was
to run on an established open-source framework that keeps improving round after round.

## What the loop needs

An LLM proposes a change, a policy trains, SELECT scores it, the best policies are kept as a population, and the whole
thing runs until stopped with checkpoints and resume. It must run on this Mac with no cloud account, and it must keep the
fair split (TEST never used to choose, plus a control arm).

## Candidates

| Candidate | What it gives | Fit here | Verdict |
|---|---|---|---|
| OpenEvolve (`ProgramDatabase`) | Open AlphaEvolve-style system. Islands, MAP-Elites grid, elite archive, parent and inspiration sampling, migration, `save`/`load` checkpoints. Pure Python, pip install, Apache licence. Checked by installing 0.4.1 and driving the database with fake programs: sampling, island handling and save/load all work offline. | A program is text; the fine-tuning recipe (settings plus the plan of extra demonstrations) is a small JSON text. The LLM operator and the evaluator are ours to supply. | Chosen |
| ShinkaEvolve (Sakana) | Same family: LLM mutation, islands, archive, SQLite store, novelty checks. | Built around evolving source files with an `evaluate.py` and EVOLVE-BLOCK markers, and its own LLM client stack; we would be wrapping a training run in a code-patch interface. Read from its README only, not run. | Runner-up |
| Ray Tune PBT / PB2 | Population Based Training. Exploit copies weights of a better trial, explore perturbs numeric hyperparameters. Needs checkpoint save and load in the trainable. | Right shape for "population of policies", but explore perturbs numbers, and our change menu includes data (extra demonstrations at failure starts) and a categorical scope. `custom_explore_fn` could call the LLM, but it adds a Ray runtime and a trainable wrapper for little gain. | Not chosen |
| Optuna (ask and tell, SQLite storage) | Mature, resumable studies, samplers. Installed 5.0.0 to check. | A study is a bag of independent trials. Our trials are children of a parent with inherited weights and data, which Optuna does not model. | Not chosen |
| Nevergrad | Derivative-free optimizers. | Same mismatch: no lineage, no LLM operator. | Not chosen |
| DSPy optimizers | Optimize prompts and programs of LLM pipelines. | Optimizes the model's prompts, not a robot policy's training. | Not applicable |
| DAgger-style aggregation | Add data where the policy fails. | A strategy, not a loop framework. It is already one of our change types (extra demonstrations at failed starts). | Kept as a knob |

## Choice and why

OpenEvolve's database. The part that has to be reliable in perpetuity is the population bookkeeping and its checkpoints,
and that is the part OpenEvolve provides and tests. Its other half (prompt building, diff patches to source code, its
evaluator pool) is for evolving programs and is not used: the proposal operator stays our schema-constrained model call,
which sees the failure report, the rendered failure frames, the population and what was already tried under the parent.

How it maps:

- Program = one trained policy's recipe (`code` is JSON). Metrics: SELECT success, extra demonstrations, training seconds.
- MAP-Elites grid over (extra demonstrations, training seconds), 2 islands, both started from the untouched skill.
- A child is admitted only if SELECT success beats its parent's. Reverted children stay in the run record and bar a repeat
  of the same change under the same parent.
- Weights of each admitted program are saved (`weights/`), and data is rebuilt from the recipe plan on resume, so a run
  continues after a crash or a stop from `runs/opt-<id>/`.
- Control arm: same number of trained candidates per round, a random valid change on its own best, same keep rule.

## Fair split in perpetual mode

- TRAIN seeds make demonstrations, SELECT decides, TEST decides nothing. The seed sets are checked disjoint for up to 400 rounds.
- TEST is scored on the current best of each arm at R0, every 4 rounds, on request (`POST /runs/<id>/test`) and at the end.
  Every look is counted (`test.peeks`, `test.peek_count`) and the page shows the count. The reported number is the latest look, not the highest.
  Repeated looks give more chances for a lucky reading; the run record says so.
- Reported with Wilson intervals, paired bootstrap and exact McNemar, as before.

## Running

    scripts/heavy.sh .venv/bin/python scripts/run_optimizer.py --skill <id> --perpetual --max-minutes 45
    touch runs/opt-<id>/STOP           # end after the round (or Ctrl-C)
    scripts/heavy.sh .venv/bin/python scripts/run_optimizer.py --resume <id> --perpetual

`optimizer.run(skill, n_rounds, opt, replay, compute, emit, train_seconds, control)` keeps its signature; `n_rounds <= 0`
means perpetual. Old run JSON is read unchanged by the Optimize overlay; new fields (`mode`, `engine`, `parent`, `program`,
`test.peeks`) are additions.
