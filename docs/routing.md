# Arm routing

The engine picks the arm. `sim/routing.py` asks `feasibility()` the same question for every available arm
(each arm with its native wrist and jaw) and chooses from the measured answers: fit, jaw width, payload at the
carry pose and reach. Nothing matches on object names.

Rules: the arm that does the whole task wins; the default SO-101 goes first when it can, otherwise the arm with
the smallest total joint torque budget (`cost_nm`). If no single arm can, the objects are split across arms and
each share is re-checked on its arm alone. An object no arm can do is excluded with the engine's reasons from
the arm that came closest. A run the user has not pinned a robot for moves to `chosen[0]`.

## Where it appears

- `spec.routing` (also `task.spec.routing` in the `task` event, and in the published skill's task record).
- `route.routing` on the `route` event and on `Run.route` (the router step card). Same object.
- A `route.checks` entry named `arm routing`: ok/detail, with the pinned-robot message when the user pinned.

## Fields (all JSON, `{}` when the task has no library objects)

- `chosen`: list of robot ids that do the task (one arm, or several when split).
- `assignments`: list of `{robot, objects: [..], reason}`.
- `excluded`: list of `{name, closest_robot, reasons: [..]}` for objects no arm can do.
- `ranked`: list, best first, of `{robot, label, cost_nm, score (0..1 share of objects passed), passes: [names], fails: [{object, reasons: [..]}]}`.
- `pinned`: `null`, or `{robot, ok, message}` when the user pinned a robot. If `ok` is false the message names
  what it cannot do and the arm that could.
- A `task` event carries it as `spec.routing`; the prompt bar's robot chip shows `chosen` (labels from `ranked`).
