A policy agent working on the task below sampled {n} candidate next actions from
the same situation. Compare them and pick the ONE most likely to move the task
toward successful completion.

[Task (excerpt)]
{task_goal}

[Situation]
Domain: {domain}
Classified workflow state: {state}
{state_stats}

Ranked next actions from this state (mined from past trajectories — population
statistics, not instructions):
{next_actions}
{tool_docs_block}{examples_block}
[Recent policy-agent activity]
{flow_tail}

[Candidates]
{candidates}

Judge each candidate by: (1) argument correctness — ids, names, dates, filters
and payload fields consistent with the task and with data already retrieved in
the trajectory; (2) progress — does it advance the plan, or repeat/undo work
already done; (3) if a candidate finishes (no tool call), whether EVERY
requirement of the task is already satisfied in the trajectory. Ignore style
and verbosity.

Compare the candidates against each other FIRST, then reply with exactly one
JSON object:
{"compare": "<one line: the material differences you found>", "best": <index of the single best candidate>, "scores": [s0, s1, ...]}
with one 0.0-1.0 score per candidate, in order. Functionally identical
candidates get equal scores; materially different candidates MUST get different
scores.
