A policy agent attempted the task below {n} times independently (sibling
rollouts of the SAME task). Compare the finished rollouts and pick the ONE
whose final answer is most likely correct and complete.

[Task (excerpt)]
{task_goal}

[Situation]
Domain: {domain}
Final workflow state of rollout 0: {state}
{state_stats}
{tool_docs_block}{examples_block}
[Candidate rollouts]
{candidates}

How to judge — compare, do not verify in isolation:
1. **Grounding**: is the final answer literally supported by data retrieved in
   that rollout's own trajectory (ids, names, counts, dates copied from tool
   results — not invented)?
2. **Task match**: does the answer address EVERY requirement of the task
   (right entity, right filter/time window, right format — a list when a list
   is asked, a refusal only if the task truly requests protected data)?
3. **Trajectory health**: prefer rollouts whose final tool calls succeeded;
   penalize answers produced after repeated failing calls, and empty/hedged
   answers ("None", "no data") when a sibling rollout actually retrieved
   matching records — but prefer a grounded "None" over an invented value.
4. Where two answers disagree on a value, use the trajectories to decide which
   value the data supports. Ignore style, verbosity and step count otherwise.

Compare the candidates against each other FIRST, then reply with exactly one
JSON object:
{"compare": "<one line: the material differences you found>", "best": <index of the single best rollout>, "scores": [s0, s1, ...]}
with one 0.0-1.0 score per candidate, in order. Rollouts with the same final
answer get equal scores; materially different answers MUST get different
scores.
