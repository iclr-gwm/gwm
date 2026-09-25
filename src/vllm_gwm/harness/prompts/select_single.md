Rate this rollout: a policy agent finished (or exhausted its budget on) a task.
Using the trajectory tail and the graph context, estimate the probability (0.0-1.0)
that its final answer is correct/complete. Penalize: unresolved errors, repeated
failing calls, answers not grounded in retrieved data, empty/hedged answers.

[Task (excerpt)]
{task_goal}

[Situation]
Domain: {domain}
Final classified workflow state: {state}
{state_stats}
{tool_docs_block}{examples_block}
[Trajectory tail]
{flow_tail}

Reply with exactly one JSON object: {"score": <0.0-1.0>, "reason": "<one line>"}.
