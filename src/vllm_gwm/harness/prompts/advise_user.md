[Task (excerpt)]
{task_goal}

[Current situation]
Benchmark domain: {domain}
Classified workflow state: {state} (cosine distance {cos_dist}, confidence {confidence})
{trap_line}
{retry_line}

[Graph context — descriptive only, NOT predictions for this agent]
These are aggregate statistics from past trajectories. They describe which actions
were common/associated-with-success historically; they do NOT tell you this
agent's odds. Never quote a success rate to the agent as if it were causal.
{state_stats}

Actions taken from this state in past trajectories (for grounding what to suggest):
{next_actions}
{tool_docs_block}
[Recent policy-agent activity (tail of the live trajectory)]
{flow_tail}

Decide: call a tool if you need grounding, then reply with your final
{"advice": "..."} (empty string if the policy agent should just continue).
