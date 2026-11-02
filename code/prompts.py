"""Every prompt the project uses. BFCL_ZERO_SHOT and CUSTOM_ONE_SHOT are Figures 5 and 6 of
the MASSIVE-Agents paper. FORMAT is AgentTune's default fallback wording, minus the {tools}
header and plus the no-call sentence.
"""
import json

FORMAT = (
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. "
    "If no tool fits the request, reply with only []. "
    'A user turn that starts with "Tool result" is the output of your call, not a new '
    "message from the user: use it to answer the user's original question as plain text."
)

POLICY = {
    "verbatim": "Argument values: copy them exactly as written in the user's request.",
    "english": "Argument values: write them in English.",
}

BFCL_ZERO_SHOT = """You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\\n
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.

{functions} User: {question} Bot:"""

CUSTOM_ONE_SHOT = """A chat between a curious User and an artificial intelligence Bot. The Bot gives helpful, detailed, and polite answers to the User's questions. In this session, the model has access to external functionalities.

The following actions are available:
{functions}

Model Instructions:
- Function calls must adhere to the programming language's syntax.
- Select the most suitable action from the given list.
- Extract relevant arguments for function calls from the user's utterance.
- Do not ask any clarifying questions.
- Preserve the original argument values as provided in the user's utterance.
- Always respond with a function call (e.g., Action: ActionOne()). Do not add further explanations or comments.

Example Interaction:
User: set an alarm for nine am this week
Bot: Action: alarm.set(time="nine am", date="this week")

User: {question} Bot:"""


def tools_block(item):
    if not item.get("compact"):
        return json.dumps(item["tools"], indent=2, ensure_ascii=False)
    lines = []
    for tool in item["tools"]:
        args = list((tool.get("parameters") or {}).get("properties") or {})
        lines.append(json.dumps({"name": tool["name"], "description": tool["description"], "args": args},
                                ensure_ascii=False))
    return "\n".join(lines)


def _head(item):
    head = f"You can call these tools:\n{tools_block(item)}"
    policy = item.get("policy")
    return f"{head}\n\n{POLICY[policy]}" if policy else head


def grpo_prompt(item):
    """Per-item tools go in the system message; AgentTune appends tools_fallback_prompt=FORMAT."""
    return [{"role": "system", "content": _head(item)}, {"role": "user", "content": item["query"]}]


def training_messages(item):
    """What the model sees in SFT and eval: must equal grpo_prompt after AgentTune's append."""
    system, user = grpo_prompt(item)
    return [{"role": "system", "content": f"{system['content']}\n\n{FORMAT.format(tools='')}"}, user]


def target(item):
    """The reply SFT teaches: what FORMAT asks for, using each argument's first accepted value."""
    calls = [{"tool_name": g["name"], "parameters": {k: v[0] for k, v in g["arguments"].items()}}
             for g in item["gold"]]
    return json.dumps(calls, ensure_ascii=False)


def bfcl_zero_shot(item):
    text = BFCL_ZERO_SHOT.format(functions=tools_block(item), question=item["query"])
    return [{"role": "user", "content": text}]


def custom_one_shot(item):
    text = CUSTOM_ONE_SHOT.format(functions=tools_block(item), question=item["query"])
    return [{"role": "user", "content": text}]
