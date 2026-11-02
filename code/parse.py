"""Turn a model reply into tool calls.

Order matters. AgentTune's parser returns None for both "[]" and garbage, so the strict
forms run first; it only sees what they reject, and anything it rescues is not strict.
"""
import ast
import importlib.util
import json
import re
from pathlib import Path


def _load_agenttune_extract():
    # Load the one stdlib file directly: importing the package would pull in torch.
    root = Path(importlib.util.find_spec("agenttune").origin).parent
    path = root / "agentic" / "rollout_engines" / "tool_call_parse.py"
    spec = importlib.util.spec_from_file_location("agenttune_tool_call_parse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_tool_calls_from_text


_extract = _load_agenttune_extract()


def _strict_json(text):
    """The training format exactly: a list of {"tool_name", "parameters"} objects."""
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    if not isinstance(obj, list):
        return None
    calls = []
    for c in obj:
        if not (isinstance(c, dict) and set(c) == {"tool_name", "parameters"}
                and isinstance(c["tool_name"], str) and isinstance(c["parameters"], dict)):
            return None
        calls.append({"name": c["tool_name"], "arguments": c["parameters"]})
    return calls


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _pythonic(text):
    """BFCL's [f(a=1), g()] and the paper's one-shot Action: f(a=1)."""
    try:
        body = ast.parse(re.sub(r"^\s*Action\s*:\s*", "", text), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    calls = []
    for node in body.elts if isinstance(body, ast.List) else [body]:
        name = _dotted(node.func) if isinstance(node, ast.Call) else None
        if not name or node.args or any(k.arg is None for k in node.keywords):
            return None
        try:
            args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        except (ValueError, TypeError):
            return None
        calls.append({"name": name, "arguments": args})
    return calls or None


def _lenient(text):
    found = _extract(text)
    if not found:
        return None
    calls = []
    for c in found:
        try:
            args = json.loads(c["function"]["arguments"])
        except ValueError:
            return None
        if not isinstance(args, dict):
            return None
        calls.append({"name": c["function"]["name"], "arguments": args})
    return calls


def parse(text):
    """{"calls": [{"name", "arguments"}], "parsed": bool, "strict": bool}."""
    text = text.strip()
    for form in (_strict_json, _pythonic):
        calls = form(text)
        if calls is not None:
            return {"calls": calls, "parsed": True, "strict": True}
    calls = _lenient(text)
    if calls is not None:
        return {"calls": calls, "parsed": True, "strict": False}
    return {"calls": [], "parsed": False, "strict": False}
