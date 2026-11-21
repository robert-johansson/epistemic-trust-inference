"""Run memo-mcp's critique_memo (the same function the MCP tool calls) on every memo model file in
the project, on the files' actual contents, and print the verdict, style findings and readback.

Files that import shared definitions from local modules are made self-contained by prepending
those modules (with the local import lines removed).

Usage (from the repository root), with memo-mcp (https://github.com/robert-johansson/memo-mcp) checked out:
    make idioms MEMO_MCP_DIR=/path/to/memo-mcp      (runs with memo-mcp's own Python environment)
or  MEMO_MCP_DIR=/path/to/memo-mcp /path/to/memo-mcp/venv/bin/python tools/check_memo_idioms.py [--readback]
(MEMO_MCP_DIR defaults to ../memo-mcp, a sibling checkout.)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import os  # noqa: E402
sys.path.insert(0, os.environ.get("MEMO_MCP_DIR", str(ROOT.parent / "memo-mcp")))
import server  # noqa: E402  (memo-mcp; its __main__ guard keeps the MCP server from starting)

PORT = ROOT / "intentions_game"
FILES = [
    (ROOT / "advice_taking" / "bart_et_advisor.py", []),
    (PORT / "memo_models.py", []),
    (PORT / "intent.py", [PORT / "memo_models.py"]),
    (PORT / "hier_spec.py", []),
    (PORT / "hier_intent_spec.py", []),
    (ROOT / "development" / "corriveau_trust.py", []),
    (ROOT / "development" / "corriveau_routes.py", [ROOT / "development" / "corriveau_trust.py"]),
]
LOCAL_IMPORT = re.compile(r"^\s*from (memo_models|intent|hier_core|data|corriveau_trust) import .*$", re.M)


def standalone(path, prepend):
    parts = [LOCAL_IMPORT.sub("", p.read_text()) for p in prepend]
    parts.append(LOCAL_IMPORT.sub("", path.read_text()))
    return "\n\n".join(parts)


if __name__ == "__main__":
    show_readback = "--readback" in sys.argv
    for path, prepend in FILES:
        out = server.critique_memo(standalone(path, prepend))
        verdict = out.split("## STYLE")[0].replace("## VERDICT", "").strip()
        style = out.split("## STYLE")[1].split("## READBACK")[0].strip() if "## STYLE" in out else ""
        models = re.findall(r"^Model: (\w+)", out, flags=re.M)
        print(f"\n=== {path.relative_to(ROOT)}  ({len(models)} memo models: {', '.join(models)})")
        print("VERDICT:", verdict.splitlines()[0] if verdict else "?")
        print("STYLE:  ", style.replace("\n", "\n         "))
        if show_readback and "## READBACK" in out:
            print(out.split("## READBACK")[1])
