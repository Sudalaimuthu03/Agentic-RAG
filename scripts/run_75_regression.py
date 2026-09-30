#!/usr/bin/env python3
"""Run the RuntimeFix2 75-turn semantic/runtime regression against the live Flask app.

This script requires the normal project environment, Python 3.11, dependencies,
and a working local Ollama/index. It does not invent expected answers; it compares
observable resolved-request fields against the forensic fixture and writes raw output.
"""
from __future__ import annotations

import json
from pathlib import Path

from app import create_app

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "runtimefix2_75q.json"
OUT = Path(__file__).resolve().parents[1] / "tests" / "artifacts" / "runtimefix2_75q_runtime.json"


def _match(expected: dict, actual: dict) -> list[str]:
    problems=[]
    checks=("entity","document","variant","topic")
    for key in checks:
        wanted=expected.get(key)
        got=actual.get(key)
        if wanted is not None and got != wanted:
            problems.append(f"{key}: expected={wanted!r} actual={got!r}")
    return problems


def main() -> int:
    cases=json.loads(FIXTURE.read_text(encoding="utf-8"))
    app=create_app()
    results=[]
    with app.test_client() as client:
        for case in cases:
            response=client.post("/api/chat", json={"question":case["query"]})
            body=response.get_data(as_text=True).splitlines()
            final={}
            for line in body:
                if not line.strip(): continue
                try:
                    msg=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("done"):
                    final=msg
            actual=final.get("resolved_request", {})
            problems=_match(case, actual)
            results.append({"id":case["id"],"query":case["query"],"http":response.status_code,"problems":problems,"actual":actual,"semantic":final.get("semantic_interpretation")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    failed=[r for r in results if r["http"] != 200 or r["problems"]]
    print(f"Executed: {len(results)}")
    print(f"Resolved-request mismatches: {len(failed)}")
    print(f"Raw report: {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
