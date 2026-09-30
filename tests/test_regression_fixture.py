import json
from pathlib import Path


def test_runtimefix2_75q_fixture_is_complete():
    data=json.loads(Path("tests/fixtures/runtimefix2_75q.json").read_text(encoding="utf-8"))
    assert len(data)==75
    assert [x["id"] for x in data]==list(range(1,76))
    assert sum(x["first_failure"]=="PASS" for x in data)==23


def test_forensic_failure_classes_are_present():
    data=json.loads(Path("tests/fixtures/runtimefix2_75q.json").read_text(encoding="utf-8"))
    classes={x["first_failure"] for x in data}
    assert {"REFERENCE","INTENT","DOCUMENT","ENTITY","EXECUTION","TOPIC","MULTI_TARGET","ANSWER","CONTEXT"}.issubset(classes)
