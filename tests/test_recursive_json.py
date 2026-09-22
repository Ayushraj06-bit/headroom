"""Unit tests for headroom.transforms.recursive_json — the structural (embedded)
JSON routing step. Uses a fake dispatch so the mechanism is tested in isolation
from the real compressors."""

from __future__ import annotations

import json

from headroom.transforms.recursive_json import (
    carries_record_array,
    json_document_spans,
    route_embedded_json,
)


def _upper_dispatch(span: str) -> str | None:
    """Fake compressor: returns a shorter deterministic stand-in for any span."""
    try:
        v = json.loads(span)
    except ValueError:
        return None
    return f"<TABLE n={len(v)}>" if isinstance(v, list) else "<OBJ>"


def test_embedded_json_routed_and_surroundings_exact() -> None:
    payload = json.dumps([{"id": i, "ok": True} for i in range(6)], separators=(",", ":"))
    content = f"Fetched rows from API:\n{payload}\nDone (200 OK)."
    out = route_embedded_json(content, _upper_dispatch)
    assert out is not None
    assert out.startswith("Fetched rows from API:\n")
    assert out.endswith("\nDone (200 OK).")
    assert "<TABLE n=6>" in out


def test_ccr_marker_span_passed_through() -> None:
    # A span already carrying a CCR marker must never be re-routed (R1).
    content = 'prefix [{"a":1,"b":2},{"a":3,"b":"<<ccr:deadbeef,json,900>>"}] suffix'
    out = route_embedded_json(content, _upper_dispatch)
    assert out is None  # only span contains a marker → skipped → nothing to do


def test_no_json_is_noop() -> None:
    assert route_embedded_json("just prose, nothing structured here", _upper_dispatch) is None


def test_whole_block_json_is_callers_job() -> None:
    # A block that IS a single JSON value is skipped (routed by the caller).
    content = json.dumps([{"a": i} for i in range(5)], separators=(",", ":"))
    assert route_embedded_json(content, _upper_dispatch) is None


def test_benefit_gate_declines_when_not_smaller() -> None:
    payload = json.dumps([{"a": i} for i in range(5)], separators=(",", ":"))
    content = f"x {payload} y"
    # Dispatch that returns something LARGER → must be declined (outcome gate).
    assert route_embedded_json(content, lambda s: s + " " * 999) is None


def test_deterministic() -> None:
    payload = json.dumps([{"k": i} for i in range(8)], separators=(",", ":"))
    content = f"a {payload} b {payload} c"
    r1 = route_embedded_json(content, _upper_dispatch)
    r2 = route_embedded_json(content, _upper_dispatch)
    assert r1 == r2 and r1 is not None
    assert r1.count("<TABLE n=8>") == 2  # both embedded spans routed


def test_scalar_array_not_routed() -> None:
    # array of scalars is not a "routable" JSON shape (no dict rows)
    content = "nums: [1,2,3,4,5,6,7,8] done"
    assert route_embedded_json(content, _upper_dispatch) is None


def test_json_document_spans_finds_containers_anywhere() -> None:
    doc = json.dumps({"domains": [{"name": "a"}, {"name": "b"}]})
    arr = json.dumps([1, 2, 3])
    text = "Tool result: " + doc + " and a list " + arr + " done"
    assert [text[a:b] for a, b in json_document_spans(text)] == [doc, arr]


def test_json_document_spans_ignores_scalars_prose_and_unbalanced_json() -> None:
    assert json_document_spans('"just a quoted sentence"') == []
    assert json_document_spans("42") == []
    assert json_document_spans("prose with a [note] and {braces} but no JSON") == []
    assert json_document_spans('{"truncated": [1, 2, 3') == []
    assert json_document_spans("{{HEADROOM_TAG_0}}") == []
    assert json_document_spans("") == []


def test_json_document_spans_whole_document() -> None:
    doc = json.dumps({"a": [1, 2]})
    assert json_document_spans(doc) == [(0, len(doc))]
    assert json_document_spans("  " + doc + "\n") == [(2, 2 + len(doc))]


def test_carries_record_array_separates_the_two_shapes() -> None:
    assert carries_record_array(json.dumps({"domains": [{"name": "a"}, {"name": "b"}]}))
    assert carries_record_array(json.dumps([{"id": 1}, {"id": 2}]))
    # A lone object, or an array of scalars, has no record delimiter whose
    # deletion leaves a valid-but-shorter document.
    assert not carries_record_array(json.dumps({"file": "src/mod.py", "line": 1}))
    assert not carries_record_array(json.dumps([1, 2, 3]))
    assert not carries_record_array(json.dumps([{"id": 1}]))
