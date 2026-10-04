"""Schemas must mirror tools.py signatures (minus server-injected params)."""

from __future__ import annotations

import inspect
import json

import pytest

from app import tools
from app.tool_schemas import SERVER_INJECTED_PARAMS, TOOL_NAMES, TOOL_SCHEMAS

EXPECTED_TOOLS = {"list_services", "check_slots", "create_booking", "get_bookings", "cancel_booking"}


def test_exactly_the_five_tools_are_exposed():
    assert TOOL_NAMES == EXPECTED_TOOLS
    assert len(TOOL_SCHEMAS) == 5


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["function"]["name"])
def test_schema_matches_python_signature(schema):
    fn = schema["function"]
    sig = inspect.signature(getattr(tools, fn["name"]))
    model_params = {n: p for n, p in sig.parameters.items() if n not in SERVER_INJECTED_PARAMS}

    props = fn["parameters"]["properties"]
    assert list(props) == list(model_params), "parameter names/order differ"
    assert set(fn["parameters"]["required"]) == {
        n for n, p in model_params.items() if p.default is inspect.Parameter.empty
    }
    for name, param in model_params.items():
        assert param.annotation in (str, "str")
        assert props[name]["type"] == "string"
        assert props[name]["description"].strip()

    # Any server-injected param must be optional on the Python side.
    for name in SERVER_INJECTED_PARAMS & set(sig.parameters):
        assert sig.parameters[name].default is None


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["function"]["name"])
def test_schema_shape_is_openai_function_format(schema):
    assert schema["type"] == "function"
    fn = schema["function"]
    assert set(fn) == {"name", "description", "parameters"}
    assert fn["parameters"]["type"] == "object"
    assert len(fn["description"]) > 40
    json.dumps(schema)  # serializable


def test_session_id_never_exposed_to_model():
    for schema in TOOL_SCHEMAS:
        assert "session_id" not in schema["function"]["parameters"]["properties"]


def test_discovery_tools_reinforce_source_of_truth():
    by_name = {s["function"]["name"]: s["function"]["description"] for s in TOOL_SCHEMAS}
    for name in ("list_services", "check_slots"):
        assert "ONLY source of truth" in by_name[name]
    assert "explicitly confirmed" in by_name["create_booking"]
