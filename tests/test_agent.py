"""
test_agent.py — unit tests for the tool-use loop in src/agent.py.

WHY these tests mock the Anthropic client instead of calling the real API:
agent.py's loop-control logic (append messages correctly, stop when
stop_reason != "tool_use", cap at MAX_TOOL_ITERATIONS) is deterministic
code we can and should verify without spending real API calls or needing
network access / an API key in CI. The two real, live-API test questions
in run_demo() remain the manual/integration check that the *model's*
behavior (citing sources, refusing out-of-scope questions) is sound —
that's a judgment call only a live model can make, so it isn't something
a mock should assert on.
"""

import unittest.mock as mock
from types import SimpleNamespace
from unittest.mock import MagicMock

import agent


def _text_response(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
    )


def _tool_use_response(tool_use_id: str, query: str):
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id=tool_use_id,
                name="search_sop_documents",
                input={"query": query},
            )
        ],
        stop_reason="tool_use",
    )


def test_format_search_results_handles_empty_list():
    assert agent._format_search_results([]) == "No results found."


def test_format_search_results_includes_source_and_distance():
    results = [
        {"text": "some chunk text", "source": "doc.txt", "chunk_index": 2, "distance": 0.1234},
    ]

    formatted = agent._format_search_results(results)

    assert "doc.txt" in formatted
    assert "chunk 2" in formatted
    assert "0.1234" in formatted
    assert "some chunk text" in formatted


def test_execute_tool_dispatches_search_sop_documents(monkeypatch):
    # WHY monkeypatch agent.query_sop_documents (the name bound in agent's
    # own module namespace via `from embed import query_sop_documents`)
    # rather than embed.query_sop_documents: agent.py already holds its own
    # reference to the function, so patching embed's copy wouldn't affect
    # what agent._execute_tool actually calls.
    stub = MagicMock(return_value=[{"text": "t", "source": "s.txt", "chunk_index": 0, "distance": 0.5}])
    monkeypatch.setattr(agent, "query_sop_documents", stub)

    output = agent._execute_tool(collection="fake-collection", tool_name="search_sop_documents", tool_input={"query": "hello"})

    stub.assert_called_once_with("fake-collection", "hello", n_results=agent.RESULTS_PER_SEARCH)
    assert "s.txt" in output


def test_execute_tool_raises_on_unknown_tool_name():
    try:
        agent._execute_tool(collection=None, tool_name="not_a_real_tool", tool_input={})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_conversation_turn_returns_text_immediately_when_no_tool_use():
    client = MagicMock()
    client.messages.create.return_value = _text_response("plain answer")

    messages = [{"role": "user", "content": "some question"}]
    answer = agent.run_conversation_turn(client, collection=None, messages=messages)

    assert answer == "plain answer"
    # WHY assert exactly one call: with no tool_use, the loop should return
    # on the first round-trip rather than looping further.
    assert client.messages.create.call_count == 1
    # WHY assert the assistant turn was appended: subsequent turns in a
    # multi-turn conversation depend on full history being preserved.
    assert messages[-1]["role"] == "assistant"


def test_run_conversation_turn_executes_tool_then_returns_final_answer(monkeypatch):
    client = MagicMock()
    client.messages.create.side_effect = [
        _tool_use_response("toolu_1", "verify identity"),
        _text_response("final grounded answer"),
    ]
    monkeypatch.setattr(agent, "_execute_tool", MagicMock(return_value="fake search results"))

    messages = [{"role": "user", "content": "how do I verify identity?"}]
    answer = agent.run_conversation_turn(client, collection=None, messages=messages)

    assert answer == "final grounded answer"
    assert client.messages.create.call_count == 2
    # WHY inspect the tool_result message specifically: this is the part of
    # the protocol most likely to break silently (wrong tool_use_id, wrong
    # role, wrong content shape) without raising an error, so it's worth
    # asserting on directly rather than only checking the final answer.
    tool_result_message = messages[-2]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["tool_use_id"] == "toolu_1"
    assert tool_result_message["content"][0]["content"] == "fake search results"


def test_run_conversation_turn_stops_after_max_iterations():
    client = MagicMock()
    client.messages.create.return_value = _tool_use_response("toolu_x", "anything")

    with mock.patch.object(agent, "_execute_tool", return_value="results"):
        messages = [{"role": "user", "content": "a question that never resolves"}]
        answer = agent.run_conversation_turn(client, collection=None, messages=messages)

    assert "wasn't able to reach a final answer" in answer
    assert client.messages.create.call_count == agent.MAX_TOOL_ITERATIONS
