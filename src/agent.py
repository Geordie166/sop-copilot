"""
agent.py — Claude tool-use loop tying retrieval to generation.

WHY THIS MODULE EXISTS:
embed.py can already find the chunks most relevant to a question, but it
can't answer in natural language or decide *when* a search is even needed.
This module gives Claude a "search_sop_documents" tool backed by
query_sop_documents() from embed.py. Claude decides whether/how to call
it, we execute the actual vector search on its behalf, and Claude turns
the retrieved chunks into a grounded, cited answer. That hand-off — model
decides, code executes, model synthesizes — is the "agentic" part of RAG;
without it we'd just have a search engine, not something that can reason
over what it found.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from embed import get_chroma_collection, query_sop_documents

load_dotenv()

# WHY claude-sonnet-5 specifically: it's the current generally-available
# Sonnet model at time of writing, and Sonnet-tier models are a good
# balance of tool-use reliability and cost for a project like this —
# Opus would be overkill (and pricier) for answering questions over a
# handful of short SOP documents, and Haiku is tuned more for speed/cost
# than for careful multi-step tool reasoning.
MODEL = "claude-sonnet-5"

# WHY cap tool-use rounds instead of looping unconditionally: Claude could
# in principle keep calling the search tool repeatedly (e.g. rephrasing a
# query that isn't matching well). A hard cap prevents a runaway loop from
# silently burning API calls if that ever happens, while still allowing
# more than one round for genuinely multi-part questions.
MAX_TOOL_ITERATIONS = 3

# WHY n_results=3 here matches embed.py's own default: keeping the two in
# sync means the "top 3" contract established during the retrieval phase
# carries through to what the agent actually shows Claude. It's exposed as
# a separate constant (not just relying on query_sop_documents' default)
# so it's visible and tunable from the agent layer without having to go
# reread embed.py.
RESULTS_PER_SEARCH = 3

# WHY spell out the citation and "don't guess" rules in the system prompt
# rather than trusting Claude to infer them: for an SOP/compliance
# assistant, a confident-sounding but ungrounded answer is worse than no
# answer — a technician following a hallucinated lockout step, or a rep
# citing a made-up escalation rule, is a real safety/compliance risk. The
# system prompt makes citation and honesty about missing information
# explicit requirements rather than hoped-for behavior.
SYSTEM_PROMPT = """You are SOP Copilot, an internal assistant that answers \
questions about a company's standard operating procedures (SOPs) and root \
cause analysis (RCA) reports.

You have access to a search_sop_documents tool that searches the document \
library and returns the most relevant text chunks along with their source \
document and a relevance distance (lower distance means a closer match).

Rules:
1. Always use the search_sop_documents tool before answering a question \
about procedures, policies, or past incidents. Do not answer from general \
knowledge about call centers or manufacturing in general.
2. Always cite the source document filename(s) your answer is based on.
3. If the search results don't clearly answer the question (e.g. the \
distances are high, or the returned chunks are about a different topic), \
say so plainly rather than guessing or extrapolating beyond what the \
documents say.
4. Keep answers concise and procedural where the source material is \
procedural — numbered steps in, numbered steps out."""

# WHY this schema shape (a single "query" string) instead of separate
# structured fields (e.g. document type, equipment): the underlying
# retrieval is a semantic vector search, not a structured filter query, so
# a free-text query is what the tool can actually use. Keeping the
# input_schema minimal also gives Claude less to get wrong when
# constructing the call.
SEARCH_TOOL = {
    "name": "search_sop_documents",
    "description": (
        "Search the SOP and RCA document library for chunks of text "
        "relevant to a natural-language question. Returns up to "
        f"{RESULTS_PER_SEARCH} results, each with its source document "
        "name, chunk index, relevance distance, and text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A natural-language search query describing what "
                    "information is needed, e.g. 'steps to verify caller "
                    "identity before discussing PHI'."
                ),
            }
        },
        "required": ["query"],
    },
}


def _format_search_results(results: list[dict]) -> str:
    """
    Render query_sop_documents() output as plain text for a tool_result
    message.

    WHY plain text instead of passing the list of dicts as JSON: Claude
    reads tool results as text either way, and a short, labeled plain-text
    block is easier for a model to skim and quote accurately (e.g. citing
    the right filename) than parsing structure out of raw JSON with no
    formatting.
    """
    if not results:
        return "No results found."

    blocks = []
    for rank, result in enumerate(results, start=1):
        blocks.append(
            f"Result {rank} (source: {result['source']}, "
            f"chunk {result['chunk_index']}, distance: {result['distance']:.4f})\n"
            f"{result['text']}"
        )
    return "\n\n".join(blocks)


def _execute_tool(collection, tool_name: str, tool_input: dict) -> str:
    """
    Dispatch a single tool_use block to the corresponding real function.

    WHY a dispatcher function even though there's only one tool right now:
    it's the natural seam for adding more tools later (e.g. a tool to list
    all available document titles) without restructuring the main loop in
    ask() — new tools get a branch here, not a rewrite of the loop.
    """
    if tool_name == "search_sop_documents":
        results = query_sop_documents(
            collection, tool_input["query"], n_results=RESULTS_PER_SEARCH
        )
        return _format_search_results(results)

    # WHY raise instead of silently returning an empty string: an unknown
    # tool name means the model and the code have drifted out of sync
    # (e.g. a tool was renamed in SEARCH_TOOL but not here). Failing loudly
    # during development is more useful than a confusing empty answer.
    raise ValueError(f"Unknown tool: {tool_name}")


def run_conversation_turn(client: Anthropic, collection, messages: list) -> str:
    """
    Run one user turn through the tool-use loop and return Claude's final
    text answer, appending every message generated along the way (both
    Claude's turns and tool results) to the `messages` list in place.

    WHY this takes and mutates a shared `messages` list rather than
    building a fresh one per call: that's what makes multi-turn
    conversation possible. The interactive CLI below calls this once per
    user question but keeps reusing the same list, so a follow-up question
    like "what about after two failed attempts?" still has the prior
    question and answer in context. A single-shot caller (see ask() below)
    can still get the old stateless behavior by just passing a
    fresh one-message list.

    WHY a while loop bounded by MAX_TOOL_ITERATIONS rather than a single
    request/response: Claude's response can have stop_reason == "tool_use",
    meaning it wants to call a tool before it can answer. We execute the
    tool, feed the result back as a new message, and ask again — repeating
    until Claude returns a normal text response (stop_reason != "tool_use")
    or we hit the iteration cap.
    """
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            messages=messages,
        )

        # WHY append response.content (not just the final text) to
        # messages: the Claude API's multi-turn tool-use protocol requires
        # the full assistant turn — including any tool_use blocks — to be
        # present in message history so the subsequent tool_result can be
        # correctly associated with the tool call it's answering.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # WHY return only the text blocks: a final response's content
            # list could in principle mix text with other block types;
            # joining just the text blocks gives a clean string answer
            # rather than leaking API response structure to the caller.
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output = _execute_tool(collection, block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    # WHY a distinct fallback message instead of raising: hitting the
    # iteration cap is an unusual but not catastrophic situation (e.g. the
    # model kept re-searching without converging). Returning an explicit
    # message keeps the caller's contract simple (ask() always returns a
    # string) while still making the failure mode visible to the user.
    return (
        "I wasn't able to reach a final answer within the allotted "
        "number of tool calls. Please try rephrasing your question."
    )


def ask(client: Anthropic, collection, question: str) -> str:
    """
    Single-shot convenience wrapper around run_conversation_turn(): ask one
    question with no memory of any prior turn.

    WHY keep this alongside run_conversation_turn instead of just inlining
    it at call sites: --demo mode and the test suite both want "ask one
    question, get one answer" without having to construct and discard a
    messages list themselves.
    """
    messages = [{"role": "user", "content": question}]
    return run_conversation_turn(client, collection, messages)


def run_demo(client: Anthropic, collection) -> None:
    """
    Run two canned questions non-interactively.

    WHY two test questions instead of one: the first exercises the normal
    "good match found, answer with citation" path. The second deliberately
    asks about something outside the document library, to verify the "say
    so plainly rather than guessing" rule actually holds up in practice
    rather than just existing in the prompt.
    """
    test_questions = [
        "What are the steps to verify a caller's identity before discussing member information?",
        "What is the company's policy on employee vacation accrual?",
    ]

    for question in test_questions:
        print(f"Q: {question}")
        answer = ask(client, collection, question)
        print(f"A: {answer}\n")
        print("-" * 80)


def run_interactive(client: Anthropic, collection) -> None:
    """
    A simple REPL: keep asking the user for questions and printing Claude's
    answers until they quit, preserving conversation history across turns.

    WHY conversation history persists across the whole session (not reset
    per question): a "copilot" is meant to be talked with, not just
    queried once — a rep should be able to ask a follow-up ("what if
    verification fails twice?") without re-stating the original question's
    context. Each call to run_conversation_turn() appends to the same
    `messages` list, so later turns still have earlier Q&A available.
    """
    print("SOP Copilot — ask a question about company SOPs/RCAs.")
    print("Type 'exit' or 'quit' to leave.\n")

    messages: list = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return

        messages.append({"role": "user", "content": question})
        answer = run_conversation_turn(client, collection, messages)
        print(f"SOP Copilot: {answer}\n")


if __name__ == "__main__":
    import argparse

    # WHY argparse for a single optional flag: it's the standard library
    # way to make "--demo" self-documenting (--help explains it) without
    # writing custom sys.argv parsing for what's currently a one-flag CLI.
    parser = argparse.ArgumentParser(
        description="SOP Copilot — ask questions about internal SOPs/RCAs."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run two canned test questions instead of an interactive session.",
    )
    args = parser.parse_args()

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    collection = get_chroma_collection()

    if args.demo:
        run_demo(client, collection)
    else:
        run_interactive(client, collection)
