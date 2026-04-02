"""Tests for planagent.ui — option parsing and rendering helpers."""

import pytest
from planagent.ui import parse_agent_response


class TestParseAgentResponse:
    """Validate that ```options blocks are correctly extracted."""

    def test_no_options(self):
        raw = "What are you building? Tell me about your project."
        text, options = parse_agent_response(raw)
        assert text == raw.strip()
        assert options == []

    def test_single_options_block(self):
        raw = (
            "Which database do you prefer?\n\n"
            "```options\n"
            '["PostgreSQL — battle-tested relational DB", '
            '"MongoDB — flexible document store", '
            '"SQLite — simple, zero-config"]\n'
            "```"
        )
        text, options = parse_agent_response(raw)
        assert "Which database" in text
        assert "```options" not in text
        assert len(options) == 3
        assert "PostgreSQL" in options[0]

    def test_options_capped_at_three(self):
        raw = (
            "Pick one:\n\n"
            "```options\n"
            '["A", "B", "C", "D", "E"]\n'
            "```"
        )
        text, options = parse_agent_response(raw)
        assert len(options) == 3

    def test_malformed_json_returns_no_options(self):
        raw = (
            "Here you go:\n\n"
            "```options\n"
            "not valid json\n"
            "```"
        )
        text, options = parse_agent_response(raw)
        assert options == []
        assert "Here you go" in text

    def test_empty_input(self):
        text, options = parse_agent_response("")
        assert text == ""
        assert options == []

    def test_options_block_in_middle(self):
        raw = (
            "I recommend PostgreSQL.\n\n"
            "```options\n"
            '["Yes, go with PostgreSQL", "No, I prefer MySQL", "Let me think"]\n'
            "```\n\n"
            "Let me know!"
        )
        text, options = parse_agent_response(raw)
        assert len(options) == 3
        assert "```options" not in text
        assert "I recommend" in text
        assert "Let me know" in text

    def test_no_newline_before_closing_backticks(self):
        """LLM sometimes puts closing ``` on the same line as the JSON."""
        raw = (
            "Which payment provider?\n\n"
            "```options\n"
            '["Stripe — strong APIs", "PayPal — flexible billing", "Custom webhook"]```'
        )
        text, options = parse_agent_response(raw)
        assert len(options) == 3
        assert "Stripe" in options[0]
        assert "```options" not in text

    def test_no_newlines_at_all(self):
        """Options block with no newlines around JSON content."""
        raw = (
            "Pick one:\n\n"
            '```options["A", "B", "C"]```'
        )
        text, options = parse_agent_response(raw)
        assert len(options) == 3
        assert "```options" not in text

    def test_conversation_complete_signal_preserved(self):
        raw = (
            "Great, I have everything.\n\n"
            "CONVERSATION_COMPLETE"
        )
        text, options = parse_agent_response(raw)
        assert "CONVERSATION_COMPLETE" in text
        assert options == []
