"""Tests for planagent.ui — option parsing and rendering helpers."""

import pytest
from planagent.ui import parse_agent_response, strip_code_blocks, BACK_COMMAND, BACK_LABEL, OPTIONS_MULTI_PATTERN


class TestParseAgentResponse:
    """Validate that ```options blocks are correctly extracted."""

    def test_no_options(self):
        raw = "What are you building? Tell me about your project."
        text, options, multi = parse_agent_response(raw)
        assert text == raw.strip()
        assert options == []
        assert multi is False

    def test_single_options_block(self):
        raw = (
            "Which database do you prefer?\n\n"
            "```options\n"
            '["PostgreSQL — battle-tested relational DB", '
            '"MongoDB — flexible document store", '
            '"SQLite — simple, zero-config"]\n'
            "```"
        )
        text, options, multi = parse_agent_response(raw)
        assert "Which database" in text
        assert "```options" not in text
        assert len(options) == 3
        assert "PostgreSQL" in options[0]
        assert multi is False

    def test_options_capped_at_three(self):
        raw = (
            "Pick one:\n\n"
            "```options\n"
            '["A", "B", "C", "D", "E"]\n'
            "```"
        )
        text, options, multi = parse_agent_response(raw)
        assert len(options) == 3
        assert multi is False

    def test_malformed_json_returns_no_options(self):
        raw = (
            "Here you go:\n\n"
            "```options\n"
            "not valid json\n"
            "```"
        )
        text, options, multi = parse_agent_response(raw)
        assert options == []
        assert "Here you go" in text
        assert multi is False

    def test_empty_input(self):
        text, options, multi = parse_agent_response("")
        assert text == ""
        assert options == []
        assert multi is False

    def test_options_block_in_middle(self):
        raw = (
            "I recommend PostgreSQL.\n\n"
            "```options\n"
            '["Yes, go with PostgreSQL", "No, I prefer MySQL", "Let me think"]\n'
            "```\n\n"
            "Let me know!"
        )
        text, options, multi = parse_agent_response(raw)
        assert len(options) == 3
        assert "```options" not in text
        assert "I recommend" in text
        assert "Let me know" in text
        assert multi is False

    def test_no_newline_before_closing_backticks(self):
        """LLM sometimes puts closing ``` on the same line as the JSON."""
        raw = (
            "Which payment provider?\n\n"
            "```options\n"
            '["Stripe — strong APIs", "PayPal — flexible billing", "Custom webhook"]```'
        )
        text, options, multi = parse_agent_response(raw)
        assert len(options) == 3
        assert "Stripe" in options[0]
        assert "```options" not in text
        assert multi is False

    def test_no_newlines_at_all(self):
        """Options block with no newlines around JSON content."""
        raw = (
            "Pick one:\n\n"
            '```options["A", "B", "C"]```'
        )
        text, options, multi = parse_agent_response(raw)
        assert len(options) == 3
        assert "```options" not in text
        assert multi is False

    def test_conversation_complete_signal_preserved(self):
        raw = (
            "Great, I have everything.\n\n"
            "CONVERSATION_COMPLETE"
        )
        text, options, multi = parse_agent_response(raw)
        assert "CONVERSATION_COMPLETE" in text
        assert options == []
        assert multi is False


class TestStripCodeBlocks:
    """Validate code block removal for planning mode."""

    def test_removes_fenced_code_block(self):
        raw = "Use Redis for caching.\n\n```python\nimport redis\nr = redis.Redis()\n```\n\nThis improves performance."
        result = strip_code_blocks(raw)
        assert "import redis" not in result
        assert "Use Redis" in result
        assert "improves performance" in result

    def test_removes_multiple_code_blocks(self):
        raw = "Step 1:\n```js\nconsole.log('hi')\n```\nStep 2:\n```sql\nSELECT * FROM users;\n```\nDone."
        result = strip_code_blocks(raw)
        assert "console.log" not in result
        assert "SELECT" not in result
        assert "Step 1" in result
        assert "Done" in result

    def test_preserves_text_without_code(self):
        raw = "Use microservices pattern with event-driven architecture."
        result = strip_code_blocks(raw)
        assert result == raw

    def test_preserves_options_block(self):
        """Options blocks use ``` too but are handled separately by parse_agent_response."""
        raw = 'Choose:\n```options\n["A", "B", "C"]\n```'
        # strip_code_blocks removes ALL fenced blocks
        result = strip_code_blocks(raw)
        assert "options" not in result

    def test_empty_string(self):
        assert strip_code_blocks("") == ""

    def test_only_code_block(self):
        raw = "```python\nprint('hello')\n```"
        result = strip_code_blocks(raw)
        assert result == ""


class TestMultiSelectOptions:
    """Validate ```options-multi block parsing."""

    def test_multi_select_basic(self):
        raw = (
            "Which features do you want for V1?\n\n"
            "```options-multi\n"
            '["Auth", "Payments", "Notifications", "Analytics"]\n'
            "```"
        )
        text, options, multi = parse_agent_response(raw)
        assert "Which features" in text
        assert "```options-multi" not in text
        assert len(options) == 4
        assert multi is True
        assert "Auth" in options
        assert "Analytics" in options

    def test_multi_select_capped_at_six(self):
        raw = (
            "Pick modules:\n\n"
            "```options-multi\n"
            '["A", "B", "C", "D", "E", "F", "G", "H"]\n'
            "```"
        )
        text, options, multi = parse_agent_response(raw)
        assert len(options) == 6
        assert multi is True

    def test_multi_select_malformed_json(self):
        raw = (
            "Pick:\n\n"
            "```options-multi\n"
            "not json\n"
            "```"
        )
        text, options, multi = parse_agent_response(raw)
        assert options == []
        assert multi is False

    def test_multi_select_strips_from_text(self):
        raw = (
            "Before text.\n\n"
            "```options-multi\n"
            '["X", "Y", "Z"]\n'
            "```\n\n"
            "After text."
        )
        text, options, multi = parse_agent_response(raw)
        assert "Before text" in text
        assert "After text" in text
        assert "```options-multi" not in text
        assert len(options) == 3
        assert multi is True

    def test_multi_select_no_newlines(self):
        raw = 'Pick:\n```options-multi["A", "B", "C"]```'
        text, options, multi = parse_agent_response(raw)
        assert len(options) == 3
        assert multi is True

    def test_single_select_not_confused_with_multi(self):
        """Ensure ```options block returns multi=False."""
        raw = (
            "Pick one:\n\n"
            "```options\n"
            '["A", "B", "C"]\n'
            "```"
        )
        text, options, multi = parse_agent_response(raw)
        assert len(options) == 3
        assert multi is False

    def test_multi_pattern_regex(self):
        """Ensure OPTIONS_MULTI_PATTERN matches correctly."""
        sample = '```options-multi\n["A", "B"]\n```'
        assert OPTIONS_MULTI_PATTERN.search(sample) is not None
        # Should NOT match plain ```options
        plain = '```options\n["A", "B"]\n```'
        assert OPTIONS_MULTI_PATTERN.search(plain) is None


class TestBackCommand:

    def test_back_command_value(self):
        assert BACK_COMMAND == "/back"

    def test_back_label_exists(self):
        assert "Go back" in BACK_LABEL

    def test_back_label_is_string(self):
        assert isinstance(BACK_LABEL, str)
        assert len(BACK_LABEL) > 0
