"""Tests for session compaction behavior."""

from __future__ import annotations

from realtalk.compact import (
    COMPACT_CONTINUATION_PREAMBLE,
    COMPACT_DIRECT_RESUME_INSTRUCTION,
    COMPACT_RECENT_MESSAGES_NOTE,
    CompactionConfig,
    compact_session,
    estimate_session_tokens,
    format_compact_summary,
    get_compact_continuation_message,
    merge_compact_summaries,
    should_compact,
)
from realtalk.session import (
    MessageAdded,
    MessageRole,
    Session,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    TurnStatus,
    _make_envelope,
    add_assistant_text,
    add_user_text,
    append_event,
    derive_messages,
    end_turn,
    new_session,
    start_turn,
)


def _build_session_with_n_turns(turns: int, text_size: int = 10) -> Session:
    s = new_session("/tmp/test", "realtalk")
    for _ in range(turns):
        s, tid = start_turn(s)
        s, _ = add_user_text(s, tid, "u" * text_size)
        s, _ = add_assistant_text(s, tid, "a" * text_size)
        s = end_turn(s, tid, TurnStatus.COMPLETED)
    return s


def _add_turns(session: Session, count: int, text_size: int = 10) -> Session:
    s = session
    for _ in range(count):
        s, tid = start_turn(s)
        s, _ = add_user_text(s, tid, "u" * text_size)
        s, _ = add_assistant_text(s, tid, "a" * text_size)
        s = end_turn(s, tid, TurnStatus.COMPLETED)
    return s


def _add_message_with_parts(
    session: Session,
    turn_id: str,
    role: MessageRole,
    parts: tuple[TextPart | ToolCallPart | ToolResultPart, ...],
) -> Session:
    envelope = _make_envelope(session, "message_added")
    msg_id = f"msg_{envelope.event_id}"
    event = MessageAdded(
        envelope=envelope,
        message_id=msg_id,
        turn_id=turn_id,
        role=role,
        parts=parts,
        metadata={},
    )
    return append_event(session, event)


def _build_session_with_compacted_summary() -> Session:
    s = new_session("/tmp/test", "realtalk")
    envelope = _make_envelope(s, "message_added")
    summary_text = COMPACT_CONTINUATION_PREAMBLE + (
        "Summary:\nConversation summary:\n- Scope: 999 earlier messages compacted.\n"
    )
    event = MessageAdded(
        envelope=envelope,
        message_id=f"msg_{envelope.event_id}",
        turn_id=None,
        role=MessageRole.SYSTEM,
        parts=(TextPart(content_id="cnt_summary", text=summary_text),),
        metadata={},
    )
    return append_event(s, event)


# --- Token estimation ---

def test_estimate_tokens_empty_session():
    s = new_session("/tmp/test", "realtalk")
    assert estimate_session_tokens(s) == 0


def test_estimate_tokens_proportional_to_content():
    s = new_session("/tmp/test", "realtalk")
    s, tid = start_turn(s)
    s, _ = add_user_text(s, tid, "x" * 4000)
    tokens = estimate_session_tokens(s)
    assert 800 <= tokens <= 1200


# --- should_compact ---

def test_should_compact_below_threshold():
    s = new_session("/tmp/test", "realtalk")
    config = CompactionConfig(max_estimated_tokens=80_000)
    assert should_compact(s, config) is False


def test_should_compact_above_threshold():
    s = _build_session_with_n_turns(20, text_size=20_000)
    config = CompactionConfig(max_estimated_tokens=1)
    assert should_compact(s, config) is True


def test_should_compact_ignores_existing_summary_prefix():
    s = _build_session_with_compacted_summary()
    s = _add_turns(s, 2, text_size=10)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1000)
    assert should_compact(s, config) is False


def test_should_compact_requires_enough_messages():
    s = new_session("/tmp/test", "realtalk")
    s, tid = start_turn(s)
    s, _ = add_user_text(s, tid, "x" * 400_000)
    config = CompactionConfig(preserve_recent_messages=4, max_estimated_tokens=1)
    assert should_compact(s, config) is False


# --- compact_session: basic ---

def test_compact_leaves_small_sessions_unchanged():
    s = _build_session_with_n_turns(2)
    config = CompactionConfig()
    result = compact_session(s, config)
    assert result.removed_message_count == 0
    assert result.compacted_session.session_id == s.session_id
    assert result.summary == ""


def test_compact_removes_older_messages():
    s = _build_session_with_n_turns(10, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1)
    result = compact_session(s, config)
    assert result.removed_message_count > 0
    messages = derive_messages(result.compacted_session)
    assert messages[0].role == MessageRole.SYSTEM
    assert "Summary:" in messages[0].parts[0].text


def test_compact_preserves_recent_messages():
    s = _build_session_with_n_turns(10, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=4, max_estimated_tokens=1)
    result = compact_session(s, config)
    messages = derive_messages(result.compacted_session)
    assert len(messages) <= 5


def test_compact_preserves_session_metadata():
    s = _build_session_with_n_turns(10, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1)
    result = compact_session(s, config)
    assert result.compacted_session.session_id == s.session_id
    assert result.compacted_session.workspace_root == s.workspace_root


# --- Structured summary ---

def test_compact_summary_has_scope():
    s = _build_session_with_n_turns(10, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1)
    result = compact_session(s, config)
    assert "Scope:" in result.formatted_summary
    assert "earlier messages compacted" in result.formatted_summary


def test_compact_summary_has_timeline():
    s = _build_session_with_n_turns(10, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1)
    result = compact_session(s, config)
    assert "Key timeline:" in result.formatted_summary


def test_compact_summary_includes_tool_names():
    s = new_session("/tmp/test", "realtalk")
    s, tid = start_turn(s)
    s, _ = add_user_text(s, tid, "do work")
    call_part = ToolCallPart(
        content_id="cnt_call",
        tool_call_id="call_1",
        tool_name="update_mood",
        input_json='{"delta": 5}',
    )
    s = _add_message_with_parts(s, tid, MessageRole.ASSISTANT, (call_part,))
    result_part = ToolResultPart(
        content_id="cnt_result",
        tool_result_id="res_1",
        tool_call_id="call_1",
        output_text="mood is now 65",
        is_error=False,
    )
    s = _add_message_with_parts(s, tid, MessageRole.USER, (result_part,))
    s, _ = add_assistant_text(s, tid, "ok")

    config = CompactionConfig(preserve_recent_messages=1, max_estimated_tokens=1)
    result = compact_session(s, config)
    assert "Tools mentioned:" in result.formatted_summary


# --- Continuation message ---

def test_continuation_message_has_preamble():
    s = _build_session_with_n_turns(10, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1)
    result = compact_session(s, config)
    messages = derive_messages(result.compacted_session)
    text = messages[0].parts[0].text
    assert "continued from a previous conversation" in text.lower()


def test_continuation_message_has_resume_instruction():
    s = _build_session_with_n_turns(10, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1)
    result = compact_session(s, config)
    messages = derive_messages(result.compacted_session)
    text = messages[0].parts[0].text
    assert "do not acknowledge the summary" in text.lower()


def test_continuation_message_has_recent_note():
    s = _build_session_with_n_turns(10, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1)
    result = compact_session(s, config)
    messages = derive_messages(result.compacted_session)
    text = messages[0].parts[0].text
    assert "recent messages are preserved" in text.lower()


# --- Tool boundary ---

def test_compact_does_not_split_tool_use_tool_result_pair():
    s = new_session("/tmp/test", "realtalk")
    s, tid = start_turn(s)
    s, _ = add_user_text(s, tid, "before tools")

    call_part = ToolCallPart(
        content_id="cnt_call",
        tool_call_id="call_1",
        tool_name="generate_options",
        input_json='{"count": 3}',
    )
    s = _add_message_with_parts(s, tid, MessageRole.ASSISTANT, (call_part,))
    result_part = ToolResultPart(
        content_id="cnt_result",
        tool_result_id="res_1",
        tool_call_id="call_1",
        output_text="done",
        is_error=False,
    )
    s = _add_message_with_parts(s, tid, MessageRole.USER, (result_part,))

    config = CompactionConfig(preserve_recent_messages=1, max_estimated_tokens=1)
    result = compact_session(s, config)
    messages = derive_messages(result.compacted_session)

    assert len(messages) == 3
    assert any(isinstance(p, ToolCallPart) for p in messages[1].parts)
    assert isinstance(messages[2].parts[0], ToolResultPart)


# --- Re-compaction ---

def test_recompact_merges_previous_summary():
    s = _build_session_with_n_turns(8, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1)
    first = compact_session(s, config)

    s2 = _add_turns(first.compacted_session, 4, text_size=2000)
    second = compact_session(s2, config)

    assert "Previously compacted context:" in second.formatted_summary
    assert "Newly compacted context:" in second.formatted_summary


def test_recompact_preserves_first_summary_content():
    s = _build_session_with_n_turns(8, text_size=2000)
    config = CompactionConfig(preserve_recent_messages=2, max_estimated_tokens=1)
    first = compact_session(s, config)

    s2 = _add_turns(first.compacted_session, 4, text_size=2000)
    second = compact_session(s2, config)

    assert "earlier messages compacted" in second.formatted_summary


# --- format_compact_summary ---

def test_format_compact_summary_strips_tags():
    summary = "<summary>Kept work</summary>"
    assert format_compact_summary(summary) == "Summary:\nKept work"


def test_format_compact_summary_strips_analysis_blocks():
    summary = "<analysis>secret</analysis><summary>Visible</summary>"
    formatted = format_compact_summary(summary)
    assert "secret" not in formatted
    assert "Visible" in formatted


def test_format_compact_summary_collapses_blank_lines():
    summary = "<summary>Line 1\n\n\nLine 2</summary>"
    formatted = format_compact_summary(summary)
    assert "\n\n\n" not in formatted


# --- get_compact_continuation_message ---

def test_get_compact_continuation_message_has_all_sections():
    msg = get_compact_continuation_message(
        "<summary>Text</summary>",
        suppress_follow_up_questions=True,
        recent_messages_preserved=True,
    )
    assert msg.startswith(COMPACT_CONTINUATION_PREAMBLE)
    assert COMPACT_RECENT_MESSAGES_NOTE in msg
    assert COMPACT_DIRECT_RESUME_INSTRUCTION in msg


def test_get_compact_continuation_message_omits_optional_sections():
    msg = get_compact_continuation_message(
        "<summary>Text</summary>",
        suppress_follow_up_questions=False,
        recent_messages_preserved=False,
    )
    assert COMPACT_RECENT_MESSAGES_NOTE not in msg
    assert COMPACT_DIRECT_RESUME_INSTRUCTION not in msg


# --- merge_compact_summaries ---

def test_merge_compact_summaries_prefers_new_timeline():
    existing = """
    <summary>
    Conversation summary:
    - Scope: 3 earlier messages compacted.
    - Key timeline:
      - user: old timeline
    </summary>
    """.strip()
    new = """
    <summary>
    Conversation summary:
    - Scope: 2 earlier messages compacted.
    - Key timeline:
      - user: new timeline
    </summary>
    """.strip()

    merged = merge_compact_summaries(existing, new)
    assert "new timeline" in merged
    assert "old timeline" not in merged
