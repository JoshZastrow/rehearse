"""
realtalk.compact -- Layer 6: session compaction.

Summarizes older messages to prevent context overflow while preserving recent turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from realtalk.session import (
    ContentPart,
    JSONValue,
    Message,
    MessageAdded,
    MessageRole,
    Session,
    SessionEvent,
    TextPart,
    ToolCallPart,
    ToolCallRecorded,
    ToolResultPart,
    ToolResultRecorded,
    _make_envelope,
    append_event,
    derive_messages,
)

COMPACT_CONTINUATION_PREAMBLE = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.\n\n"
)

COMPACT_RECENT_MESSAGES_NOTE = "Recent messages are preserved verbatim."

COMPACT_DIRECT_RESUME_INSTRUCTION = (
    "Continue the conversation from where it left off without asking the user "
    "any further questions. Resume directly -- do not acknowledge the summary, "
    "do not recap what was happening, and do not preface with continuation text."
)


@dataclass(frozen=True)
class CompactionConfig:
    """Thresholds controlling when and how a session is compacted."""

    preserve_recent_messages: int = 4
    max_estimated_tokens: int = 80_000


@dataclass(frozen=True)
class CompactionResult:
    """Result of compacting a session."""

    summary: str
    formatted_summary: str
    compacted_session: Session
    removed_message_count: int


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_event_tokens(event: SessionEvent) -> int:
    """Rough token estimate for one event. Heuristic: len/4 + 1."""
    if isinstance(event, MessageAdded):
        total_len = 0
        for part in event.parts:
            if isinstance(part, TextPart):
                total_len += len(part.text)
            elif isinstance(part, ToolCallPart):
                total_len += len(part.tool_name) + len(part.input_json)
            elif isinstance(part, ToolResultPart):
                total_len += len(part.tool_call_id) + len(part.output_text)
        return total_len // 4 + 1
    if isinstance(event, ToolCallRecorded):
        return (len(event.tool_name) + len(event.input_json)) // 4 + 1
    if isinstance(event, ToolResultRecorded):
        return (len(event.tool_call_id) + len(event.output_text)) // 4 + 1
    return 0


def estimate_session_tokens(session: Session) -> int:
    """Sum token estimates across all session events."""
    return sum(estimate_event_tokens(e) for e in session.events)


def _estimate_message_tokens(message: Message) -> int:
    total_len = 0
    for part in message.parts:
        if isinstance(part, TextPart):
            total_len += len(part.text)
        elif isinstance(part, ToolCallPart):
            total_len += len(part.tool_name) + len(part.input_json)
        elif isinstance(part, ToolResultPart):
            total_len += len(part.tool_call_id) + len(part.output_text)
    return total_len // 4 + 1


# ---------------------------------------------------------------------------
# Compaction logic
# ---------------------------------------------------------------------------


def should_compact(session: Session, config: CompactionConfig) -> bool:
    """True when the session exceeds the compaction budget.

    Skips the compacted summary prefix (if present) when counting.
    Requires BOTH: enough messages to preserve AND enough tokens.
    """
    messages = derive_messages(session)
    start = _compacted_summary_prefix_len(messages)
    compactable = messages[start:]

    if len(compactable) <= config.preserve_recent_messages:
        return False

    token_sum = sum(_estimate_message_tokens(m) for m in compactable)
    return token_sum >= config.max_estimated_tokens


def compact_session(session: Session, config: CompactionConfig) -> CompactionResult:
    """Compact a session by summarizing older messages and preserving the recent tail."""
    if not should_compact(session, config):
        return CompactionResult(
            summary="",
            formatted_summary="",
            compacted_session=session,
            removed_message_count=0,
        )

    messages = derive_messages(session)
    existing_summary = _extract_existing_compacted_summary(messages)
    prefix_len = 1 if existing_summary else 0

    raw_keep_from = max(prefix_len, len(messages) - config.preserve_recent_messages)
    keep_from = _safe_boundary(messages, raw_keep_from, prefix_len)

    removed = messages[prefix_len:keep_from]
    preserved = messages[keep_from:]

    summary = merge_compact_summaries(existing_summary, summarize_messages(removed))
    formatted_summary = format_compact_summary(summary)
    continuation = get_compact_continuation_message(
        summary,
        suppress_follow_up_questions=True,
        recent_messages_preserved=bool(preserved),
    )

    compacted_session = _rebuild_session(session, continuation, preserved)

    return CompactionResult(
        summary=summary,
        formatted_summary=formatted_summary,
        compacted_session=compacted_session,
        removed_message_count=len(removed),
    )


# ---------------------------------------------------------------------------
# Summary formatting and merging
# ---------------------------------------------------------------------------


def summarize_messages(messages: list[Message]) -> str:
    """Build a structured summary of removed messages."""
    user_count = sum(1 for m in messages if m.role == MessageRole.USER)
    assistant_count = sum(1 for m in messages if m.role == MessageRole.ASSISTANT)
    tool_count = sum(1 for m in messages if m.role == MessageRole.TOOL)

    tool_names = sorted(set(_collect_tool_names(messages)))
    recent_user = _collect_recent_role_summaries(messages, MessageRole.USER, limit=3)
    pending = _infer_pending_work(messages)
    current = _infer_current_work(messages)

    lines = [
        "<summary>",
        "Conversation summary:",
        (
            f"- Scope: {len(messages)} earlier messages compacted "
            f"(user={user_count}, assistant={assistant_count}, tool={tool_count})."
        ),
    ]

    if tool_names:
        lines.append(f"- Tools mentioned: {', '.join(tool_names)}.")

    if recent_user:
        lines.append("- Recent user requests:")
        lines.extend(f"  - {r}" for r in recent_user)

    if pending:
        lines.append("- Pending work:")
        lines.extend(f"  - {p}" for p in pending)

    if current:
        lines.append(f"- Current work: {current}")

    lines.append("- Key timeline:")
    for msg in messages:
        role = msg.role.value
        content = _summarize_message_content(msg)
        lines.append(f"  - {role}: {content}")

    lines.append("</summary>")
    return "\n".join(lines)


def merge_compact_summaries(existing: str | None, new_summary: str) -> str:
    """Merge a previous compacted summary with a new one."""
    if existing is None:
        return new_summary

    prev_highlights = _extract_summary_highlights(existing)
    new_highlights = _extract_summary_highlights(new_summary)
    new_timeline = _extract_summary_timeline(new_summary)

    lines = ["<summary>", "Conversation summary:"]

    if prev_highlights:
        lines.append("- Previously compacted context:")
        lines.extend(f"  {h}" for h in prev_highlights)

    if new_highlights:
        lines.append("- Newly compacted context:")
        lines.extend(f"  {h}" for h in new_highlights)

    if new_timeline:
        lines.append("- Key timeline:")
        lines.extend(f"  {t}" for t in new_timeline)

    lines.append("</summary>")
    return "\n".join(lines)


def format_compact_summary(summary: str) -> str:
    """Normalize a compaction summary into human-readable text."""
    without_analysis = _strip_tag_block(summary, "analysis")
    content = _extract_tag_block(without_analysis, "summary")
    if content is not None:
        formatted = without_analysis.replace(
            f"<summary>{content}</summary>",
            f"Summary:\n{content.strip()}",
        )
    else:
        formatted = without_analysis
    return _collapse_blank_lines(formatted).strip()


def get_compact_continuation_message(
    summary: str,
    suppress_follow_up_questions: bool = True,
    recent_messages_preserved: bool = True,
) -> str:
    """Build the synthetic system message used after compaction."""
    base = COMPACT_CONTINUATION_PREAMBLE + format_compact_summary(summary)

    if recent_messages_preserved:
        base += "\n\n" + COMPACT_RECENT_MESSAGES_NOTE

    if suppress_follow_up_questions:
        base += "\n" + COMPACT_DIRECT_RESUME_INSTRUCTION

    return base


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_boundary(messages: list[Message], raw_keep_from: int, prefix_len: int) -> int:
    """Walk the compaction boundary backward to avoid splitting tool pairs."""
    if raw_keep_from >= len(messages):
        return len(messages)

    k = raw_keep_from
    while k > prefix_len:
        first_preserved = messages[k]
        if not _starts_with_tool_result(first_preserved):
            break
        preceding = messages[k - 1]
        if _has_tool_use(preceding):
            k -= 1
            break
        k -= 1
    return k


def _rebuild_session(
    session: Session, continuation: str, preserved: Iterable[Message]
) -> Session:
    """Rebuild a compacted session with a summary system message + preserved tail."""
    if not session.events:
        return session

    base = Session(
        session_id=session.session_id,
        created_at=session.created_at,
        workspace_root=session.workspace_root,
        game_name=session.game_name,
        model_hint=session.model_hint,
        events=(session.events[0],),
        metadata=session.metadata,
    )

    base = _append_message(base, MessageRole.SYSTEM, continuation)

    for msg in preserved:
        base = _append_message(base, msg.role, msg.parts, metadata=msg.metadata)

    return base


def _append_message(
    session: Session,
    role: MessageRole,
    text_or_parts: str | tuple[ContentPart, ...],
    metadata: Mapping[str, JSONValue] | None = None,
) -> Session:
    envelope = _make_envelope(session, "message_added")
    if isinstance(text_or_parts, str):
        parts = (TextPart(content_id=_make_content_id(envelope.event_id), text=text_or_parts),)
    else:
        parts = tuple(text_or_parts)
    event = MessageAdded(
        envelope=envelope,
        message_id=_make_message_id(envelope.event_id),
        turn_id=None,
        role=role,
        parts=parts,
        metadata=metadata or {},
    )
    return append_event(session, event)


def _make_message_id(seed: str) -> str:
    return f"msg_{seed}"


def _make_content_id(seed: str) -> str:
    return f"cnt_{seed}"


def _extract_existing_compacted_summary(messages: list[Message]) -> str | None:
    if not messages:
        return None
    first = messages[0]
    if first.role != MessageRole.SYSTEM:
        return None
    text = _first_text(first) or ""
    if not text.startswith(COMPACT_CONTINUATION_PREAMBLE):
        return None
    summary = text[len(COMPACT_CONTINUATION_PREAMBLE) :]
    for suffix in [
        f"\n\n{COMPACT_RECENT_MESSAGES_NOTE}",
        f"\n{COMPACT_DIRECT_RESUME_INSTRUCTION}",
    ]:
        if suffix in summary:
            summary = summary.split(suffix)[0]
    return summary.strip()


def _compacted_summary_prefix_len(messages: list[Message]) -> int:
    return 1 if _extract_existing_compacted_summary(messages) is not None else 0


def _starts_with_tool_result(message: Message) -> bool:
    if not message.parts:
        return False
    return isinstance(message.parts[0], ToolResultPart)


def _has_tool_use(message: Message) -> bool:
    return any(isinstance(part, ToolCallPart) for part in message.parts)


def _collect_tool_names(messages: list[Message]) -> list[str]:
    names: list[str] = []
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                names.append(part.tool_name)
    return names


def _collect_recent_role_summaries(
    messages: list[Message], role: MessageRole, limit: int
) -> list[str]:
    role_texts = [
        _first_text(m)
        for m in messages
        if m.role == role and _first_text(m)
    ]
    return [_truncate(t, 160) for t in role_texts[-limit:]]


def _infer_pending_work(messages: list[Message]) -> list[str]:
    keywords = {"todo", "next", "pending", "follow up", "remaining"}
    results: list[str] = []
    for msg in reversed(messages):
        text = _first_text(msg)
        if text and any(kw in text.lower() for kw in keywords):
            results.append(_truncate(text, 160))
        if len(results) >= 3:
            break
    return list(reversed(results))


def _infer_current_work(messages: list[Message]) -> str | None:
    for msg in reversed(messages):
        text = _first_text(msg)
        if text and text.strip():
            return _truncate(text, 200)
    return None


def _summarize_message_content(message: Message) -> str:
    parts: list[str] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            parts.append(_truncate(part.text, 160))
        elif isinstance(part, ToolCallPart):
            parts.append(f"tool_use {part.tool_name}({_truncate(part.input_json, 80)})")
        elif isinstance(part, ToolResultPart):
            error_prefix = "error " if part.is_error else ""
            parts.append(
                f"tool_result {part.tool_call_id}: {error_prefix}{_truncate(part.output_text, 80)}"
            )
    return " | ".join(parts)


def _first_text(message: Message) -> str | None:
    for part in message.parts:
        if isinstance(part, TextPart) and part.text.strip():
            return part.text
    return None


def _truncate(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "..."


def _extract_tag_block(content: str, tag: str) -> str | None:
    start = f"<{tag}>"
    end = f"</{tag}>"
    s = content.find(start)
    if s == -1:
        return None
    s += len(start)
    e = content.find(end, s)
    if e == -1:
        return None
    return content[s:e]


def _strip_tag_block(content: str, tag: str) -> str:
    start = f"<{tag}>"
    end = f"</{tag}>"
    s = content.find(start)
    e = content.find(end)
    if s == -1 or e == -1:
        return content
    return content[:s] + content[e + len(end) :]


def _collapse_blank_lines(content: str) -> str:
    result: list[str] = []
    last_blank = False
    for line in content.splitlines():
        is_blank = not line.strip()
        if is_blank and last_blank:
            continue
        result.append(line)
        last_blank = is_blank
    return "\n".join(result)


def _extract_summary_highlights(summary: str) -> list[str]:
    lines: list[str] = []
    in_timeline = False
    for line in format_compact_summary(summary).splitlines():
        trimmed = line.rstrip()
        if not trimmed or trimmed in ("Summary:", "Conversation summary:"):
            continue
        if trimmed == "- Key timeline:":
            in_timeline = True
            continue
        if in_timeline:
            continue
        lines.append(trimmed)
    return lines


def _extract_summary_timeline(summary: str) -> list[str]:
    lines: list[str] = []
    in_timeline = False
    for line in format_compact_summary(summary).splitlines():
        trimmed = line.rstrip()
        if trimmed == "- Key timeline:":
            in_timeline = True
            continue
        if not in_timeline:
            continue
        if not trimmed:
            break
        lines.append(trimmed)
    return lines
