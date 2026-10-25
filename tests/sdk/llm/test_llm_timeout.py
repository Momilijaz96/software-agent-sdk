"""Tests for LLM timeout configuration."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from aiohttp import web
from litellm.exceptions import Timeout as LiteLLMTimeout
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
from pydantic import SecretStr

from openhands.sdk.agent import Agent
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.llm import LLM, Message, TextContent
from openhands.sdk.llm.exceptions import LLMTimeoutError


# Default timeout in seconds (5 minutes)
DEFAULT_LLM_TIMEOUT_SECONDS = 300


class TestLLMTimeoutDefaults:
    """Tests for default LLM timeout behavior."""

    def test_default_timeout_is_5_minutes(self):
        """Test that the default LLM timeout is 300 seconds (5 minutes).

        This test ensures that LLM requests have a reasonable default timeout
        to prevent indefinitely hanging requests that could cause runtime
        idle detection to kill active runtimes.

        See: https://github.com/OpenHands/software-agent-sdk/issues/1633
        """
        llm = LLM(model="gpt-4o-mini", usage_id="test-llm")

        assert llm.timeout == DEFAULT_LLM_TIMEOUT_SECONDS, (
            f"Expected default timeout of {DEFAULT_LLM_TIMEOUT_SECONDS}s (5 minutes), "
            f"but got {llm.timeout}. "
            "A reasonable default timeout is needed to prevent LLM calls from "
            "hanging indefinitely and causing runtime idle detection issues."
        )
        assert llm.stream_idle_timeout == DEFAULT_LLM_TIMEOUT_SECONDS
        assert llm.hard_timeout_max_attempts == 2

    def test_hard_timeout_attempt_limit_can_be_changed_per_llm(self):
        llm = LLM(model="gpt-4o-mini", usage_id="test-llm", hard_timeout_max_attempts=4)
        assert llm.hard_timeout_max_attempts == 4
        assert LLM.model_validate(llm.model_dump()).hard_timeout_max_attempts == 4

        with pytest.raises(ValueError):
            LLM(model="gpt-4o-mini", usage_id="test-llm", hard_timeout_max_attempts=0)

    def test_stream_idle_timeout_defaults_to_configured_timeout(self):
        llm = LLM(model="gpt-4o-mini", usage_id="test-llm", timeout=600)
        assert llm.stream_idle_timeout == 600

    def test_stream_idle_timeout_can_be_disabled(self):
        llm = LLM(
            model="gpt-4o-mini",
            usage_id="test-llm",
            timeout=600,
            stream_idle_timeout=None,
        )
        assert llm.timeout == 600
        assert llm.stream_idle_timeout is None

    def test_timeout_can_be_overridden(self):
        """Test that the timeout can be explicitly set to a custom value."""
        custom_timeout = 600  # 10 minutes
        llm = LLM(model="gpt-4o-mini", usage_id="test-llm", timeout=custom_timeout)

        assert llm.timeout == custom_timeout

    def test_timeout_can_be_set_to_none_for_no_timeout(self):
        """Test that timeout can be explicitly set to None to disable timeout.

        Users who need very long LLM calls (e.g., extended reasoning with high
        thinking budgets) can explicitly disable the timeout by setting it to None.
        """
        llm = LLM(model="gpt-4o-mini", usage_id="test-llm", timeout=None)

        # When explicitly set to None, it should remain None
        assert llm.timeout is None

    def test_timeout_validation_rejects_negative_values(self):
        """Test that negative timeout values are rejected."""
        with pytest.raises(Exception):  # ValidationError from pydantic
            LLM(model="gpt-4o-mini", usage_id="test-llm", timeout=-1)

    def test_timeout_accepts_zero(self):
        """Test that zero timeout is valid (immediate timeout)."""
        llm = LLM(model="gpt-4o-mini", usage_id="test-llm", timeout=0)
        assert llm.timeout == 0


class TestLLMTimeoutPassthrough:
    """Tests that timeout is correctly passed to litellm."""

    @patch("openhands.sdk.llm.llm.litellm_completion")
    def test_default_timeout_passed_to_litellm(self, mock_completion):
        """Test that the default timeout is passed to litellm completion calls."""
        from litellm.types.utils import (
            Choices,
            Message as LiteLLMMessage,
            ModelResponse,
            Usage,
        )

        # Create a proper mock response
        mock_response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(content="Test response", role="assistant"),
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_completion.return_value = mock_response

        llm = LLM(
            model="gpt-4o-mini",
            api_key=SecretStr("test_key"),
            usage_id="test-llm",
        )

        messages = [Message(role="user", content=[TextContent(text="Hello")])]
        llm.completion(messages=messages)

        # Verify that timeout was passed to litellm
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]

        assert "timeout" in call_kwargs, "timeout should be passed to litellm"
        assert call_kwargs["timeout"] == DEFAULT_LLM_TIMEOUT_SECONDS, (
            f"Expected timeout of {DEFAULT_LLM_TIMEOUT_SECONDS}s to be passed "
            f"to litellm, but got {call_kwargs['timeout']}"
        )

    @patch("openhands.sdk.llm.llm.litellm_completion")
    def test_custom_timeout_passed_to_litellm(self, mock_completion):
        """Test that a custom timeout is passed to litellm completion calls."""
        from litellm.types.utils import (
            Choices,
            Message as LiteLLMMessage,
            ModelResponse,
            Usage,
        )

        mock_response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(content="Test response", role="assistant"),
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_completion.return_value = mock_response

        custom_timeout = 120
        llm = LLM(
            model="gpt-4o-mini",
            api_key=SecretStr("test_key"),
            usage_id="test-llm",
            timeout=custom_timeout,
        )

        messages = [Message(role="user", content=[TextContent(text="Hello")])]
        llm.completion(messages=messages)

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]

        assert call_kwargs["timeout"] == custom_timeout

    @patch("openhands.sdk.llm.llm.litellm_completion")
    def test_none_timeout_passed_to_litellm(self, mock_completion):
        """Test that None timeout is passed to litellm (no timeout)."""
        from litellm.types.utils import (
            Choices,
            Message as LiteLLMMessage,
            ModelResponse,
            Usage,
        )

        mock_response = ModelResponse(
            id="test-id",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(content="Test response", role="assistant"),
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_completion.return_value = mock_response

        llm = LLM(
            model="gpt-4o-mini",
            api_key=SecretStr("test_key"),
            usage_id="test-llm",
            timeout=None,  # Explicitly set to None
        )

        messages = [Message(role="user", content=[TextContent(text="Hello")])]
        llm.completion(messages=messages)

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["timeout"] is None


@asynccontextmanager
async def _hung_chat_completion_server(
    *, keep_sending: bool = False
) -> AsyncIterator[tuple[str, list[int]]]:
    attempts: list[int] = []

    async def completion(request: web.Request) -> web.StreamResponse:
        await request.json()
        attempts.append(len(attempts) + 1)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        chunk = {
            "id": f"chatcmpl-{attempts[-1]}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}],
        }
        while True:
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if not keep_sending:
                await asyncio.Event().wait()
            await asyncio.sleep(0.01)

    app = web.Application()
    app.router.add_post("/chat/completions", completion)
    runner = web.AppRunner(app, shutdown_timeout=0.01)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    address = runner.addresses[0]
    assert isinstance(address, tuple)
    port = address[1]
    assert isinstance(port, int)
    try:
        yield f"http://127.0.0.1:{port}", attempts
    finally:
        await runner.cleanup()


def _streaming_llm(base_url: str, **kwargs) -> LLM:
    return LLM(
        model="openai/test-model",
        api_key=SecretStr("test-key"),
        base_url=base_url,
        usage_id="timeout-test",
        stream=True,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=0,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_async_stream_idle_timeout_retries_real_http_stream() -> None:
    async with _hung_chat_completion_server() as (base_url, attempts):
        # The hard timeout needs real headroom: the idle-trigger path takes a
        # couple of seconds here (two attempts to first chunk), so a tight
        # `timeout` would let the outer hard timeout win under CPU contention
        # and the error would read "hard timeout" instead of "idle timeout".
        llm = _streaming_llm(
            base_url,
            timeout=30,
            stream_idle_timeout=0.05,
            num_retries=2,
        )

        with pytest.raises(LLMTimeoutError, match="idle timeout"):
            await llm.acompletion(
                messages=[Message(role="user", content=[TextContent(text="Hello")])],
                on_token=lambda _: None,
            )

    assert len(attempts) == 2


class _TransportTimeoutStream:
    """Async iterator whose first chunk raises a transport-level TimeoutError."""

    def __init__(self, message: str = "transport read timed out") -> None:
        self.message = message

    def __aiter__(self) -> "_TransportTimeoutStream":
        return self

    async def __anext__(self):
        raise TimeoutError(self.message)


@pytest.mark.asyncio
async def test_idle_timeout_disabled_preserves_transport_timeout() -> None:
    """With `stream_idle_timeout=None` a real TimeoutError must pass through."""
    llm = LLM(
        model="gpt-4o-mini",
        usage_id="test-llm",
        timeout=300,
        stream_idle_timeout=None,
    )

    with pytest.raises(TimeoutError, match="transport read timed out") as exc_info:
        async for _ in llm._aiter_with_idle_timeout(_TransportTimeoutStream()):
            pass

    assert not isinstance(exc_info.value, LiteLLMTimeout)


@pytest.mark.asyncio
async def test_transport_timeout_is_not_relabelled_as_idle_timeout() -> None:
    """An enabled idle timer must not claim a transport TimeoutError."""
    llm = LLM(
        model="gpt-4o-mini",
        usage_id="test-llm",
        timeout=300,
        stream_idle_timeout=5,
    )

    with pytest.raises(TimeoutError, match="transport read timed out") as exc_info:
        async for _ in llm._aiter_with_idle_timeout(_TransportTimeoutStream()):
            pass

    assert not isinstance(exc_info.value, LiteLLMTimeout)
    assert "idle timeout" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_idle_timeout_still_fires_for_stalled_stream() -> None:
    """The idle timer itself still raises the idle-timeout error."""
    llm = LLM(
        model="gpt-4o-mini",
        usage_id="test-llm",
        timeout=300,
        stream_idle_timeout=0.01,
    )

    async def stalled() -> AsyncIterator[None]:
        await asyncio.Event().wait()
        yield None

    with pytest.raises(LiteLLMTimeout, match="stream idle timeout after 0.01"):
        async for _ in llm._aiter_with_idle_timeout(stalled()):
            pass


@pytest.mark.asyncio
async def test_hard_timeout_keeps_attribution_over_idle_timeout() -> None:
    """An outer hard timeout must not be reported as an idle timeout."""
    llm = LLM(
        model="gpt-4o-mini",
        usage_id="test-llm",
        timeout=1,
        stream_idle_timeout=30,
    )

    async def stalled() -> AsyncIterator[None]:
        await asyncio.Event().wait()
        yield None

    async def consume() -> None:
        async for _ in llm._aiter_with_idle_timeout(stalled()):
            pass

    wrapped = llm._async_hard_timeout_decorator()(consume)
    with pytest.raises(LiteLLMTimeout, match="hard timeout after 1 seconds"):
        await wrapped()


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_timeout", [None, 30])
async def test_transport_timeout_survives_both_timeout_layers(
    idle_timeout: float | None,
) -> None:
    """A transport TimeoutError reaches the caller unchanged, with either setting."""
    llm = LLM(
        model="gpt-4o-mini",
        usage_id="test-llm",
        timeout=300,
        stream_idle_timeout=idle_timeout,
    )

    async def consume() -> None:
        async for _ in llm._aiter_with_idle_timeout(_TransportTimeoutStream()):
            pass

    wrapped = llm._async_hard_timeout_decorator()(consume)
    with pytest.raises(TimeoutError, match="transport read timed out") as exc_info:
        await wrapped()

    assert type(exc_info.value) is TimeoutError


@pytest.mark.asyncio
async def test_async_hard_timeout_expires_active_attempt() -> None:
    llm = _streaming_llm("http://127.0.0.1", timeout=0, num_retries=1)

    async def never_finishes() -> None:
        await asyncio.Event().wait()

    wrapped = llm._async_hard_timeout_decorator()(never_finishes)
    with pytest.raises(LiteLLMTimeout, match="hard timeout"):
        await wrapped()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hard_timeout_max_attempts", "num_retries", "expected_attempts"),
    [(2, 5, 2), (4, 5, 4), (4, 2, 2), (2, 0, 1)],
)
async def test_hard_timeout_attempt_limit(
    hard_timeout_max_attempts: int,
    num_retries: int,
    expected_attempts: int,
    caplog,
) -> None:
    attempts = 0
    retry_updates: list[tuple[int, int]] = []

    async def never_responds(**_kwargs):
        nonlocal attempts
        attempts += 1
        await asyncio.Event().wait()

    llm = LLM(
        model="openai/test-model",
        api_key=SecretStr("test-key"),
        usage_id="timeout-test",
        timeout=0,
        num_retries=num_retries,
        hard_timeout_max_attempts=hard_timeout_max_attempts,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=0,
        retry_listener=lambda attempt, total, _error: retry_updates.append(
            (attempt, total)
        ),
    )
    with (
        patch("openhands.sdk.llm.llm.litellm_acompletion", side_effect=never_responds),
        patch("openhands.sdk.llm.llm.logger") as timeout_logger,
    ):
        with pytest.raises(LLMTimeoutError, match="shorter or chunked"):
            await llm.acompletion(
                messages=[Message(role="user", content=[TextContent(text="Hello")])]
            )

    assert attempts == expected_attempts
    assert retry_updates == [
        (attempt, expected_attempts) for attempt in range(1, expected_attempts)
    ]
    for attempt in range(1, expected_attempts):
        assert f"Attempt #{attempt}/{expected_attempts}" in caplog.text
    terminal_log = timeout_logger.error.call_args
    assert terminal_log is not None
    assert f"Attempt #{expected_attempts}/{expected_attempts}" in (
        terminal_log.args[0] % terminal_log.args[1:]
    )
    assert "elapsed" in caplog.text or "elapsed" in terminal_log.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunk_content", "expected_attempts"),
    [("partial output", 3), (None, 2)],
)
async def test_hard_timeout_distinguishes_stream_output_from_metadata(
    chunk_content: str | None, expected_attempts: int
) -> None:
    attempts = 0
    emitted: list[ModelResponseStream] = []

    async def streams_then_hangs(**_kwargs):
        nonlocal attempts
        attempts += 1

        async def chunks():
            yield ModelResponseStream(
                id=f"chunk-{attempts}",
                choices=[StreamingChoices(delta=Delta(content=chunk_content))],
            )
            await asyncio.Event().wait()

        return chunks()

    llm = LLM(
        model="openai/test-model",
        api_key=SecretStr("test-key"),
        usage_id="timeout-test",
        stream=True,
        timeout=1,
        stream_idle_timeout=30,
        num_retries=3,
        hard_timeout_max_attempts=2,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=0,
    )
    with patch(
        "openhands.sdk.llm.llm.litellm_acompletion", side_effect=streams_then_hangs
    ):
        with pytest.raises(LLMTimeoutError, match="hard timeout") as exc_info:
            await llm.acompletion(
                messages=[Message(role="user", content=[TextContent(text="Hello")])],
                on_token=emitted.append,
            )

    assert attempts == expected_attempts
    assert len(emitted) == expected_attempts
    assert ("shorter or chunked" in str(exc_info.value)) is (chunk_content is None)


@pytest.mark.asyncio
async def test_aresponses_hard_timeout_uses_same_attempt_limit() -> None:
    attempts = 0

    async def never_responds(**_kwargs):
        nonlocal attempts
        attempts += 1
        await asyncio.Event().wait()

    llm = LLM(
        model="openai/test-model",
        api_key=SecretStr("test-key"),
        usage_id="timeout-test",
        timeout=0,
        num_retries=5,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=0,
    )
    with patch("openhands.sdk.llm.llm.litellm_aresponses", side_effect=never_responds):
        with pytest.raises(LLMTimeoutError, match="shorter or chunked"):
            await llm.aresponses(
                messages=[Message(role="user", content=[TextContent(text="Hello")])]
            )

    assert attempts == 2


@pytest.mark.asyncio
async def test_provider_timeout_keeps_regular_retry_limit() -> None:
    attempts = 0

    async def provider_times_out(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise LiteLLMTimeout(
            message="provider timeout", model="test-model", llm_provider="openai"
        )

    llm = LLM(
        model="openai/test-model",
        api_key=SecretStr("test-key"),
        usage_id="timeout-test",
        timeout=30,
        num_retries=4,
        hard_timeout_max_attempts=2,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=0,
    )
    with patch(
        "openhands.sdk.llm.llm.litellm_acompletion", side_effect=provider_times_out
    ):
        with pytest.raises(LLMTimeoutError, match="provider timeout"):
            await llm.acompletion(
                messages=[Message(role="user", content=[TextContent(text="Hello")])]
            )

    assert attempts == 4


class _TimedOutLLM(LLM):
    async def acompletion(self, *args, **kwargs):
        raise LLMTimeoutError("stream idle timeout")


@pytest.mark.asyncio
async def test_exhausted_stream_timeout_transitions_conversation_to_error(
    tmp_path,
) -> None:
    conversation = LocalConversation(
        agent=Agent(llm=_TimedOutLLM(model="test-model"), tools=[]),
        workspace=str(tmp_path),
        visualizer=None,
    )
    conversation.send_message("Hello")

    with pytest.raises(ConversationRunError):
        await conversation.arun()

    assert conversation.state.execution_status == ConversationExecutionStatus.ERROR
