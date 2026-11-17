# Copyright (c) Microsoft. All rights reserved.

"""Tests for AGUIHttpService."""

import copy
import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from ag_ui.core import Interrupt, ResumeEntry, RunAgentInput
from agent_framework import Message, WorkflowBuilder, WorkflowContext, executor
from fastapi import FastAPI

from agent_framework_ag_ui import add_agent_framework_fastapi_endpoint
from agent_framework_ag_ui._http_service import AGUIHttpService


@pytest.fixture
def mock_http_client():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
def sample_events():
    """Sample AG-UI events for testing."""
    return [
        {"type": "RUN_STARTED", "threadId": "thread_123", "runId": "run_456"},
        {"type": "TEXT_MESSAGE_START", "messageId": "msg_1", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": "Hello"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": " world"},
        {"type": "TEXT_MESSAGE_END", "messageId": "msg_1"},
        {"type": "RUN_FINISHED", "threadId": "thread_123", "runId": "run_456"},
    ]


def create_sse_response(events: list[dict]) -> str:
    """Create SSE formatted response from events."""
    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}\n")
    return "\n".join(lines)


async def test_http_service_initialization():
    """Test AGUIHttpService initialization."""
    # Test with default client
    service = AGUIHttpService("http://localhost:8888/")
    assert service.endpoint == "http://localhost:8888"
    assert service._owns_client is True
    assert isinstance(service.http_client, httpx.AsyncClient)
    await service.close()

    # Test with custom client
    custom_client = httpx.AsyncClient()
    service = AGUIHttpService("http://localhost:8888/", http_client=custom_client)
    assert service._owns_client is False
    assert service.http_client is custom_client
    # Shouldn't close the custom client
    await service.close()
    await custom_client.aclose()


async def test_http_service_strips_trailing_slash():
    """Test that endpoint trailing slash is stripped."""
    service = AGUIHttpService("http://localhost:8888/")
    assert service.endpoint == "http://localhost:8888"
    await service.close()


async def test_post_run_successful_streaming(mock_http_client, sample_events):
    """Test successful streaming of events."""

    # Create async generator for lines
    async def mock_aiter_lines():
        sse_data = create_sse_response(sample_events)
        for line in sse_data.split("\n"):
            if line:
                yield line

    # Create mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    # aiter_lines is called as a method, so it should return a new generator each time
    mock_response.aiter_lines = mock_aiter_lines

    # Setup mock streaming context manager
    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    mock_stream_context.__aexit__.return_value = None
    mock_http_client.stream.return_value = mock_stream_context

    service = AGUIHttpService("http://localhost:8888/", http_client=mock_http_client)

    events = []
    async for event in service.post_run(
        thread_id="thread_123", run_id="run_456", messages=[{"role": "user", "content": "Hello"}]
    ):
        events.append(event)

    assert len(events) == len(sample_events)
    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1]["type"] == "RUN_FINISHED"

    # Verify request was made correctly
    mock_http_client.stream.assert_called_once()
    call_args = mock_http_client.stream.call_args
    assert call_args.args[0] == "POST"
    assert call_args.args[1] == "http://localhost:8888"
    assert call_args.kwargs["headers"] == {"Accept": "text/event-stream"}


async def test_post_run_with_state_tools_and_interrupts(mock_http_client):
    """Test posting run with state, tools, and interrupt metadata."""

    async def mock_aiter_lines():
        return
        yield  # Make it an async generator

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines

    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    mock_stream_context.__aexit__.return_value = None
    mock_http_client.stream.return_value = mock_stream_context

    service = AGUIHttpService("http://localhost:8888/", http_client=mock_http_client)

    state = {"user_context": {"name": "Alice"}}
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    available_interrupts = [{"id": "req_1", "type": "request_info"}]
    expected_available_interrupts = [{"id": "req_1", "reason": "input_required"}]
    resume = {"interrupts": [{"id": "req_1", "value": "approved"}]}
    expected_resume = [{"interruptId": "req_1", "status": "resolved", "payload": "approved"}]

    async for _ in service.post_run(
        thread_id="thread_123",
        run_id="run_456",
        messages=[],
        state=state,
        tools=tools,
        available_interrupts=available_interrupts,
        resume=resume,
    ):
        pass

    # Verify state and tools were included in request
    call_args = mock_http_client.stream.call_args
    request_data = call_args.kwargs["json"]
    assert request_data["state"] == state
    assert request_data["tools"] == tools
    assert request_data["availableInterrupts"] == expected_available_interrupts
    assert request_data["resume"] == expected_resume


async def test_post_run_serializes_typed_interrupts_and_resume_with_protocol_aliases(mock_http_client):
    """Typed protocol interrupt and resume models are serialized to canonical wire fields."""

    async def mock_aiter_lines():
        return
        yield  # Make it an async generator

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines

    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    mock_stream_context.__aexit__.return_value = None
    mock_http_client.stream.return_value = mock_stream_context

    service = AGUIHttpService("http://localhost:8888/", http_client=mock_http_client)

    async for _ in service.post_run(
        thread_id="thread_123",
        run_id="run_456",
        messages=[],
        available_interrupts=[
            Interrupt(
                id="approval_1",
                reason="tool_call",
                tool_call_id="call_1",
                response_schema={"type": "object"},
            )
        ],
        resume=[ResumeEntry(interrupt_id="approval_1", status="resolved", payload={"approved": True})],
    ):
        pass

    request_data = mock_http_client.stream.call_args.kwargs["json"]
    assert request_data["availableInterrupts"] == [
        {
            "id": "approval_1",
            "reason": "tool_call",
            "toolCallId": "call_1",
            "responseSchema": {"type": "object"},
        }
    ]
    assert request_data["resume"] == [
        {"interruptId": "approval_1", "status": "resolved", "payload": {"approved": True}}
    ]


async def test_post_run_serializes_legacy_single_resume_mapping_as_canonical_list(mock_http_client):
    """Legacy single-entry resume mappings are sent as canonical ResumeEntry arrays."""

    async def mock_aiter_lines():
        return
        yield  # Make it an async generator

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines

    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    mock_stream_context.__aexit__.return_value = None
    mock_http_client.stream.return_value = mock_stream_context

    service = AGUIHttpService("http://localhost:8888/", http_client=mock_http_client)

    async for _ in service.post_run(
        thread_id="thread_123",
        run_id="run_456",
        messages=[],
        resume={"id": "approval_1", "value": {"approved": True}},
    ):
        pass

    request_data = mock_http_client.stream.call_args.kwargs["json"]
    assert request_data["resume"] == [
        {"interruptId": "approval_1", "status": "resolved", "payload": {"approved": True}}
    ]


async def test_post_run_serializes_legacy_interrupt_without_type(mock_http_client):
    """Legacy interrupt metadata without reason/type does not crash client serialization."""

    async def mock_aiter_lines():
        return
        yield  # Make it an async generator

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines

    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    mock_stream_context.__aexit__.return_value = None
    mock_http_client.stream.return_value = mock_stream_context

    service = AGUIHttpService("http://localhost:8888/", http_client=mock_http_client)

    async for _ in service.post_run(
        thread_id="thread_123",
        run_id="run_456",
        messages=[],
        available_interrupts=[{"id": "req_1", "value": {"prompt": "Choose"}}],
    ):
        pass

    request_data = mock_http_client.stream.call_args.kwargs["json"]
    assert request_data["availableInterrupts"][0]["id"] == "req_1"
    assert request_data["availableInterrupts"][0]["reason"] == "input_required"


async def test_post_run_http_error(mock_http_client):
    """Test handling of HTTP errors."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    def raise_http_error():
        raise httpx.HTTPStatusError("Server error", request=Mock(), response=mock_response)

    mock_response_async = AsyncMock()
    mock_response_async.raise_for_status = raise_http_error

    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response_async
    mock_stream_context.__aexit__.return_value = None
    mock_http_client.stream.return_value = mock_stream_context

    service = AGUIHttpService("http://localhost:8888/", http_client=mock_http_client)

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in service.post_run(thread_id="thread_123", run_id="run_456", messages=[]):
            pass


async def test_post_run_invalid_json(mock_http_client):
    """Test handling of invalid JSON in SSE stream."""
    invalid_sse = "data: {invalid json}\n\ndata: " + json.dumps({"type": "RUN_FINISHED"}) + "\n"

    async def mock_aiter_lines():
        for line in invalid_sse.split("\n"):
            if line:
                yield line

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines

    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    mock_stream_context.__aexit__.return_value = None
    mock_http_client.stream.return_value = mock_stream_context

    service = AGUIHttpService("http://localhost:8888/", http_client=mock_http_client)

    events = []
    async for event in service.post_run(thread_id="thread_123", run_id="run_456", messages=[]):
        events.append(event)

    # Should skip invalid JSON and continue with valid events
    assert len(events) == 1
    assert events[0]["type"] == "RUN_FINISHED"


async def test_post_run_accepts_data_lines_without_space(sample_events):
    """The space after "data:" is optional in SSE, so "data:{...}" lines must be parsed too."""
    body = "".join(f"data:{json.dumps(event)}\n\n" for event in sample_events)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = AGUIHttpService("http://localhost:8888/", http_client=client)
        events = [event async for event in service.post_run(thread_id="thread_123", run_id="run_456", messages=[])]

    assert events == sample_events


@pytest.mark.parametrize(
    "state", [None, {}, {"counter": 1, "nested": {"value": None}}], ids=["no-state", "empty-state", "state"]
)
@pytest.mark.parametrize(
    "tools",
    [None, [], [{"name": "lookup", "description": "Look up an item", "parameters": {"type": "object"}}]],
    ids=["unspecified-tools", "empty-tools", "tool"],
)
async def test_post_run_sends_required_run_input_defaults(state, tools):
    """Default calls must validate as official run inputs without losing supplied values."""
    received: list[RunAgentInput] = []
    expected_state = copy.deepcopy(state)
    expected_tools = copy.deepcopy(tools)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        validated = RunAgentInput.model_validate(payload)
        assert "state" in payload
        received.append(validated)
        assert validated.thread_id == "thread-1" and validated.run_id == "run-1"
        assert validated.messages == []
        assert validated.state == expected_state
        assert [tool.model_dump() for tool in validated.tools] == (expected_tools if expected_tools is not None else [])
        assert validated.context == [] and validated.forwarded_props is None
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"type":"RUN_FINISHED","threadId":"thread-1","runId":"run-1"}\n\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = AGUIHttpService("https://agui.example.test/run", http_client=client)
        events = [
            event
            async for event in service.post_run(
                thread_id="thread-1", run_id="run-1", messages=[], state=state, tools=tools
            )
        ]

    assert len(received) == 1
    assert state == expected_state and tools == expected_tools
    assert events == [{"type": "RUN_FINISHED", "threadId": "thread-1", "runId": "run-1"}]


async def test_post_run_preserves_workflow_invocation_defaults():
    """Default requests must not inject empty forwarded properties into workflow execution."""

    @executor(id="echo")
    async def echo(messages: list[Message], ctx: WorkflowContext[None, str]) -> None:
        await ctx.yield_output(messages[-1].text)

    workflow = WorkflowBuilder(start_executor=echo).build()
    app = FastAPI()
    add_agent_framework_fastapi_endpoint(app, workflow, path="/run", single_user=True, keepalive_seconds=None)

    with patch.object(workflow, "run", wraps=workflow.run) as run:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            service = AGUIHttpService("https://agui.example.test/run", http_client=client)
            events = [
                event
                async for event in service.post_run(
                    thread_id="thread-1",
                    run_id="run-1",
                    messages=[{"id": "m1", "role": "user", "content": "Hello"}],
                )
            ]

    assert events[-1]["type"] == "RUN_FINISHED"
    assert not any(event["type"] == "RUN_ERROR" for event in events)
    run.assert_called_once()
    assert "function_invocation_kwargs" not in run.call_args.kwargs


async def test_context_manager():
    """Test context manager functionality."""
    async with AGUIHttpService("http://localhost:8888/") as service:
        assert service.http_client is not None
        assert service._owns_client is True

    # Client should be closed after exiting context


async def test_context_manager_with_external_client():
    """Test context manager doesn't close external client."""
    external_client = httpx.AsyncClient()

    async with AGUIHttpService("http://localhost:8888/", http_client=external_client) as service:
        assert service.http_client is external_client
        assert service._owns_client is False

    # External client should still be open
    # (caller's responsibility to close)
    await external_client.aclose()


async def test_post_run_empty_response(mock_http_client):
    """Test handling of empty response stream."""

    async def mock_aiter_lines():
        return
        yield  # Make it an async generator

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines

    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    mock_stream_context.__aexit__.return_value = None
    mock_http_client.stream.return_value = mock_stream_context

    service = AGUIHttpService("http://localhost:8888/", http_client=mock_http_client)

    events = []
    async for event in service.post_run(thread_id="thread_123", run_id="run_456", messages=[]):
        events.append(event)

    assert len(events) == 0
