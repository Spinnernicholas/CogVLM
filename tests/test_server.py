import pytest
import httpx
import asyncio
import threading
import uvicorn
import time
import json
import logging

# Assuming server code is importable from cogvlm.server
try:
    from cogvlm.server import (
        create_app,
        InferenceRequest,
        InferenceResponse,
        StreamChunk,
    )
except ImportError as e:
    # If server code isn't found, tests will fail later, but provide a clear message
    pytest.fail(f"Failed to import server components: {e}", pytrace=False)

# Import TEST_MODEL from conftest.py for assertions
# Ensure conftest.py is in the same directory or adjust path
try:
    # Assuming conftest.py is in the same directory (e.g., tests/)
    from .conftest import TEST_MODEL
except ImportError:
    # Fallback or define directly if conftest structure differs
    # Make sure this matches the model used in conftest.py
    TEST_MODEL = "THUDM/cogvlm2-llama3-chat-19B"
    logging.warning(f"Could not import TEST_MODEL from conftest, using default: {TEST_MODEL}")

_logger = logging.getLogger(__name__)
# Configure logging level if needed (e.g., logging.INFO)
# logging.basicConfig(level=logging.INFO)

# --- Test Functions ---

@pytest.mark.asyncio
async def test_health_check(api_client: httpx.AsyncClient):
    """Tests the /health endpoint."""
    print("\nRunning test_health_check...")
    response = await api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    # Check model_info reflects pre-initialization or the correct path
    assert "Pre-initialized" in data.get("model_info", "") or TEST_MODEL in data.get("model_info", "")
    print("Health check passed.")


@pytest.mark.asyncio
async def test_server_standard_inference(api_client: httpx.AsyncClient):
    """Tests standard (non-streaming) inference via the /inference endpoint."""
    print("\nRunning test_server_standard_inference...")
    query = "What is the capital of France?"
    payload = InferenceRequest(query=query, stream=False).model_dump()

    try:
        response = await api_client.post("/inference", json=payload)
        response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
    except httpx.HTTPStatusError as e:
        pytest.fail(f"HTTP error during standard inference: {e.response.status_code} - {e.response.text}", pytrace=False)
    except Exception as e:
         pytest.fail(f"Unexpected error during standard inference request: {e}", pytrace=False)

    data = response.json()
    assert isinstance(data, dict)
    assert "response" in data and isinstance(data["response"], str) and len(data["response"]) > 0
    assert "history" in data and isinstance(data["history"], list) and len(data["history"]) == 2
    assert data.get("error") is None
    assert data["history"][0]["role"] == "USER" and data["history"][0]["content"] == query
    assert data["history"][1]["role"] == "ASSISTANT" and data["history"][1]["content"] == data["response"]
    print(f"Standard inference response (first 50 chars): '{data['response'][:50]}...'")
    print("Standard server inference test passed.")


@pytest.mark.asyncio
async def test_server_streaming_endpoint_basic(api_client: httpx.AsyncClient):
    """Tests basic interaction with the /inference/stream endpoint."""
    print("\nRunning test_server_streaming_endpoint_basic...")
    query = "Describe the color blue in one word."
    # Keep max_tokens low for a quick test
    payload = InferenceRequest(query=query, stream=True, max_new_tokens=5).model_dump()

    full_response_text = ""
    last_chunk_data = None
    chunks_received = 0

    try:
        async with api_client.stream("POST", "/inference/stream", json=payload) as response:
            response.raise_for_status() # Check for HTTP errors immediately
            assert "application/x-ndjson" in response.headers.get("content-type", "")
            # Iterate through the stream - this consumes it
            async for line in response.aiter_lines():
                if line:
                    chunks_received += 1
                    chunk_data = json.loads(line)
                    last_chunk_data = chunk_data # Store the latest chunk received
                    StreamChunk.model_validate(chunk_data) # Validate structure
                    # Accumulate text only from non-final chunks
                    if not chunk_data.get("finished", False):
                        full_response_text += chunk_data["token"]
            # REMOVED: await response.aread() - async for loop consumes the stream
    except httpx.HTTPStatusError as e:
         pytest.fail(f"HTTP error during streaming: {e.response.status_code} - {e.response.text}", pytrace=False)
    except Exception as e:
        # Catch the specific error if it happens *during* iteration
        if "content has already been streamed" in str(e):
             pytest.fail(f"Streaming error during iteration: {e}", pytrace=False)
        else:
             pytest.fail(f"Unexpected error during streaming: {e}", pytrace=False)

    # Checks happen after the 'async with' block has exited and closed the connection
    assert chunks_received > 0, "No stream chunks were received"
    # Check the *last* chunk received to ensure it signaled completion
    assert last_chunk_data is not None, "Did not receive any chunks"
    assert last_chunk_data.get("finished") is True, "Last chunk received did not have finished=True"
    assert last_chunk_data.get("error") is None, f"Stream ended with error: {last_chunk_data.get('error')}"
    print("Basic streaming interaction test passed.")


@pytest.mark.asyncio
async def test_server_streaming_content_and_final_data(api_client: httpx.AsyncClient):
    """
    Tests that streamed content matches the final response data in the last chunk.
    """
    print("\nRunning test_server_streaming_content_and_final_data...")
    query = "Why is the sky blue?"
    max_tokens = 50
    payload = InferenceRequest(
        query=query,
        stream=True,
        max_new_tokens=max_tokens
    ).model_dump()

    streamed_tokens_list = []
    streamed_response_concat = ""
    last_chunk_data = None # Store the most recent chunk
    start_time = time.time()

    print("Streaming tokens:")
    try:
        async with api_client.stream("POST", "/inference/stream", json=payload) as response:
            response.raise_for_status()
            # Iterate through the stream - this consumes it
            async for line in response.aiter_lines():
                if line:
                    chunk_data = json.loads(line)
                    last_chunk_data = chunk_data # Keep track of the last received chunk
                    token = chunk_data.get("token", "")
                    finished = chunk_data.get("finished", False)
                    streamed_tokens_list.append(token)
                    # Concatenate only the actual tokens, not the empty one in the final chunk
                    if not finished:
                        streamed_response_concat += token
            # REMOVED: await response.aread()
    except httpx.HTTPStatusError as e:
         pytest.fail(f"HTTP error during streaming: {e.response.status_code} - {e.response.text}", pytrace=False)
    except Exception as e:
        if "content has already been streamed" in str(e):
             pytest.fail(f"Streaming error during iteration: {e}", pytrace=False)
        else:
             pytest.fail(f"Unexpected error during streaming: {e}", pytrace=False)

    end_time = time.time()
    print(f"\nStreaming finished in {end_time - start_time:.2f}s. Received {len(streamed_tokens_list)} chunks.")

    # Perform checks using the last received chunk
    assert last_chunk_data is not None, "No chunks were received"
    assert last_chunk_data["finished"] is True, "Last received chunk 'finished' flag is not True"
    assert last_chunk_data.get("error") is None, f"Stream ended with error: {last_chunk_data.get('error')}"

    # Extract final data from the last chunk
    final_response_from_chunk = last_chunk_data.get("response")
    final_history_from_chunk = last_chunk_data.get("history")

    assert isinstance(final_response_from_chunk, str), "Final response missing/wrong type in last chunk"
    assert isinstance(final_history_from_chunk, list), "Final history missing/wrong type in last chunk"

    # Compare concatenated stream with the final response from the last chunk
    print(f"Streamed Response ({len(streamed_response_concat)} chars): '{streamed_response_concat[:100]}...'")
    print(f"Final Response    ({len(final_response_from_chunk)} chars): '{final_response_from_chunk[:100]}...'")
    assert streamed_response_concat == final_response_from_chunk, \
        "Concatenated streamed response does not match the final response from the last chunk"

    # Check history structure from the last chunk
    assert len(final_history_from_chunk) == 2, f"History should have 2 turns, got {len(final_history_from_chunk)}"
    assert final_history_from_chunk[0]["role"] == "USER" and final_history_from_chunk[0]["content"] == query
    assert final_history_from_chunk[1]["role"] == "ASSISTANT" and final_history_from_chunk[1]["content"] == final_response_from_chunk

    # Check that some actual content was generated/streamed
    if max_tokens > 0:
        # Check if any non-empty token was received OR if final response has content
        assert any(t for t in streamed_tokens_list if t) or final_response_from_chunk, \
            "No actual content tokens were streamed or generated"
    print("Streaming content and final data test passed.")


@pytest.mark.asyncio
async def test_server_streaming_with_seed_response(api_client: httpx.AsyncClient):
    """Tests streaming via the server when a seed_response is provided."""
    print("\nRunning test_server_streaming_with_seed_response...")
    query = "Tell me a short story about a robot."
    seed = "Once upon a time, there was a small robot named Bolt who "
    max_tokens = 50
    payload = InferenceRequest(
        query=query,
        seed_response=seed,
        max_new_tokens=max_tokens,
        stream=True
    ).model_dump()

    streamed_tokens_list = []
    streamed_response_concat = ""
    last_chunk_data = None # Store the most recent chunk
    first_token_matches_seed = False

    print("Streaming tokens (with seed):")
    try:
        async with api_client.stream("POST", "/inference/stream", json=payload) as response:
            response.raise_for_status()
            # Iterate through the stream - this consumes it
            async for line in response.aiter_lines():
                if line:
                    chunk_data = json.loads(line)
                    last_chunk_data = chunk_data # Keep track of the last received chunk
                    token = chunk_data.get("token", "")
                    finished = chunk_data.get("finished", False)

                    # Check if the *first non-empty* token matches the seed
                    if not first_token_matches_seed and token:
                         assert token == seed, f"First non-empty streamed token ('{token}') did not match seed ('{seed}')"
                         first_token_matches_seed = True

                    streamed_tokens_list.append(token)
                    # Concatenate only the actual tokens, not the empty one in the final chunk
                    if not finished:
                        streamed_response_concat += token
            # REMOVED: await response.aread()
    except httpx.HTTPStatusError as e:
         pytest.fail(f"HTTP error during streaming with seed: {e.response.status_code} - {e.response.text}", pytrace=False)
    except Exception as e:
        if "content has already been streamed" in str(e):
             pytest.fail(f"Streaming error during iteration: {e}", pytrace=False)
        else:
             pytest.fail(f"Unexpected error during streaming with seed: {e}", pytrace=False)

    # Perform checks using the last received chunk
    assert first_token_matches_seed, "The seed response was not received as the first non-empty token"
    assert last_chunk_data is not None, "No chunks were received"
    assert last_chunk_data["finished"] is True, "Last received chunk 'finished' flag is not True"
    assert last_chunk_data.get("error") is None, f"Stream ended with error: {last_chunk_data.get('error')}"

    # Extract final data from the last chunk
    final_response_from_chunk = last_chunk_data.get("response")
    final_history_from_chunk = last_chunk_data.get("history")

    assert isinstance(final_response_from_chunk, str), "Final response missing/wrong type in last chunk"
    assert final_response_from_chunk.startswith(seed), \
        "Final response from chunk does not start with the seed response"

    # Compare concatenated stream with the final response
    print(f"Streamed Response ({len(streamed_response_concat)} chars): '{streamed_response_concat[:100]}...'")
    print(f"Final Response    ({len(final_response_from_chunk)} chars): '{final_response_from_chunk[:100]}...'")
    assert streamed_response_concat == final_response_from_chunk, \
        "Concatenated streamed response does not match the final response (with seed)"

    # Check history structure from the last chunk
    assert len(final_history_from_chunk) == 2, f"History should have 2 turns, got {len(final_history_from_chunk)}"
    assert final_history_from_chunk[1]["role"] == "ASSISTANT" and final_history_from_chunk[1]["content"] == final_response_from_chunk
    assert final_history_from_chunk[0]["role"] == "USER" and final_history_from_chunk[0]["content"] == query

    # Check that more than just the seed was generated/streamed
    if max_tokens > 0:
         # Check length of final response is greater than seed length
         assert len(final_response_from_chunk) > len(seed), \
            "Only the seed response seems to have been generated/streamed."

    print("Streaming with seed response test passed.")
