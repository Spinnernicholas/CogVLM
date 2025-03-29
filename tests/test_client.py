import pytest
import logging
import time
import os # For file path test cleanup if needed
from typing import List, Tuple
from PIL import Image
from collections.abc import Iterator, Callable # Import Callable

# Import the client class
from cogvlm.client import CogVLMClient

# Import fixtures from test_server indirectly via pytest's discovery
# We need server_base_url and uvicorn_server (which depends on shared_model)

_logger = logging.getLogger(__name__)
# Configure logging level if needed for debugging tests
# logging.basicConfig(level=logging.DEBUG)

# --- Helper Functions ---

def create_dummy_image(size=(64, 64), color="blue") -> Image.Image:
    """Creates a simple PIL Image for testing."""
    return Image.new("RGB", size, color=color)

# --- Fixtures ---

@pytest.fixture(scope="module") # Scope can be module as client is stateless
def client(server_base_url, uvicorn_server) -> CogVLMClient:
    """
    Provides an initialized CogVLMClient instance pointing to the test server.
    Depends on the server being up.
    """
    # Ensure uvicorn_server fixture runs first and server is ready
    _ = uvicorn_server
    # Add a small delay to ensure server is fully up after fixture setup
    time.sleep(1)
    _logger.info(f"Creating client for server at {server_base_url}")
    client_instance = CogVLMClient(base_url=server_base_url)
    # Perform a quick health check within the fixture to catch early issues
    health = client_instance.check_health()
    assert health.get("status") == "ok", f"Server health check failed in fixture: {health}"
    assert health.get("model_loaded") is True, "Model not loaded according to health check in fixture"
    _logger.info("Client created and server health checked successfully.")
    return client_instance

# --- Test Functions ---

def test_client_initialization(server_base_url):
    """Tests basic client initialization."""
    print("\nRunning test_client_initialization...")
    client = CogVLMClient(base_url=server_base_url)
    assert client.base_url == server_base_url
    assert client.inference_url == f"{server_base_url}/inference"
    assert client.stream_url == f"{server_base_url}/inference/stream"
    assert client.health_url == f"{server_base_url}/health"
    print("Client initialization test passed.")

def test_client_check_health_success(client: CogVLMClient):
    """Tests the health check against a running server."""
    print("\nRunning test_client_check_health_success...")
    health_status = client.check_health()
    assert isinstance(health_status, dict)
    assert health_status.get("status") == "ok"
    assert health_status.get("model_loaded") is True
    assert "model_info" in health_status # Check if model info is present
    print(f"Client health check success test passed. Status: {health_status}")

# Note: Testing health check *failure* (e.g., connection error) typically
# requires mocking 'requests.Session.get' or stopping the server, which
# complicates fixture management. Skipping direct failure test here.

def test_client_standard_inference_simple(client: CogVLMClient):
    """Tests a basic standard inference call (text-only)."""
    print("\nRunning test_client_standard_inference_simple...")
    query = "What is 2 + 2?"
    response_text, history = client.inference(query=query, max_new_tokens=10)

    assert isinstance(response_text, str)
    assert len(response_text) > 0
    # Basic check for the answer (model dependent, adjust if needed)
    # The dummy model in test_server returns a fixed response

    assert isinstance(history, list)
    assert len(history) == 2 # User query + Assistant response
    assert history[0] == ("USER", query)
    assert history[1][0] == "ASSISTANT"
    assert history[1][1] == response_text
    print(f"Client standard inference (simple) test passed. Response: '{response_text[:50]}...'")

def test_client_standard_inference_with_history(client: CogVLMClient):
    """Tests standard inference with pre-existing history."""
    print("\nRunning test_client_standard_inference_with_history...")
    initial_history: List[Tuple[str, str]] = [
        ("USER", "My favorite color is blue."),
        ("ASSISTANT", "That's a nice color.")
    ]
    query = "What was my favorite color?"
    response_text, history = client.inference(
        query=query,
        history=initial_history,
        max_new_tokens=20
    )

    assert isinstance(response_text, str)
    # The dummy model doesn't use history, so we just check format

    assert isinstance(history, list)
    assert len(history) == 4 # Initial 2 turns + new query/response
    assert history[0] == initial_history[0]
    assert history[1] == initial_history[1]
    assert history[2] == ("USER", query)
    assert history[3] == ("ASSISTANT", response_text)
    print("Client standard inference (with history) test passed.")

def test_client_standard_inference_with_pil_image(client: CogVLMClient):
    """Tests standard inference passing a PIL Image."""
    print("\nRunning test_client_standard_inference_with_pil_image...")
    query = "Describe the color of this image."
    dummy_image = create_dummy_image(color="red")

    response_text, history = client.inference(
        query=query,
        images=[dummy_image], # Pass the PIL Image object
        max_new_tokens=20
    )

    assert isinstance(response_text, str)

    assert isinstance(history, list)
    assert len(history) == 2
    assert history[0] == ("USER", query)
    assert history[1] == ("ASSISTANT", response_text)
    print("Client standard inference (with PIL image) test passed.")

def test_client_standard_inference_with_image_path(client: CogVLMClient, tmp_path):
    """Tests standard inference passing an image file path."""
    print("\nRunning test_client_standard_inference_with_image_path...")
    query = "Describe the color of the image file."
    dummy_image = create_dummy_image(color="green")
    image_path = tmp_path / "test_image_client.png" # Use unique name
    dummy_image.save(image_path)
    print(f"Saved dummy image to: {image_path}")

    response_text, history = client.inference(
        query=query,
        images=[str(image_path)], # Pass the file path as string
        max_new_tokens=20
    )

    assert isinstance(response_text, str)

    assert isinstance(history, list)
    assert len(history) == 2
    print("Client standard inference (with image path) test passed.")
    # Optional: Clean up the temporary file if needed, though tmp_path handles it
    # os.remove(image_path)

def test_client_streaming_inference(client: CogVLMClient):
    """Tests streaming inference, checking iterator and finalizer."""
    print("\nRunning test_client_streaming_inference...")
    query = "Count to three."
    max_tokens = 15

    # Get the iterator and finalizer
    result = client.inference(query=query, max_new_tokens=max_tokens, stream=True)
    assert isinstance(result, tuple) and len(result) == 2
    token_iterator, finalizer = result
    assert isinstance(token_iterator, Iterator), "First element should be an iterator"
    assert isinstance(finalizer, Callable), "Second element should be a callable finalizer"

    # Consume the iterator
    tokens_received: List[str] = []
    streamed_response = ""
    print("Streaming tokens: ", end="")
    for token in token_iterator:
        assert isinstance(token, str)
        print(token, end="", flush=True)
        tokens_received.append(token)
        streamed_response += token
    print("\nStreaming finished.")

    assert len(tokens_received) > 0, "No tokens were received from iterator"

    # Call the finalizer
    print("Calling finalizer...")
    final_response, final_history = finalizer()
    print("Finalizer returned.")

    assert isinstance(final_response, str)
    assert isinstance(final_history, list)

    # Check finalizer results
    # Dummy model's final response should match streamed content
    assert final_response == streamed_response

    assert len(final_history) == 2 # User query + Assistant response
    assert final_history[0] == ("USER", query)
    assert final_history[1] == ("ASSISTANT", final_response)

    print("Client streaming inference (iterator and finalizer) test passed.")


def test_client_streaming_inference_seed(client: CogVLMClient):
    """Tests streaming inference with a seed response."""
    print("\nRunning test_client_streaming_inference_seed...")
    query = "Continue this sequence: A, B,"
    # Seed ends mid-thought to encourage continuation
    seed = "Okay, the next letter is "
    max_tokens = 10

    token_iterator, finalizer = client.inference(
        query=query,
        seed_response=seed,
        max_new_tokens=max_tokens,
        stream=True
    )

    assert isinstance(token_iterator, Iterator)
    assert isinstance(finalizer, Callable)

    # Consume the iterator
    tokens_received: List[str] = []
    streamed_response = ""
    print("Streaming tokens: ", end="")
    for token in token_iterator:
        assert isinstance(token, str)
        print(token, end="", flush=True)
        tokens_received.append(token)
        streamed_response += token
    print("\nStreaming finished.")

    assert len(tokens_received) > 0, "No tokens received"

    # Check if the streamed response starts with the seed
    # The dummy server yields the seed first, then the rest
    assert streamed_response.startswith(seed), f"Streamed response did not start with seed. Got: '{streamed_response[:len(seed)+10]}...'"
    # Check if more than just the seed was generated (dummy adds more)
    assert len(streamed_response.strip()) > len(seed.strip()), "Only the seed was generated"

    # Call the finalizer
    print("Calling finalizer...")
    final_response, final_history = finalizer()
    print("Finalizer returned.")

    assert isinstance(final_response, str)
    assert isinstance(final_history, list)

    # Check finalizer results
    assert final_response == streamed_response # Should match concatenated stream
    assert final_response.startswith(seed) # Final response should also start with seed

    assert len(final_history) == 2
    assert final_history[0] == ("USER", query)
    assert final_history[1] == ("ASSISTANT", final_response) # History uses the full response

    print("Client streaming inference (seed) test passed.")

def test_client_inference_invalid_image_type(client: CogVLMClient):
    """Tests that the client raises TypeError for invalid image types."""
    print("\nRunning test_client_inference_invalid_image_type...")
    query = "This should fail."
    invalid_image = 12345 # Not a str or PIL Image

    with pytest.raises(TypeError, match="Unsupported image type"):
        client.inference(query=query, images=[invalid_image])
    print("Client inference with invalid image type test passed.")

def test_client_inference_image_path_not_found(client: CogVLMClient):
    """Tests that the client raises FileNotFoundError for non-existent paths."""
    print("\nRunning test_client_inference_image_path_not_found...")
    query = "Describe non-existent image."
    non_existent_path = "path/to/absolutely/nothing.jpg"

    with pytest.raises(FileNotFoundError, match="Client could not find local image file"):
        client.inference(query=query, images=[non_existent_path])
    print("Client inference with non-existent image path test passed.")
