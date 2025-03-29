import pytest
import time
import logging
from collections.abc import Iterator, Callable # For type checking

from cogvlm import CogVLM

from .conftest import TEST_MODEL

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Configure basic logging for visibility

def test_model_initialization_via_fixture(shared_model):
    """Checks if the shared model instance from the fixture is valid."""
    print("\nRunning test_model_initialization_via_fixture...")
    assert shared_model is not None
    assert hasattr(shared_model, 'model_path')
    assert shared_model.model_path == TEST_MODEL
    print("Fixture check passed.")


def test_standard_inference(shared_model):
    """Tests standard (non-streaming) inference."""
    print("\nRunning test_standard_inference...")
    query = "What is the capital of France?"
    response, history = shared_model.inference(query=query, stream=False)

    assert isinstance(response, str)
    assert len(response) > 0, "Response should not be empty"
    assert isinstance(history, list)
    assert len(history) == 2, "History should contain one turn [(user query), (assistant response)]"
    assert history[0][0] == "USER" and history[0][1] == query
    # Note: The actual response content check is omitted as it depends on the model
    print(f"Standard inference response (first 50 chars): '{response[:50]}...'")
    print("Standard inference test passed.")


def test_streaming_output_types(shared_model):
    """Tests that stream=True returns the correct types (iterator, callable)."""
    print("\nRunning test_streaming_output_types...")
    query = "Describe the color blue."
    result = shared_model.inference(query=query, stream=True)

    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Tuple should have two elements"

    iterator, finalizer = result

    # Check if the first element is an iterator
    assert hasattr(iterator, "__iter__") and hasattr(iterator, "__next__"), \
        "First element should be an iterator"
    # More specific check using collections.abc
    assert isinstance(iterator, Iterator), "First element should be an Iterator"

    # Check if the second element is callable
    assert callable(finalizer), "Second element should be a callable function"
    assert isinstance(finalizer, Callable), "Second element should be a Callable"

    # Clean up by consuming the iterator and calling finalizer (important!)
    print("Consuming stream and finalizing to clean up...")
    _ = list(iterator) # Consume the iterator
    _ = finalizer()    # Call the finalizer
    print("Cleanup complete.")
    print("Streaming output types test passed.")


def test_streaming_content_and_finalizer(shared_model):
    """
    Tests that the streamed content matches the finalizer's response
    and the finalizer returns correct history.
    """
    print("\nRunning test_streaming_content_and_finalizer...")
    query = "Why is the sky blue?"
    # Use a small max_new_tokens for faster testing if needed
    max_tokens = 50

    iterator, finalizer = shared_model.inference(
        query=query,
        max_new_tokens=max_tokens,
        stream=True
    )

    streamed_tokens = []
    streamed_response = ""
    print("Streaming tokens:")
    try:
        for i, token in enumerate(iterator):
            # Use logger or print for debugging stream output
            # print(f"  Token {i}: '{token}'") # Uncomment for verbose token output
            assert isinstance(token, str), f"Token {i} is not a string: {type(token)}"
            streamed_tokens.append(token)
            streamed_response += token
            # time.sleep(0.01) # Optional small delay
    except Exception as e:
        pytest.fail(f"Error occurred during streaming: {e}")

    print(f"\nStreaming finished. Received {len(streamed_tokens)} tokens.")
    print("Calling finalizer...")
    # Call the finalizer to get the complete response and history
    final_response, final_history = finalizer()
    print("Finalizer finished.")

    assert isinstance(final_response, str), "Final response should be a string"
    assert isinstance(final_history, list), "Final history should be a list"

    print(f"Streamed Response ({len(streamed_response)} chars): '{streamed_response[:100]}...'")
    print(f"Final Response    ({len(final_response)} chars): '{final_response[:100]}...'")

    # Compare the concatenated streamed response with the final response
    # Allow for minor whitespace differences at the end if necessary, though ideally they match exactly
    assert streamed_response == final_response, \
        "Concatenated streamed response does not match the final response"

    # Check the history structure
    assert len(final_history) > 0, "History should not be empty"
    # Assuming default user/assistant names and no initial history
    assert final_history[-1][0] == "ASSISTANT", "Last history entry should be from ASSISTANT"
    assert final_history[-1][1] == final_response, \
        "Last history message should match the final response"
    assert final_history[-2][0] == "USER", "Second to last history entry should be from USER"
    assert final_history[-2][1] == query, "Second to last history message should match the query"

    # Check that some tokens were actually streamed (unless max_tokens was 0)
    if max_tokens > 0:
        assert len(streamed_tokens) > 0 or len(final_response) > 0, \
            "No tokens were streamed or generated"
    print("Streaming content and finalizer test passed.")


def test_streaming_with_seed_response(shared_model):
    """Tests streaming when a seed_response is provided."""
    print("\nRunning test_streaming_with_seed_response...")
    query = "Tell me a short story about a robot."
    seed = "Once upon a time, there was a small robot named Bolt who "
    max_tokens = 50

    iterator, finalizer = shared_model.inference(
        query=query,
        seed_response=seed,
        max_new_tokens=max_tokens,
        stream=True
    )

    streamed_tokens = []
    streamed_response = ""
    print("Streaming tokens (with seed):")
    first_token_received = False
    try:
        for i, token in enumerate(iterator):
            # print(f"  Token {i}: '{token}'") # Uncomment for verbose token output
            assert isinstance(token, str)
            if i == 0:
                # Check if the first yielded item is the seed response
                assert token == seed, "First streamed token should be the seed response"
                first_token_received = True
            streamed_tokens.append(token)
            streamed_response += token
    except Exception as e:
        pytest.fail(f"Error occurred during streaming with seed: {e}")

    assert first_token_received, "Iterator did not yield the seed response first"
    print(f"\nStreaming finished. Received {len(streamed_tokens)} tokens (including seed).")

    print("Calling finalizer...")
    final_response, final_history = finalizer()
    print("Finalizer finished.")

    print(f"Streamed Response ({len(streamed_response)} chars): '{streamed_response[:100]}...'")
    print(f"Final Response    ({len(final_response)} chars): '{final_response[:100]}...'")

    # Check that the final response starts with the seed
    assert final_response.startswith(seed), \
        "Final response does not start with the seed response"

    # Compare the concatenated streamed response with the final response
    assert streamed_response == final_response, \
        "Concatenated streamed response does not match the final response (with seed)"

    # Check history
    assert len(final_history) > 0
    assert final_history[-1][0] == "ASSISTANT"
    assert final_history[-1][1] == final_response # Ensure history includes seed
    assert final_history[-2][0] == "USER"
    assert final_history[-2][1] == query

    # Check that more than just the seed was streamed (if max_tokens > 0 and model generated something)
    if max_tokens > 0:
         assert len(streamed_tokens) > 1 or len(final_response) > len(seed), \
            "Only the seed response was streamed, or model generated nothing after seed."

    print("Streaming with seed response test passed.")
