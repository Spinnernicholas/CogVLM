# conftest.py (in tests directory)
import time
import socket
import threading
import logging
import httpx
import uvicorn
import pytest
import pytest_asyncio
from unittest.mock import MagicMock
from typing import AsyncGenerator, Dict, Any, List, Tuple

from cogvlm import CogVLM, ICogVLM
from cogvlm.server import (
    create_app,
    InferenceRequest,
    InferenceResponse,
    StreamChunk,
)

_logger = logging.getLogger(__name__)

# --- Helper Function ---
def find_free_port():
    """Finds an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0)) # Bind to "" or "127.0.0.1"
        s.listen(1)
        port = s.getsockname()[1]
    return port

# --- Fixtures ---
@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "Test response"
    return tokenizer

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.generate.return_value = MagicMock()
    return model

@pytest.fixture
def mock_image():
    from PIL import Image
    import numpy as np
    array = np.zeros((100, 100, 3), dtype=np.uint8)
    return Image.fromarray(array)

TEST_MODEL = "THUDM/cogvlm2-llama3-chat-19B"

@pytest.fixture(scope="session")
def shared_model() -> ICogVLM: # Use the interface for better type hinting
    """
    Initializes the CogVLM model once for the entire test session.
    Located in conftest.py for sharing across test files.
    """
    print(f"\n--- [conftest.py] Initializing shared model: {TEST_MODEL} (scope: session) ---")
    start_time = time.time()
    try:
        # Ensure CogVLM is defined/imported before this line
        model = CogVLM(model_path=TEST_MODEL)
        # You could add a quick check here, e.g., a dummy inference
        # model.inference("hello", max_new_tokens=1)
    except NameError:
        pytest.fail("CogVLM class is not defined or imported.", pytrace=False)
    except Exception as e:
        pytest.fail(f"Failed to initialize shared CogVLM model: {e}", pytrace=False)

    end_time = time.time()
    print(f"--- [conftest.py] Shared model initialized in {end_time - start_time:.2f}s ---")

    yield model # Provide the initialized model instance

    # Teardown (optional, runs after all tests using this fixture finish)
    print("\n--- [conftest.py] Tearing down shared model ---")
    # If CogVLM requires specific cleanup (e.g., releasing GPU memory explicitly), do it here.
    # del model # Python's garbage collector usually handles this

@pytest.fixture(scope="session")
def server_port():
    """Provides a free port for the test server."""
    return find_free_port()

@pytest.fixture(scope="session")
def server_base_url(server_port):
    """Provides the base URL for the test server."""
    return f"http://127.0.0.1:{server_port}"

# Keep server session-scoped for performance
@pytest.fixture(scope="session")
def uvicorn_server(shared_model, server_port): # Depends on shared_model from conftest
    """
    Starts/stops Uvicorn server in a thread, using pre-initialized shared_model.
    """
    _logger.info("[test_server.py] Creating FastAPI app with pre-initialized shared_model...")
    # Pass the already initialized model instance
    app = create_app(model_instance=shared_model)

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=server_port,
        log_level="info", # Use "warning" or "error" for less noise
        lifespan="on",
        access_log=False, # Disable access logs unless debugging specific requests
    )
    server = uvicorn.Server(config=config)
    # Run server in a background thread
    thread = threading.Thread(target=server.run, daemon=True)

    _logger.info(f"[test_server.py] Starting Uvicorn server on port {server_port}...")
    thread.start()

    # --- Health check / wait logic ---
    startup_timeout = 60 # Allow ample time for server init + health check
    start_time = time.time()
    server_ready = False
    health_check_url = f"http://127.0.0.1:{server_port}/health"
    # Use standard httpx client for synchronous health check during setup
    with httpx.Client(timeout=10.0) as client:
        while time.time() - start_time < startup_timeout:
            try:
                response = client.get(health_check_url)
                # Check status and specific payload indicating readiness
                if response.status_code == 200 and response.json().get("model_loaded"):
                    _logger.info("[test_server.py] Server is up and model state confirmed via health check.")
                    server_ready = True
                    break
            except httpx.ConnectError:
                # Log connection errors less frequently during startup polling
                if (time.time() - start_time) % 5 < 0.5: # Log roughly every 5s
                     _logger.info("[test_server.py] Server not ready yet (connection refused), retrying...")
            except Exception as e:
                 # Log other errors during health check attempt
                 _logger.warning(f"[test_server.py] Error during health check: {e}")
            time.sleep(0.5) # Check reasonably often

    if not server_ready:
        # If server didn't become ready, attempt cleanup and fail the test setup
        server.should_exit = True # Signal server thread to stop
        thread.join(timeout=5)
        pytest.fail(f"Server failed to report model loaded via /health within {startup_timeout} seconds.", pytrace=False)

    yield server # Provide the server object (mainly for teardown signal)

    # Teardown: Stop the server after all session tests are done
    _logger.info("[test_server.py] Stopping Uvicorn server (session end)...")
    server.should_exit = True # Signal the server thread to stop
    thread.join(timeout=10) # Wait for the thread to finish gracefully
    _logger.info("[test_server.py] Server stopped.")


# Keep client function-scoped to avoid event loop issues
@pytest_asyncio.fixture(scope="function")
async def api_client(server_base_url, uvicorn_server) -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Provides an httpx AsyncClient scoped to each test function.
    Ensures proper cleanup within the function's event loop.
    """
    # uvicorn_server fixture (session-scoped) must have run already.
    # Create a new client for each test function.
    async with httpx.AsyncClient(base_url=server_base_url, timeout=60.0) as client:
        yield client
    # Client is closed here, within the test function's async context