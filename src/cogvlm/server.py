import logging
import argparse
from typing import List, Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
from threading import Thread
from queue import Queue, Empty as QueueEmpty # Import correct Empty exception
import json
import sys

# Assuming ICogVLM and CogVLM are in cogvlm.core
try:
    from cogvlm.core import ICogVLM, CogVLM
except ImportError:
    # Provide dummy classes if core is not available, tests might fail later
    # but allows server structure to be checked.
    _logger.warning("Could not import from cogvlm.core, using dummy classes.")
    class ICogVLM: pass
    class CogVLM(ICogVLM):
        def __init__(self, model_path: str): self.model_path = model_path
        def inference(self, *args, **kwargs): raise NotImplementedError("Dummy CogVLM")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class HistoryItem(BaseModel):
    role: str = Field(..., description="Role of the speaker ('USER' or 'ASSISTANT')")
    content: str = Field(..., description="The message content")

class InferenceRequest(BaseModel):
    query: str = Field(..., description="The user's current query text")
    system_prmpt: Optional[str] = Field(None, description="Optional system prompt")
    images: Optional[List[str]] = Field(None, description="List of image URLs, local file paths, or base64 data URIs")
    history: Optional[List[HistoryItem]] = Field(None, description="Conversation history")
    max_new_tokens: int = Field(2048, description="Maximum number of new tokens to generate")
    top_k: int = Field(1, description="Top-k sampling parameter")
    user_name: str = Field("USER", description="Name for the user role in history")
    assistant_name: str = Field("ASSISTANT", description="Name for the assistant role in history")
    seed_response: str = Field("", description="Optional text to seed the assistant's response")
    stream: bool = Field(False, description="Whether to stream the response")

class InferenceResponse(BaseModel):
    response: str = Field(..., description="The generated response from the model")
    history: List[HistoryItem] = Field(..., description="The updated conversation history")
    error: Optional[str] = Field(None, description="Error message if processing failed")

class StreamChunk(BaseModel):
    token: str = Field(..., description="Token being streamed")
    finished: bool = Field(False, description="Whether this is the final token")
    response: Optional[str] = Field(None, description="The full final response (only in last chunk)")
    history: Optional[List[HistoryItem]] = Field(None, description="The final history (only in last chunk)")
    error: Optional[str] = Field(None, description="Error message if streaming failed (only in last chunk)")


# --- Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the CogVLM model lifecycle using app.state."""
    model_was_preinitialized = hasattr(app.state, 'cogvlm_model') and app.state.cogvlm_model is not None

    if model_was_preinitialized:
        _logger.info(f"Using pre-initialized CogVLM model stored in app.state: {type(app.state.cogvlm_model).__name__}")
    else:
        _logger.info("Attempting to load CogVLM model during startup...")
        parser = argparse.ArgumentParser(description="CogVLM API Server Startup Arguments")
        parser.add_argument(
            "--model-path",
            type=str,
            default="THUDM/cogvlm2-llama3-chat-19B",
            help="Path or Hugging Face identifier for the CogVLM model",
        )
        args, _ = parser.parse_known_args(sys.argv[1:])

        try:
            model_instance = CogVLM(model_path=args.model_path)
            app.state.cogvlm_model = model_instance
            _logger.info(f"CogVLM model loaded from '{args.model_path}' and stored in app.state.")
        except Exception as e:
            _logger.error(f"Failed to load CogVLM model from '{args.model_path}': {e}", exc_info=True)
            app.state.cogvlm_model = None

    yield # Application runs here

    # --- Cleanup ---
    _logger.info("Server shutting down...")
    loaded_model = getattr(app.state, 'cogvlm_model', None)

    if loaded_model is not None:
        if not model_was_preinitialized:
            _logger.info("Unloading internally loaded CogVLM model from app.state.")
            # Add specific cleanup if needed, e.g., del loaded_model
        else:
            _logger.info("Removing reference to pre-initialized model from app.state. External cleanup is expected.")
        try:
            delattr(app.state, 'cogvlm_model')
        except AttributeError:
            pass # Already removed or never set
    else:
         _logger.info("No model found in app.state, nothing to clean up.")


# --- App Factory ---

def create_app(model_instance: Optional[ICogVLM] = None) -> FastAPI:
    """
    Creates the FastAPI application, managing the model via app.state.
    """
    app = FastAPI(
        title="CogVLM Inference API",
        description="API server to interact with the CogVLM model (using app.state).",
        version="1.0.2", # Incremented version
    )

    if model_instance:
        _logger.info("create_app received a pre-initialized model instance. Storing in app.state.")
        if not isinstance(model_instance, ICogVLM):
             raise TypeError("Provided model_instance must implement the ICogVLM interface.")
        app.state.cogvlm_model = model_instance
    else:
        app.state.cogvlm_model = None

    # Attach lifespan AFTER potentially setting state
    app.router.lifespan_context = lifespan

    # --- Helper Function ---
    def convert_history(history_items: Optional[List[HistoryItem]]) -> Optional[List[Tuple[str, str]]]:
        """Convert history from API format to model format"""
        if not history_items:
            return None
        return [(item.role.upper(), item.content) for item in history_items]

    # --- Route Handlers ---
    @app.post("/inference", response_model=InferenceResponse, responses={503: {"description": "Model not available"}})
    async def run_inference(request: Request, payload: InferenceRequest = Body(...)):
        """Runs standard inference (non-streaming)."""
        cogvlm_model = getattr(request.app.state, 'cogvlm_model', None)
        if cogvlm_model is None:
            _logger.error("Model not available in app.state.")
            raise HTTPException(status_code=503, detail="Model is not available.")

        if payload.stream:
            raise HTTPException(status_code=400, detail="Streaming requested. Please use the /inference/stream endpoint.")

        _logger.info(f"Received inference request: '{payload.query[:50]}...'")
        if payload.images: _logger.info(f"Request includes {len(payload.images)} image(s).")
        if payload.history: _logger.info(f"Request includes history with {len(payload.history)} turns.")

        history_tuples = convert_history(payload.history)

        try:
            result = cogvlm_model.inference(
                query=payload.query,
                system_prmpt=payload.system_prmpt,
                images=payload.images,
                history=history_tuples,
                max_new_tokens=payload.max_new_tokens,
                top_k=payload.top_k,
                user_name=payload.user_name,
                assistant_name=payload.assistant_name,
                seed_response=payload.seed_response,
                stream=False,
            )

            if not isinstance(result, tuple) or len(result) != 2:
                 _logger.error(f"Unexpected result type from non-streaming inference: {type(result)}")
                 raise RuntimeError("Model returned unexpected result format for non-streaming inference.")

            response_text, updated_history_tuples = result
            updated_history_items = [
                HistoryItem(role=role, content=content) for role, content in updated_history_tuples
            ]

            _logger.info("Inference successful.")
            return InferenceResponse(response=response_text, history=updated_history_items)

        except FileNotFoundError as e:
            _logger.error(f"Image loading error: {e}")
            raise HTTPException(status_code=400, detail=f"Bad Request: Invalid image path/URL/format. {e}")
        except ValueError as e:
            _logger.error(f"Data processing error: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Bad Request: {e}")
        except RuntimeError as e:
            _logger.error(f"Model inference error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal Server Error: Model inference failed. {e}")
        except Exception as e:
            _logger.error(f"Unexpected error during inference: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal Server Error: An unexpected error occurred.")

    @app.post("/inference/stream", responses={503: {"description": "Model not available"}})
    async def run_streaming_inference(request: Request, payload: InferenceRequest = Body(...)):
        """Streams the inference response."""
        cogvlm_model = getattr(request.app.state, 'cogvlm_model', None)
        if cogvlm_model is None:
            _logger.error("Model not available in app.state for streaming.")
            raise HTTPException(status_code=503, detail="Model is not available.")

        _logger.info(f"Received streaming inference request: '{payload.query[:50]}...'")
        if payload.images: _logger.info(f"Request includes {len(payload.images)} image(s).")
        if payload.history: _logger.info(f"Request includes history with {len(payload.history)} turns.")

        history_tuples = convert_history(payload.history)
        token_queue: Queue[Dict[str, Any]] = Queue()
        job_done = asyncio.Event() # Use asyncio event for signaling

        async def stream_generator():
            """Generator function that yields JSON chunks."""
            _logger.debug("Stream generator started.")
            finished_processing = False
            while not finished_processing:
                try:
                    # Get item without blocking. Raises QueueEmpty if unavailable.
                    chunk_data = token_queue.get_nowait()
                    # _logger.debug(f"Generator got chunk: {chunk_data}") # Verbose
                    yield json.dumps(chunk_data) + "\n"
                    token_queue.task_done() # Mark task as done *after* yielding

                    # Check if this chunk signals the end
                    if chunk_data.get("finished"):
                        _logger.debug("Generator received finished signal in chunk.")
                        finished_processing = True
                        job_done.set() # Ensure event is set
                        break # Exit loop once finished chunk is processed

                except QueueEmpty: # Use the correct exception type
                    # Queue is empty, check if the worker thread is done *before* sleeping
                    if job_done.is_set():
                        # Worker is done, and queue is empty, so we should stop
                        _logger.debug("Generator found queue empty and job done.")
                        finished_processing = True
                        break
                    else:
                        # Worker not done, queue just temporarily empty, wait briefly
                        # _logger.debug("Generator found queue empty, sleeping.") # Verbose
                        await asyncio.sleep(0.01) # Use asyncio sleep

                except Exception as e:
                    # Catch potential errors during json.dumps or validation
                    _logger.error(f"Error in stream generator processing queue item: {e}", exc_info=True)
                    # Attempt to yield an error chunk
                    try:
                        yield json.dumps({"token": "", "finished": True, "error": f"Generator Error: {e}"}) + "\n"
                    except Exception as json_err:
                        _logger.error(f"Failed to yield error JSON: {json_err}")
                    finished_processing = True
                    job_done.set() # Signal error completion
                    break # Exit loop on error

            _logger.debug("Stream generator finished loop.")


        def inference_worker():
            """Worker function runs inference, puts tokens in queue."""
            nonlocal cogvlm_model # Capture from outer scope
            final_result = {}
            worker_error = None
            try:
                _logger.debug("Inference worker started.")
                result = cogvlm_model.inference(
                    query=payload.query,
                    system_prmpt=payload.system_prmpt,
                    images=payload.images,
                    history=history_tuples,
                    max_new_tokens=payload.max_new_tokens,
                    top_k=payload.top_k,
                    user_name=payload.user_name,
                    assistant_name=payload.assistant_name,
                    seed_response=payload.seed_response,
                    stream=True,
                )

                if not isinstance(result, tuple) or len(result) != 2 or not callable(result[1]):
                    raise RuntimeError("Model returned unexpected result format for streaming inference.")

                token_iterator, finalizer = result

                for i, token in enumerate(token_iterator):
                    # _logger.debug(f"Worker putting token {i}") # Verbose
                    chunk = StreamChunk(token=token, finished=False).model_dump(exclude_none=True)
                    token_queue.put(chunk)

                _logger.debug("Worker finished token iteration, calling finalizer.")
                response_text, updated_history_tuples = finalizer()
                _logger.debug("Worker finalizer completed.")
                final_result = {
                    "response": response_text,
                    "history": [HistoryItem(role=role, content=content).model_dump()
                                for role, content in updated_history_tuples]
                }
                final_chunk = StreamChunk(token="", finished=True, **final_result).model_dump(exclude_none=True)
                # _logger.debug("Worker putting final chunk.") # Verbose
                token_queue.put(final_chunk)

            except Exception as e:
                worker_error = e
                _logger.error(f"Error in streaming inference worker: {e}", exc_info=True)
                # Ensure error chunk is put even if finalizer fails
                error_chunk = StreamChunk(token="", finished=True, error=str(e)).model_dump(exclude_none=True)
                token_queue.put(error_chunk)
            finally:
                # Signal that the worker has completed its task *after* putting final/error chunk.
                _logger.debug(f"Inference worker finished. Error: {worker_error}")
                job_done.set() # Crucial: Set event here

        # Start inference in a separate thread
        thread = Thread(target=inference_worker, daemon=True)
        thread.start()

        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson"
        )

    @app.get("/health")
    async def health_check(request: Request):
        """Checks model status via request.app.state."""
        model_instance = getattr(request.app.state, 'cogvlm_model', None)
        model_loaded = model_instance is not None
        model_info = "N/A"

        if model_loaded:
            if hasattr(model_instance, 'model_path'):
                 model_info = getattr(model_instance, 'model_path', 'Custom Instance (No path attribute)')
            else:
                 model_info = f"Instance of {type(model_instance).__name__}"
            # Add check for pre-initialized status if needed
            if hasattr(request.app.state, '_model_was_preinitialized') and request.app.state._model_was_preinitialized:
                 model_info += " (Pre-initialized)"
            return {"status": "ok", "model_loaded": True, "model_info": model_info}
        else:
            status_detail = "Model not loaded or loading failed."
            return {"status": "error", "model_loaded": False, "detail": status_detail}

    return app

# --- Main execution block (for standalone server) ---
def main():
    """Entry point for the server when called via command line."""
    parser = argparse.ArgumentParser(description="CogVLM API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--model-path", type=str, default="THUDM/cogvlm2-llama3-chat-19B", help="Model path/ID")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args() # Parse args to display them

    print(f"Starting server on {args.host}:{args.port} with {args.workers} worker(s)...")
    print(f"Default model path if loading internally: {args.model_path}")
    print(f"Arguments passed to process: {sys.argv}") # Show args lifespan will see

    import uvicorn
    # Uvicorn calls create_app factory. Lifespan uses sys.argv for model path.
    uvicorn.run(
        "cogvlm.server:create_app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        factory=True,
        reload=False # Disable reload for stability
    )

if __name__ == "__main__":
    main()
