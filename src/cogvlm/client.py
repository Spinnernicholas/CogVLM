import logging
import requests
import json
import base64
import io
from typing import List, Optional, Tuple, Union, Iterator, Callable, Dict, Any
from PIL import Image # Now used for processing

# Attempt to import ICogVLM from core, provide dummy if not found
try:
    from cogvlm.core import ICogVLM
except ImportError:
    logging.warning("Could not import ICogVLM from cogvlm.core, using dummy class.")
    from abc import ABC, abstractmethod
    class ICogVLM(ABC):
        @abstractmethod
        def inference(
            self,
            query: str,
            system_prmpt: Optional[str] = None,
            images: Optional[List[Union[str, Image.Image]]] = None,
            history: Optional[List[Tuple[str, str]]] = None,
            max_new_tokens: int = 2048,
            top_k: int = 1,
            user_name: str = "USER",
            assistant_name: str = "ASSISTANT",
            seed_response: str = "",
            stream: bool = False,
        ) -> Union[Tuple[str, List[Tuple[str, str]]], Tuple[Iterator[str], Callable]]:
            raise NotImplementedError

_logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _convert_history_to_api(
    history_tuples: Optional[List[Tuple[str, str]]]
) -> Optional[List[Dict[str, str]]]:
    """Converts model history format to API format."""
    if not history_tuples:
        return None
    return [{"role": role, "content": content} for role, content in history_tuples]

def _convert_history_from_api(
    history_items: Optional[List[Dict[str, Any]]]
) -> List[Tuple[str, str]]:
    """Converts API history format back to model format."""
    if not history_items:
        return []
    # Ensure role is uppercase as expected by core logic if reused
    return [(item.get("role", "UNKNOWN").upper(), item.get("content", "")) for item in history_items]


class CogVLMClient(ICogVLM):
    """
    Client implementation for interacting with the CogVLM FastAPI server.
    Implements the ICogVLM interface. Handles local image paths and PIL Images
    by converting them to base64 data URIs before transmission.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initializes the CogVLM client.

        Args:
            base_url (str): The base URL of the CogVLM server (e.g., "http://localhost:8000").
        """
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.base_url: str = base_url.rstrip("/")
        self.inference_url: str = f"{self.base_url}/inference"
        self.stream_url: str = f"{self.base_url}/inference/stream"
        self.health_url: str = f"{self.base_url}/health"
        self.session: requests.Session = requests.Session() # Use session for potential connection pooling
        self.logger: logging.Logger = _logger
        self.logger.info(f"CogVLMClient initialized for server at {self.base_url}")

    def _pil_to_base64_uri(self, img: Image.Image, format: str = "PNG") -> str:
        """Converts a PIL Image object to a base64 data URI."""
        buffered = io.BytesIO()
        img_format = img.format or format # Use original format if available, else default
        if img_format not in ["PNG", "JPEG", "WEBP"]: # Ensure supported format
             self.logger.warning(f"Unsupported image format '{img_format}', defaulting to PNG for base64 encoding.")
             img_format = "PNG"

        # Ensure image is in RGB or RGBA for formats like JPEG/PNG
        if img_format == "JPEG" and img.mode != "RGB":
            img = img.convert("RGB")
        elif img_format == "PNG" and img.mode not in ("RGB", "RGBA"):
             img = img.convert("RGBA") # PNG supports transparency

        img.save(buffered, format=img_format)
        img_bytes = buffered.getvalue()
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        mime_type = Image.MIME.get(img_format, f"image/{img_format.lower()}")
        return f"data:{mime_type};base64,{encoded}"

    def _process_image_input(self, image_input: Union[str, Image.Image]) -> str:
        """
        Processes a single image input (path, URL, base64, PIL Image)
        and returns a string (URL or base64 URI) suitable for the API.
        """
        if isinstance(image_input, Image.Image):
            self.logger.debug("Converting PIL Image to base64 URI.")
            return self._pil_to_base64_uri(image_input)
        elif isinstance(image_input, str):
            if image_input.startswith(("http://", "https://", "data:image")):
                # Pass through URLs and existing data URIs
                self.logger.debug(f"Passing through image string: {image_input[:60]}...")
                return image_input
            else:
                # Assume it's a local file path (relative to the client)
                self.logger.debug(f"Attempting to load local image path: {image_input}")
                try:
                    pil_img = Image.open(image_input)
                    return self._pil_to_base64_uri(pil_img)
                except FileNotFoundError:
                    self.logger.error(f"Client could not find local image file: {image_input}")
                    raise FileNotFoundError(f"Client could not find local image file: {image_input}")
                except Exception as e:
                    self.logger.error(f"Client failed to load or process image file {image_input}: {e}", exc_info=True)
                    raise ValueError(f"Client failed to load or process image file {image_input}: {e}") from e
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}. Expected str or PIL.Image.Image.")


    def check_health(self) -> Dict[str, Any]:
        """Checks the health of the server."""
        try:
            response = self.session.get(self.health_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "error", "detail": f"Client connection error: {e}"}
        except json.JSONDecodeError as e:
             self.logger.error(f"Failed to decode health check response: {e}")
             return {"status": "error", "detail": f"Invalid JSON response from server: {e}"}


    def inference(
        self,
        query: str,
        system_prmpt: Optional[str] = None,
        images: Optional[List[Union[str, Image.Image]]] = None,
        history: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 2048,
        top_k: int = 1,
        user_name: str = "USER",
        assistant_name: str = "ASSISTANT",
        seed_response: str = "",
        stream: bool = False,
    ) -> Union[Tuple[str, List[Tuple[str, str]]], Tuple[Iterator[str], Callable]]:
        """
        Sends an inference request to the CogVLM server.

        Args:
            query (str): The user's query.
            system_prmpt (Optional[str]): Optional system prompt.
            images (Optional[List[Union[str, Image.Image]]]): List of image inputs.
                Can contain:
                - URLs (str): Passed directly to the server.
                - Base64 data URIs (str): Passed directly to the server.
                - Local file paths (str): Loaded by the client and sent as base64 URIs.
                - PIL Image objects: Converted to base64 URIs by the client.
            history (Optional[List[Tuple[str, str]]]): Conversation history in (role, content) tuple format.
            max_new_tokens (int): Maximum number of new tokens to generate.
            top_k (int): Top-k sampling parameter.
            user_name (str): Name for the user role.
            assistant_name (str): Name for the assistant role.
            seed_response (str): Optional text to seed the assistant's response.
            stream (bool): If True, request streaming response.

        Returns:
            Union[Tuple[str, List[Tuple[str, str]]], Tuple[Iterator[str], Callable]]:
                If stream=False: (response_text, updated_history_tuples)
                If stream=True: (token_iterator, finalizer_function)

        Raises:
            FileNotFoundError: If a local image path provided by the client cannot be found.
            TypeError: If the images list contains unsupported types.
            ValueError: If image loading/processing fails, or if the server returns an error
                        or unexpected response format.
            requests.exceptions.RequestException: For network or server connection errors.
        """
        processed_images_payload: Optional[List[str]] = None
        if images:
            processed_images_payload = []
            self.logger.info(f"Processing {len(images)} image input(s) on the client...")
            for i, img_input in enumerate(images):
                try:
                    processed_img_str = self._process_image_input(img_input)
                    processed_images_payload.append(processed_img_str)
                    self.logger.debug(f"Processed image {i+1} successfully.")
                except (FileNotFoundError, ValueError, TypeError) as e:
                    # Re-raise errors encountered during client-side processing
                    raise e
                except Exception as e:
                     # Catch unexpected errors during processing
                     self.logger.error(f"Unexpected error processing image input {i+1}: {e}", exc_info=True)
                     raise ValueError(f"Unexpected client error processing image input: {e}") from e
            self.logger.info("Client-side image processing complete.")


        api_history = _convert_history_to_api(history)

        payload: Dict[str, Any] = {
            "query": query,
            "system_prmpt": system_prmpt,
            "images": processed_images_payload, # Use the processed list
            "history": api_history,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "user_name": user_name,
            "assistant_name": assistant_name,
            "seed_response": seed_response,
            "stream": stream, # Include stream flag in payload for non-streaming endpoint check
        }

        # Filter out None values from payload before sending
        payload = {k: v for k, v in payload.items() if v is not None}

        if stream:
            # --- Streaming Inference ---
            payload["stream"] = True # Ensure stream is True for the request
            target_url = self.stream_url
            self.logger.info(f"Sending streaming request to {target_url}")

            try:
                response = self.session.post(
                    target_url,
                    json=payload,
                    stream=True, # Enable streaming in requests library
                    timeout=300 # Longer timeout for potentially long streams
                )
                response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

                # Variables to store the final state captured from the stream
                final_data = {
                    "response": None,
                    "history": None,
                    "error": None
                }

                def stream_iterator() -> Iterator[str]:
                    """Iterator that yields tokens and captures final state."""
                    nonlocal final_data
                    try:
                        for line in response.iter_lines():
                            if line:
                                try:
                                    chunk_data = json.loads(line.decode('utf-8'))
                                    token = chunk_data.get("token", "")
                                    is_finished = chunk_data.get("finished", False)
                                    error_msg = chunk_data.get("error")

                                    if error_msg:
                                        self.logger.error(f"Server reported stream error: {error_msg}")
                                        final_data["error"] = error_msg
                                        break # Stop iteration on error

                                    if token:
                                        yield token

                                    if is_finished:
                                        self.logger.info("Received finished signal in stream.")
                                        final_data["response"] = chunk_data.get("response")
                                        final_data["history"] = chunk_data.get("history") # API format
                                        break # Stop iteration once finished

                                except json.JSONDecodeError as e:
                                    self.logger.error(f"Failed to decode stream chunk: {line.decode('utf-8')}. Error: {e}")
                                    final_data["error"] = f"Client JSON decode error: {e}"
                                    break
                                except Exception as e:
                                    self.logger.error(f"Unexpected error processing stream chunk: {e}", exc_info=True)
                                    final_data["error"] = f"Client stream processing error: {e}"
                                    break
                    finally:
                        response.close() # Ensure connection is closed

                def finalizer() -> Tuple[str, List[Tuple[str, str]]]:
                    """Returns the final response and history captured by the iterator."""
                    if final_data["error"]:
                        raise ValueError(f"Streaming failed: {final_data['error']}")
                    if final_data["response"] is None or final_data["history"] is None:
                         self.logger.warning("Finalizer called but final data not fully captured from stream.")
                         raise ValueError("Streaming ended unexpectedly without final data.")

                    final_history_tuples = _convert_history_from_api(final_data["history"])
                    return final_data["response"], final_history_tuples

                return stream_iterator(), finalizer

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Streaming request failed: {e}", exc_info=True)
                raise  # Re-raise the exception
            except Exception as e:
                self.logger.error(f"Unexpected error setting up streaming request: {e}", exc_info=True)
                raise ValueError(f"Failed to initiate streaming: {e}") from e

        else:
            # --- Non-Streaming Inference ---
            payload["stream"] = False # Ensure stream is False
            target_url = self.inference_url
            self.logger.info(f"Sending non-streaming request to {target_url}")

            try:
                response = self.session.post(
                    target_url,
                    json=payload,
                    timeout=180 # Shorter timeout for non-streaming
                )
                response.raise_for_status() # Check for HTTP errors

                response_data = response.json()

                # Validate response structure
                if "error" in response_data and response_data["error"]:
                    self.logger.error(f"Server returned error: {response_data['error']}")
                    raise ValueError(f"Server Error: {response_data['error']}")
                if "response" not in response_data or "history" not in response_data:
                    self.logger.error(f"Invalid response format from server: {response_data}")
                    raise ValueError("Invalid response format received from server.")

                response_text = response_data["response"]
                updated_history_api = response_data["history"]
                updated_history_tuples = _convert_history_from_api(updated_history_api)

                self.logger.info("Non-streaming inference successful.")
                return response_text, updated_history_tuples

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Non-streaming request failed: {e}", exc_info=True)
                raise # Re-raise the exception
            except json.JSONDecodeError as e:
                 self.logger.error(f"Failed to decode non-streaming response: {e}")
                 raise ValueError(f"Invalid JSON response from server: {e}") from e
            except Exception as e:
                self.logger.error(f"Unexpected error during non-streaming inference: {e}", exc_info=True)
                raise ValueError(f"Unexpected client error: {e}") from e


# Example Usage (requires the server from server.py to be running)
if __name__ == '__main__':
    import os
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    SERVER_URL = "http://localhost:8000" # Make sure this matches your server address

    # --- Create a dummy image file for local path testing ---
    DUMMY_IMAGE_FILENAME = "client_test_image.png"
    try:
        img = Image.new('RGB', (60, 30), color = 'red')
        img.save(DUMMY_IMAGE_FILENAME)
        _logger.info(f"Created dummy image file: {DUMMY_IMAGE_FILENAME}")
    except Exception as e:
        _logger.error(f"Could not create dummy image file: {e}. Local path test may fail.")
        DUMMY_IMAGE_FILENAME = None # Prevent using it if creation failed

    # --- Image Inputs (Choose one or combine) ---
    # 1. URL
    IMAGE_URL = "https://github.com/THUDM/CogVLM/raw/main/assets/cat.png"
    # 2. Local Path (using the dummy file created above)
    IMAGE_LOCAL_PATH = DUMMY_IMAGE_FILENAME if DUMMY_IMAGE_FILENAME else None
    # 3. PIL Image object (created in memory)
    IMAGE_PIL = Image.new('RGB', (80, 40), color = 'blue')
    # 4. None
    # IMAGE_INPUT = None

    # Select which image(s) to use for the test
    # Example: Use the URL and the PIL image
    # images_to_use = [IMAGE_URL, IMAGE_PIL]
    # Example: Use the local path if available
    images_to_use = [IMAGE_LOCAL_PATH] if IMAGE_LOCAL_PATH else None
    # Example: Use only the URL
    # images_to_use = [IMAGE_URL]
    # Example: Text-only
    # images_to_use = None

    # --- Initialize Client ---
    client = CogVLMClient(base_url=SERVER_URL)

    # --- Check Server Health ---
    print("\n--- Health Check ---")
    health_status = client.check_health()
    print(f"Server Health: {health_status}")
    if health_status.get("status") != "ok" or not health_status.get("model_loaded"):
        _logger.error("Server is not healthy or model not loaded. Aborting tests.")
        if DUMMY_IMAGE_FILENAME and os.path.exists(DUMMY_IMAGE_FILENAME):
            os.remove(DUMMY_IMAGE_FILENAME) # Clean up dummy file
        exit(1)

    # --- Prepare Inputs ---
    query = "Describe the image(s)." if images_to_use else "Tell me a short story about a curious robot."
    history = [] # Start with empty history

    # --- Non-Streaming Inference ---
    print("\n--- Non-Streaming Test ---")
    try:
        response, updated_history = client.inference(
            query=query,
            images=images_to_use,
            history=history,
            stream=False
        )
        print(f"User: {query}")
        print(f"Assistant: {response}")
        # History is now List[Tuple[str, str]]
        history = updated_history # Update history for next turn
        print(f"Updated History (tuples): {history}")
    except Exception as e:
        _logger.error(f"Non-streaming inference failed: {e}", exc_info=True)


    # --- Streaming Inference ---
    print("\n--- Streaming Test ---")
    query_2 = "What is the main color in the image(s)?" if images_to_use else "What was the robot's primary function?"
    try:
        # Pass the history obtained from the previous non-streaming call
        token_iterator, finalizer = client.inference(
            query=query_2,
            images=images_to_use, # Reuse image(s) if needed
            history=history, # Use history from previous turn
            stream=True
        )

        print(f"User: {query_2}")
        print(f"Assistant (streaming): ", end="", flush=True)
        assistant_response_streamed = ""
        for token in token_iterator:
            print(token, end="", flush=True)
            assistant_response_streamed += token
        print() # Newline after streaming finishes

        # Call finalize to get the complete response and updated history
        final_response, final_history = finalizer()

        # Verify streamed response matches finalized response (should be identical)
        print(f"Finalized Response Length: {len(final_response)}")
        print(f"Final History (tuples): {final_history}") # Optional: print history

        # Check consistency
        if assistant_response_streamed.strip() != final_response.strip():
             _logger.warning("Streamed response differs from finalized response!")
             print("--- Streamed ---")
             print(assistant_response_streamed)
             print("--- Finalized ---")
             print(final_response)
        else:
             print("Streamed and finalized responses match.")


    except Exception as e:
        _logger.error(f"Streaming inference failed: {e}", exc_info=True)

    # --- Clean up dummy image file ---
    if DUMMY_IMAGE_FILENAME and os.path.exists(DUMMY_IMAGE_FILENAME):
        try:
            os.remove(DUMMY_IMAGE_FILENAME)
            _logger.info(f"Cleaned up dummy image file: {DUMMY_IMAGE_FILENAME}")
        except Exception as e:
            _logger.warning(f"Could not remove dummy image file {DUMMY_IMAGE_FILENAME}: {e}")

    print("\n--- Done ---")
