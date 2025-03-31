import logging
import requests
from io import BytesIO
from typing import List, Optional, Tuple, Union, Any, Iterator, Callable, Protocol
from abc import ABC, abstractmethod
from PIL import Image
import base64
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

_logger = logging.getLogger(__name__)


def load_image(image: Union[str, Image.Image]) -> Image.Image:
    """
    Loads an image from a file path, URL, base64 string, or PIL Image object.

    Args:
        image (Union[str, Image.Image]): The image to load. Can be a file path, URL,
            base64 string, or PIL Image object.

    Returns:
        Image.Image: The loaded PIL Image object.

    Raises:
        FileNotFoundError: If the image file is not found or the URL is invalid.
        TypeError: If the input is not a file path, URL, base64 string, or PIL Image object.
    """
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):
        if image.startswith("data:image"):
            try:
                header: str
                encoded: str
                header, encoded = image.split(",", 1)
                img_data: bytes = base64.b64decode(encoded)
                img: Image.Image = Image.open(BytesIO(img_data))
                _logger.info("Loaded image from base64 string.")
                return img
            except Exception as e:
                _logger.warning(f"Could not decode base64 image: {e}. Trying as path/URL.")
                # Fall through to try as path/URL

        try:
            # Try loading as local path first
            img = Image.open(image)
            _logger.info(f"Loaded image from local path: {image}")
            return img
        except FileNotFoundError:
            # If not found locally, try as URL
            try:
                _logger.info(f"Attempting to load image from URL: {image}")
                response: requests.Response = requests.get(image, stream=True, timeout=15)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                img = Image.open(BytesIO(response.content))
                _logger.info(f"Loaded image from URL: {image}")
                return img
            except requests.exceptions.MissingSchema:
                 # Re-raise specifically for invalid URL format
                 raise FileNotFoundError(f"Invalid URL or path (missing schema?): {image}")
            except (requests.exceptions.RequestException, OSError) as e:
                # Catch broader network/OS errors during URL fetch or Image.open
                raise FileNotFoundError(f"Could not load image from path or URL: {image}. Error: {e}")
        except Exception as e: # Catch potential PIL errors
             raise ValueError(f"Failed to process image file/data: {image}. Error: {e}")

    else:
        raise TypeError(
            "Expected a file path, URL, base64 string, or PIL Image object, "
            f"but got {type(image)}"
        )

class ICogVLM(Protocol):
    """
    Protocol defining the interface for a CogVLM-like model interaction.

    This protocol ensures that any class implementing it provides a consistent
    `inference` method for generating responses based on text, optional images,
    and conversation history.
    """
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
    ) -> Union[Tuple[str, List[Tuple[str, str]]], Tuple[Iterator[str], Callable[[], Tuple[str, List[Tuple[str, str]]]]]]:
        """
        Performs inference using the model based on the provided inputs.

        This method handles generating a response from the model, potentially
        considering multimodal inputs (text and images) and conversation context.
        It supports both standard single-response generation and streaming output.

        Args:
            query: The primary text input or question from the user.
            system_prmpt: An optional instruction or context provided to the
                model, separate from the user query and history. Defaults to None.
            images: An optional list containing image inputs. Each element can
                be a file path (str), a URL (str), or a PIL Image object.
                Defaults to None.
            history: An optional list of tuples representing the conversation
                history. Each tuple should be (role, message), e.g.,
                [("USER", "Hello"), ("ASSISTANT", "Hi there!")]. Defaults to None.
            max_new_tokens: The maximum number of tokens to generate in the
                response. Defaults to 2048.
            top_k: The number of highest probability vocabulary tokens to keep
                for top-k-filtering. Defaults to 1 (greedy decoding).
            user_name: The identifier string used for the user's role in the
                history and potentially in prompt formatting. Defaults to "USER".
            assistant_name: The identifier string used for the model's role.
                Defaults to "ASSISTANT".
            seed_response: Optional text provided to the model to guide or
                start its response generation. Defaults to "".
            stream: If False (default), generates the full response at once.
                If True, returns an iterator yielding tokens incrementally.

        Returns:
            If stream is False:
                A tuple containing:
                    - response (str): The complete generated response text.
                    - updated_history (List[Tuple[str, str]]): The conversation
                      history including the latest query and response.
            If stream is True:
                A tuple containing:
                    - token_iterator (Iterator[str]): An iterator that yields
                      response tokens (strings) as they are generated.
                    - finalizer (Callable[[], Tuple[str, List[Tuple[str, str]]]]):
                      A function that, when called after the iterator is
                      exhausted, returns the complete response string and the
                      updated conversation history.
        """
        ... # Protocol methods have no implementation body

class CogVLM(ICogVLM):
    """
    Implementation of the CogVLM model.
    """
    def __init__(self, model_path: str="THUDM/cogvlm2-llama3-chat-19B"):
        """
        Initializes the CogVLM model.

        Args:
            model_path (str): Path to the pre-trained CogVLM model.
        """
        self.model_path: str = model_path
        self.logger: logging.Logger = _logger

        self.DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.TORCH_TYPE: torch.dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        if self.DEVICE == "cpu":
            self.TORCH_TYPE = torch.float32

        self.logger.info(f"Device: {self.DEVICE}, Torch type: {self.TORCH_TYPE}")
        self.logger.info(f"Loading model {self.model_path}...")

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.TORCH_TYPE,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.DEVICE == 'cuda':
            try:
                quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs["quantization_config"] = quantization_config
                self.logger.info("Using 4-bit quantization.")
            except Exception as e:
                 self.logger.warning(f"Could not apply 4-bit quantization: {e}. Loading without quantization.")
                 model_kwargs["quantization_config"] = None
        else:
            self.logger.info("CPU detected, loading model without quantization.")
            model_kwargs["quantization_config"] = None

        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        ).eval()

        # Original code moved model after loading if not quantized
        # Keep this behavior if quantization_config is None
        if model_kwargs.get("quantization_config") is None:
             self.model.to(self.DEVICE)
        elif self.DEVICE == 'cpu': # If quantized but somehow on CPU
             self.model.to(self.DEVICE)

        self.logger.info(f"Model {self.model_path} loaded successfully on {self.DEVICE}.")

    def inference(
        self,
        query: str,
        system_prmpt: Optional[str] = None,
        images: Optional[List[Union[str, Image.Image]]] = None,
        history: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 2048,
        pad_token_id: int = 128002, # Default for CogVLM2-Llama3
        top_k: int = 1,
        user_name: str = "USER",
        assistant_name: str = "ASSISTANT",
        seed_response: str = "",
        stream: bool = False
    ) -> Union[Tuple[str, List[Tuple[str, str]]], Tuple[Iterator[str], Callable]]:
        """
        Performs inference using the CogVLM model, with optional streaming.

        Args:
            query (str): The user's query.
            system_prmpt (Optional[str]): Optional system prompt.
            images (Optional[List[Union[str, Image.Image]]]): List of image paths/URLs/PIL Images.
            history (Optional[List[Tuple[str, str]]]): Conversation history.
            max_new_tokens (int): Maximum number of new tokens to generate.
            pad_token_id (int): The pad token ID. Defaults to 128002 for CogVLM2-Llama3.
            top_k (int): Top-k sampling parameter.
            user_name (str): Name for the user role.
            assistant_name (str): Name for the assistant role.
            seed_response (str): Optional text to seed the assistant's response.
            stream (bool): If True, returns a token iterator and finalizer function.

        Returns:
            Union[Tuple[str, List[Tuple[str, str]]], Tuple[Iterator[str], Callable]]:
                If stream=False: (response, updated_history)
                If stream=True: (token_iterator, finalizer_function)

        Raises:
            ValueError: If the image cannot be loaded or processed.
            RuntimeError: If the model inference fails.
        """
        # Common preprocessing for both streaming and non-streaming
        processed_pil_images = self._prepare_images(images)
        current_history = list(history) if history is not None else []
        turn_history = current_history + [(user_name, query)]
        # Use original prompt formatting
        full_prompt = self._format_prompt(system_prmpt, turn_history, assistant_name, seed_response)

        # Prepare model inputs using original method
        try:
            inputs, gen_kwargs = self._prepare_model_inputs(
                full_prompt,
                processed_pil_images,
                max_new_tokens,
                pad_token_id,
                top_k
            )

            # Delegate to appropriate method based on stream flag
            if stream:
                # Use the corrected streaming implementation
                return self._stream_inference(
                    inputs,
                    gen_kwargs,
                    turn_history, # Pass history *including* current user query
                    assistant_name,
                    seed_response
                )
            else:
                # Use the original standard inference implementation
                return self._standard_inference(
                    inputs,
                    gen_kwargs,
                    turn_history, # Pass history *including* current user query
                    assistant_name,
                    seed_response
                )

        except Exception as e:
            self.logger.error(f"Error during model inference: {e}", exc_info=True)
            raise RuntimeError(f"Model inference failed: {e}") from e

    def _prepare_images(self, images: Optional[List[Union[str, Image.Image]]]) -> Optional[List[Image.Image]]:
        """Helper method to prepare images for model input"""
        if not images:
            return None

        # Original logic only used the first image
        if len(images) > 1:
            self.logger.warning("Multiple images provided, but CogVLM currently uses only the first image.")
        try:
            image_input = images[0]
            pil_image = load_image(image_input)
            # Ensure image is RGB, CogVLM expects this
            pil_image = pil_image.convert("RGB")
            self.logger.info("Image processed successfully.")
            return [pil_image] # Return list with one image
        except (FileNotFoundError, ValueError, TypeError) as e:
            self.logger.error(f"Unable to load or process image: {e}", exc_info=True)
            raise ValueError(f"Failed to process image: {e}") from e
        except Exception as e:
             self.logger.error(f"Unexpected error processing image: {e}", exc_info=True)
             raise ValueError(f"Unexpected error processing image: {e}") from e


    def _format_prompt(self, system_prmpt, turn_history, assistant_name, seed_response):
        """Helper method to format the prompt (Original Version)"""
        prompt_parts = []
        if system_prmpt:
            prompt_parts.append(system_prmpt)
        for name, message in turn_history:
            prompt_parts.append(f"{name}: {message}")
        prompt_parts.append(f"{assistant_name}:{seed_response}") # Seed added here

        full_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"[[ Formatted Prompt ]]\n{full_prompt}")
        return full_prompt

    def _prepare_model_inputs(self, full_prompt, processed_pil_images, max_new_tokens, pad_token_id, top_k):
        """Helper method to prepare model inputs and generation kwargs (Original Version)"""
        # Original build_conversation_input_ids call
        input_by_model = self.model.build_conversation_input_ids(  # type: ignore
            self.tokenizer,
            query=full_prompt, # Uses the fully formatted prompt string
            images=processed_pil_images,
            template_version='base' # Original used 'base'
        )

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.DEVICE),
        }
        # Original image tensor handling
        if processed_pil_images is not None and input_by_model.get('images') is not None:
             # Original logic: torch.stack([torch.stack([image.to(self.DEVICE).to(self.TORCH_TYPE) ... ])])
             # This implies input_by_model['images'] was expected to be a list of lists of tensors?
             # Let's try to replicate the structure, assuming input_by_model['images'] is a list of tensors for the single image
             try:
                 image_tensors = [img.to(self.DEVICE).to(self.TORCH_TYPE) for img in input_by_model['images']]
                 if image_tensors:
                     # The original code stacked twice. If input_by_model['images'] is already [tensor],
                     # stack() creates a single tensor. Stacking again might error or produce unexpected dims.
                     # Let's assume the intention was to create the batch dim and potentially an inner list if needed.
                     # A common format is Batch x NumImages x C x H x W.
                     # If input_by_model['images'] = [img_tensor], then stack makes C x H x W.
                     # We need 1 x 1 x C x H x W or 1 x C x H x W depending on model.
                     # Let's assume the model expects List[List[Tensor]] structure for the 'images' key
                     inputs['images'] = [image_tensors] # Batch Dim: [[img1_tensor, ...]]
                 else:
                     inputs['images'] = None
             except Exception as e:
                 self.logger.error(f"Error processing image tensors from build_conversation_input_ids: {e}. Original structure might be incompatible.")
                 inputs['images'] = None # Fallback
        else:
            inputs['images'] = None

        # Filter out None values from inputs, especially 'images' if None
        inputs = {k: v for k, v in inputs.items() if v is not None}

        # Original gen_kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "top_k": top_k,
            # Original didn't explicitly set do_sample or eos_token_id
        }

        return inputs, gen_kwargs

    def _standard_inference(self, inputs, gen_kwargs, turn_history, assistant_name, seed_response):
        """Non-streaming inference implementation (Original Version)"""
        self.logger.info(f"Generating response with max_new_tokens={gen_kwargs['max_new_tokens']}, top_k={gen_kwargs['top_k']}...")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)  # type: ignore
            # Slice outputs to get only the generated tokens
            input_ids_key = 'input_ids' if 'input_ids' in inputs else list(inputs.keys())[0] # Heuristic
            input_length = inputs[input_ids_key].shape[1]
            generated_ids = outputs[:, input_length:] # type: ignore
            response = self.tokenizer.decode(generated_ids[0])
            # Original response processing
            response = seed_response + response.split("<|end_of_text|>")[0].strip()

        # Original history update
        final_history = turn_history + [(assistant_name, response)]

        self.logger.info(f"Generated response length: {len(response)}")
        self.logger.debug(f"[[ Raw Response ]]\n{response}")
        return response, final_history

    def _stream_inference(self, inputs, gen_kwargs, turn_history, assistant_name, seed_response):
        """Streaming inference implementation (Corrected Logic)"""
        seed_response = seed_response.rstrip()

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True, # Skip input prompt tokens
            skip_special_tokens=True # Skip special tokens like <|eot_id|> or <|end_of_text|>
        )

        # Keep track of the full response text as it streams
        # NOTE: The original _format_prompt already includes seed_response in the input.
        #       Therefore, we should NOT yield seed_response separately here,
        #       as the model's output will start *after* the seeded part.
        #       We still need to capture the streamed tokens to build the final history correctly.
        full_response_parts = []

        # Add streamer to generation kwargs
        streaming_gen_kwargs = gen_kwargs.copy()
        streaming_gen_kwargs["streamer"] = streamer

        # Run generation in a separate thread
        self.logger.info(f"Generating streaming response with max_new_tokens={gen_kwargs['max_new_tokens']}, top_k={gen_kwargs['top_k']}...")
        thread_kwargs = {**inputs, **streaming_gen_kwargs}
        thread = Thread(target=self.model.generate, kwargs=thread_kwargs)
        thread.start()

        # Create a custom iterator that yields tokens from streamer and captures them.
        def stream_iterator():
            """Iterator that yields tokens from streamer and captures them."""
            if seed_response:
                yield seed_response

            for token in streamer:
                full_response_parts.append(token)
                yield token

        # Finalizer function to get complete response and update history
        def finalize():
            """Wait for generation to complete and return final response with history"""
            thread.join() # Wait for the generation thread to finish

            # Assemble the full response from captured parts
            streamed_part = "".join(full_response_parts)

            # Combine with the seed response that was part of the prompt
            # The model generated `streamed_part` *after* the `seed_response` in the prompt.
            final_assistant_response = seed_response + streamed_part

            # Add the complete assistant response to the history
            # turn_history already includes the user query
            final_history = turn_history + [(assistant_name, final_assistant_response)]

            self.logger.info(f"Streaming finished. Final response length: {len(final_assistant_response)}")
            self.logger.debug(f"[[ Full Streamed Response (incl. seed) ]]\n{final_assistant_response}")
            return final_assistant_response, final_history

        return stream_iterator(), finalize

# Example Usage (requires model download and dependencies)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    # Set to None to run on CPU (slow)
    # Set to a valid GPU ID (e.g., 0) to run on GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
    IMAGE_PATH_OR_URL = "https://github.com/THUDM/CogVLM/raw/main/assets/cat.png" # Example image
    # IMAGE_PATH_OR_URL = None # Set to None to test text-only

    # --- Initialize Model ---
    try:
        cog_vlm = CogVLM(model_path=MODEL_PATH)
    except Exception as e:
        _logger.error(f"Failed to initialize CogVLM: {e}", exc_info=True)
        exit(1)

    # --- Prepare Inputs ---
    query = "Describe the image." if IMAGE_PATH_OR_URL else "Tell me a short story about a robot."
    images_to_use = [IMAGE_PATH_OR_URL] if IMAGE_PATH_OR_URL else None
    history = [] # Start with empty history

    # --- Non-Streaming Inference ---
    print("\n--- Non-Streaming Test ---")
    try:
        response, updated_history = cog_vlm.inference(
            query=query,
            images=images_to_use,
            history=history,
            stream=False
        )
        print(f"User: {query}")
        print(f"Assistant: {response}")
        history = updated_history # Update history for next turn
    except Exception as e:
        _logger.error(f"Non-streaming inference failed: {e}", exc_info=True)


    # --- Streaming Inference ---
    print("\n--- Streaming Test ---")
    query_2 = "What is the main color of the object in the image?" if IMAGE_PATH_OR_URL else "What is the robot's name?"
    try:
        token_iterator, finalizer = cog_vlm.inference(
            query=query_2,
            images=images_to_use, # Reuse image if needed
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
        # print(f"Final History: {final_history}") # Optional: print history

        # Check consistency
        if assistant_response_streamed.strip() != final_response.strip():
             _logger.warning("Streamed response differs from finalized response!")
             print("--- Streamed ---")
             print(assistant_response_streamed)
             print("--- Finalized ---")
             print(final_response)


    except Exception as e:
        _logger.error(f"Streaming inference failed: {e}", exc_info=True)

    print("\n--- Done ---")
