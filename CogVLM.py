import logging
import random
import argparse
import json
from jsonschema import validate, ValidationError

from PIL import Image
from io import BytesIO
from typing import Union
import numpy as np
import requests

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_logger = logging.getLogger(__name__)


def extract_json(response):
    """
    Extracts and validates JSON from an AI response. Handles potential errors.
    """
    try:
        # Attempt to load the entire response as JSON first
        data = json.loads(response)
        return data
    except json.JSONDecodeError:
        # If that fails, try to find the JSON within the string
        try:
            start_index = response.find("{")  # Find the first opening brace
            end_index = response.rfind("}")  # Find the last closing brace
            if start_index != -1 and end_index != -1:
                json_string = response[start_index : end_index + 1]
                data = json.loads(json_string)
                return data
            else:
                _logger.warning("No JSON found in the response.")
                return None
        except json.JSONDecodeError as e:
            _logger.error(f"Error decoding JSON: {e}")
            return None


def validate_json_schema(json_data, schema):
    """
    Validates a JSON object against a JSON schema using jsonschema library.

    Args:
        json_data: The JSON data to validate (as a Python dictionary).
        schema: The JSON schema (as a Python dictionary).

    Returns:
        Tuple: (bool, str). The first element is True if the JSON is valid, False otherwise.
               The second element is None if valid, or an error message if invalid.
    """
    try:
        validate(instance=json_data, schema=schema)
        return True, None  # Valid, no error
    except ValidationError as e:
        _logger.error(f"JSON Validation Error: {e}")
        return False, str(e)  # Invalid, with error message


def get_random_examples(data, num_examples=3):
    if num_examples > len(data):
        logging.getLogger(__name__).warning(
            "Warning: num_examples is greater than the length of the data. Returning empty list."
        )
        return []

    return random.sample(data, num_examples)


def load_image(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):
        try:
            # Try to open the image from a local file path
            img = Image.open(image)
            return img
        except FileNotFoundError:
            try:
                # Try to open the image from a URL
                response = requests.get(image, stream=True)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                img = Image.open(BytesIO(response.content))
                return img
            except (requests.exceptions.RequestException, OSError):
                raise FileNotFoundError(
                    f"Could not load image from path or URL: {image}"
                )
    else:
        raise TypeError(
            "Expected a file path, URL, or PIL Image object, "
            f"but got {type(image)}"
        )

class CogVLMChat:
    def __init__(self, model, user_name="USER", assistant_name="ASSISTANT"):
        self.model = model
        self.user_name = user_name
        self.assistant_name = assistant_name
        self.history = []
        self.image = None
        self.image_path = None  # Store the path or URL of the loaded image

    def chat(self, query):
        response, self.history = self.model.inference(
            query,
            images=[self.image] if self.image is not None else None,
            history=self.history,
        )
        return response

    def open_image(self, image_path):
        """
        Opens an image from a file path or URL and sets it as the current image.

        Args:
            image_path (str): Path or URL to the image

        Returns:
            tuple: (success, message) where success is a boolean indicating if the operation
                  was successful, and message is a string with information about the result
        """
        try:
            img = load_image(image_path)
            img = img.convert("RGB")
            self.image = img
            self.image_path = image_path  # Save the image path
            return (
                True,
                f"Successfully opened image: {image_path} ({img.size[0]}x{img.size[1]})",
            )
        except FileNotFoundError:
            return False, f"Image file not found: {image_path}"
        except requests.exceptions.RequestException as e:
            return False, f"Error downloading image: {str(e)}"
        except Exception as e:
            return False, f"Error processing image: {str(e)}"

    def get_image_info(self):
        """
        Returns information about the currently loaded image.

        Returns:
            str: Information about the image or a message if no image is loaded
        """
        if self.image is None:
            return "No image is currently loaded"

        # Include the image path in the information
        path_info = (
            f"Path: {self.image_path}"
            if self.image_path
            else "Path: Unknown (loaded directly)"
        )
        return f"Current image: {self.image.size[0]}x{self.image.size[1]} pixels, mode: {self.image.mode}\n{path_info}"

    def clear_history(self):
        """Clears the chat history"""
        self.history = []
        return "Chat history cleared"

    def reset(self):
        """Resets both the image and chat history"""
        self.history = []
        self.image = None
        self.image_path = None  # Also reset the image path
        return "Chat history and image have been reset"

    def start_cmd_chat(self):
        print("Interactive chat started. Type /help for available commands.")

        # Define available commands and their help text
        commands = {
            "/help": "Show this help message",
            "/exit": "Exit the chat",
            "/open [path]": "Open an image from a file path or URL",
            "/clear": "Clear the chat history",
            "/image": "Show information about the currently loaded image",
            "/reset": "Reset the image and chat history",
        }

        while True:
            try:
                query = input(f"{self.user_name}: ")
            except EOFError: # Handle Ctrl+D or end of input stream
                print("\nExiting chat...")
                break

            # Handle commands (starting with /)
            if query.startswith("/"):
                cmd_parts = query.split(" ", 1)
                cmd = cmd_parts[0].lower()
                args = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd == "/exit":
                    print("Exiting chat...")
                    break

                elif cmd == "/help":
                    print("\nAvailable commands:")
                    for command, description in commands.items():
                        print(f"  {command:<15} - {description}")
                    print()
                    continue

                elif cmd == "/open":
                    if not args:
                        print("Error: Please provide a path or URL to an image")
                        continue

                    success, message = self.open_image(args.strip())
                    print(message)
                    continue

                elif cmd == "/clear":
                    print(self.clear_history())
                    continue

                elif cmd == "/image":
                    print(self.get_image_info())
                    continue

                elif cmd == "/reset":
                    print(self.reset())
                    continue

                else:
                    print(
                        f"Unknown command: {cmd}. Type /help for available commands."
                    )
                    continue

            # Regular chat message
            try:
                response = self.chat(query)
                print(f"{self.assistant_name}: {response}")
            except Exception as e:
                _logger.error(f"Error during model inference: {e}")
                print("ASSISTANT: Sorry, an error occurred while generating the response.")

class CogVLM:
    def __init__(self, model_path="THUDM/cogvlm2-llama3-chat-19B"):
        self.model_path = model_path
        # Max text tokens including response: 8192

        self.logger = logging.getLogger(__name__)

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.TORCH_TYPE = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self.logger.info(
            f" Device: {self.DEVICE}, Torch type: {self.TORCH_TYPE}"
        )

        self.logger.info(f"Loading model {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # Determine quantization based on device
        quantization_config = None
        if self.DEVICE == 'cuda':
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.logger.info("Using 4-bit quantization.")
        else:
            self.logger.info("CPU detected, loading model without quantization.")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.TORCH_TYPE,
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        ).eval()
        self.logger.info(f"Model {self.model_path} loaded successfully.")

    def inference(
        self,
        query,
        system_prmpt=None,
        images=None,
        history=None,
        max_new_tokens=2048,
        pad_token_id=128002,
        top_k=1,
        user_name="USER",
        assistant_name="ASSISTANT",
        seed_response=""
    ):
        processed_images = None
        if images is not None:
            if len(images) > 1:
                self.logger.warning(
                    "Multiple images provided, but only the first image will be used."
                )
            try:
                image = load_image(images[0]) # Use only the first image
                image = image.convert("RGB")
                processed_images = [image] # Keep as a list for build_conversation_input_ids
            except Exception as e:
                self.logger.error(f"Unable to load or process image: {e}")
                # Decide how to handle image error: return error message or proceed without image?
                # For now, let's proceed without the image and log the error.
                processed_images = None # Ensure it's None if loading failed

        if history is None:
            history = []

        # Create a temporary history copy for this inference call if needed,
        # or modify the original history list directly.
        # Current implementation modifies the original history list passed in.
        history += [(user_name, query)]

        # Build the prompt string from history
        prompt = "" if system_prmpt is None else system_prmpt + "\n"
        for name, message in history:
            prompt += f"{name}: {message}\n"
        prompt += f"{assistant_name}:{seed_response}" # Add assistant tag to prompt generation

        self.logger.debug(f"[[ Formatted Prompt ]]\n{prompt}")

        try:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=prompt,
                images=processed_images,
                template_version='base'
            )

            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.DEVICE),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.DEVICE),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.DEVICE),
                #'images': [[input_by_model['images'][0].to(self.DEVICE).to(self.TORCH_TYPE)]] if images is not None else None,
                'images': [[image.to(self.DEVICE).to(self.TORCH_TYPE) for image in input_by_model['images']]] if images is not None else None,
            }

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": pad_token_id,
                "top_k": top_k
            }

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
                # Slice outputs to get only the generated tokens
                generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
                response = self.tokenizer.decode(generated_ids[0])
                response = seed_response + response.split("<|end_of_text|>")[0].strip()

            # Append the successful query and response to the original history
            history.append((assistant_name, response))

            self.logger.debug(f"[[ Raw Response ]]\n{response}")
            return response, history

        except Exception as e:
            self.logger.error(f"Error during model inference pipeline: {e}", exc_info=True)
            # Return an error message and the history *before* this failed turn
            return "Sorry, an error occurred during processing.", history


    def create_chat(self, user_name=None, assistant_name=None):
        # Pass default names if None are provided
        user_name = user_name if user_name is not None else "USER"
        assistant_name = assistant_name if assistant_name is not None else "ASSISTANT"
        return CogVLMChat(
            self, user_name=user_name, assistant_name=assistant_name
        )

    def generate_caption(
        self,
        image,
        query=None,
    ):
        if not query:
            query="Describe what you see in the image below. Write a concise, descriptive caption at least 10 words long."

        if image is None:
            self.logger.error("Image is None, cannot generate caption.")
            return None
        try:
            # Use inference with no history and the provided image/query
            response, _ = self.inference(query, images=[image])
            return response
        except Exception as e:
            self.logger.error(f"Error generating caption: {e}")
            return None # Or raise the exception, depending on desired behavior

    def request_json(
        self,
        query,
        image=None,
        extract=False,
        schema=None,
        validate_schema=False,
        max_retries=0,
    ):
        if validate_schema and not extract:
            validate_schema = False
            self.logger.warning(
                "validate_schema can only be True if extract is True. Disabling schema validation."
            )
        if validate_schema and not schema:
            validate_schema = False
            self.logger.warning(
                "validate_schema can only be True if schema is provided. Disabling schema validation."
            )

        current_history = [] # Maintain history across retries
        system_prmpt = "You are a helpful assistant that responds in a valid JSON format."
        original_query = query # Keep the original query for retries

        if schema:
            try:
                # Ensure schema is a dict
                if isinstance(schema, str):
                    schema_dict = json.loads(schema)
                elif isinstance(schema, dict):
                    schema_dict = schema
                else:
                    raise TypeError("Schema must be a dict or a JSON string.")

                system_prmpt += f"\nResponse must conform to the following JSON Schema:\n```json\n{json.dumps(schema_dict)}\n```"
                loaded_schema = schema_dict # Use the loaded dict for validation

            except (json.JSONDecodeError, TypeError) as e:
                self.logger.error(f"Invalid JSON schema provided: {e}")
                # Proceed without schema constraint in prompt, disable validation
                loaded_schema = None
                validate_schema = False
        else:
            loaded_schema = None # No schema provided

        current_query = original_query
        last_raw_response = None # Store the last raw response

        for attempt in range(max_retries + 1):
            self.logger.info(f"JSON Request - Attempt {attempt + 1}")
            response, current_history = self.inference(
                current_query,
                images=[image] if image else None,
                system_prmpt=system_prmpt,
                history=current_history, # Pass the evolving history
                seed_response="\n```json\n{"
            )
            last_raw_response = response # Update last raw response

            if not extract:
                self.logger.debug("Extraction not requested. Returning raw response.")
                return response, response # Return raw response as both results

            extracted_json = extract_json(response)

            if not extracted_json:
                self.logger.warning(f"Attempt {attempt + 1}: Failed to extract JSON from response.")
                if attempt < max_retries:
                    current_query = f"The previous response was not valid JSON or no JSON was found. Please provide a response strictly in JSON format."
                    if len(response) > 0:
                        current_query += f"Your previous response started with: '{response[:100]}...'"
                    else:
                        current_query += "Your previous response was blank"
                    continue # Retry
                else:
                    self.logger.error("Max retries reached. Failed to extract JSON.")
                    return None, last_raw_response # Failed after retries

            # JSON extracted, now validate if requested
            if not validate_schema or not loaded_schema:
                self.logger.debug(
                    "Returning extracted JSON without schema validation."
                )
                return extracted_json, last_raw_response # Success (no validation needed)

            is_valid, error_message = validate_json_schema(
                extracted_json, loaded_schema
            )

            if is_valid:
                self.logger.debug(
                    "Returning validated JSON and raw response."
                )
                return extracted_json, last_raw_response # Success (validation passed)
            else:
                self.logger.warning(
                    f"Attempt {attempt + 1}: Extracted JSON failed schema validation: {error_message}"
                )
                if attempt < max_retries:
                    current_query = f"The previous JSON response was invalid according to the schema. Error: '{error_message}'. Please correct the JSON structure and content to match the required schema."
                    continue # Retry
                else:
                    self.logger.error(
                        "Max retries reached. Failed to get schema-valid JSON."
                    )
                    return None, last_raw_response # Failed after retries

        # Should not be reached, but as a fallback
        self.logger.error("Ran out of retries. failed to get valid resposne.")
        return None, last_raw_response

def run_json_demo(
    cogVLM, image=None, schema=None, query=None, retries=0
):
    _logger.info("--- Starting JSON Demo ---")
    if not image:
        default_image_url = "https://www.contracosta.ca.gov/ImageRepository/Document?documentId=85645"
        _logger.info(f"No image provided, using default: {default_image_url}")
        image = default_image_url

    if not schema:
        _logger.info("No schema provided, using default schema.")
        schema = {
            "type": "object",
            "properties": {
                "caption": {
                    "type": "string",
                    "description": "A detailed text description of the image content.",
                },
                "ocr": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Text extracted from the image using Optical Character Recognition (OCR).",
                    },
                    "description": "A list of strings, each representing a text segment recognized in the image.",
                },
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A label identifying an object detected in the image.",
                    },
                    "description": "A list of strings, where each string is the name of an object detected in the image.",
                },
            },
            "required": ["caption", "ocr", "objects"],
            "description": "Schema for an object describing an image. It includes a text description, OCR results, and detected objects.",
        }

    if not query:
        default_query = "Write a detailed description of the image. Make sure to fill in all fields."
        _logger.info(f"No query provided, using default: '{default_query}'")
        query = default_query

    print(f"Requesting JSON for image: {image}")
    print(f"Query: {query}")
    print(f"Using Schema: {'Yes' if schema else 'No'}")
    print(f"Max Retries: {retries}")

    response, raw_response = cogVLM.request_json(
        query,
        image=image,
        extract=True,
        schema=schema,
        validate_schema=True, # Always attempt validation if schema is present
        max_retries=retries,
    )

    print("\n--- JSON Demo Results ---")
    print(f"\n[ Raw Response ]\n{raw_response}\n")
    if response:
        print(f"[ Extracted & Validated JSON ]\n{json.dumps(response, indent=2)}\n")
    else:
        print("[ Extracted & Validated JSON ]\nNone (Extraction or Validation Failed)\n")
    print("--- JSON Demo End ---")


def parse_arguments():
    parser = argparse.ArgumentParser(description="CogVLM Image Analysis Tool")

    # Main operation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--json-demo", action="store_true", help="Run the JSON demo"
    )
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode (default)", # Updated help text
    )
    mode_group.add_argument(
        "--caption",
        action="store_true",
        help="Generate a caption for an image (requires --image)",
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default="THUDM/cogvlm2-llama3-chat-19B",
        help="Path or HuggingFace ID of the model",
    )

    # Image input
    parser.add_argument(
        "--image",
        type=str,
        help="Path or URL to an image (used by --caption, --interactive, --json-demo)",
    )

    # JSON demo options
    parser.add_argument(
        "--schema",
        type=str,
        help="Path to a JSON schema file (used by --json-demo)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to send to the model (used by --caption, --json-demo, or as initial query in --interactive if image is provided)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries for JSON validation (used by --json-demo)",
    )

    # Logging options
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )

    # Chat options
    parser.add_argument(
        "--user-name",
        type=str,
        default="USER",
        help="Name for the user in chat",
    )
    parser.add_argument(
        "--assistant-name",
        type=str,
        default="ASSISTANT",
        help="Name for the assistant in chat",
    )

    return parser.parse_args()


def setup_logging(verbosity):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    # Clamp verbosity level to valid range
    level_key = max(0, min(verbosity, max(log_levels.keys())))
    level = log_levels[level_key]
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress overly verbose logs from libraries if not in DEBUG mode
    if level > logging.DEBUG:
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def main():
    args = parse_arguments()
    setup_logging(args.verbose)

    try:
        # Initialize the model
        cogVLM = CogVLM(model_path=args.model_path)

        # Determine which mode to run
        if args.json_demo:
            _logger.info("Running JSON demo mode.")
            schema_data = None
            if args.schema:
                try:
                    with open(args.schema, "r") as f:
                        schema_data = json.load(f)
                    _logger.info(f"Loaded schema from: {args.schema}")
                except (json.JSONDecodeError, FileNotFoundError, IsADirectoryError) as e:
                    _logger.error(f"Error loading schema file '{args.schema}': {e}")
                    print(f"Error: Could not load schema file '{args.schema}'. Exiting.")
                    return # Exit if schema loading fails
                except Exception as e:
                    _logger.error(f"Unexpected error loading schema file '{args.schema}': {e}")
                    print(f"Error: Could not load schema file '{args.schema}'. Exiting.")
                    return

            run_json_demo(
                cogVLM,
                image=args.image,
                schema=schema_data,
                query=args.query,
                retries=args.retries,
            )

        elif args.caption:
            _logger.info("Running caption generation mode.")
            if not args.image:
                _logger.error(
                    "Image path or URL is required for caption generation (--image)"
                )
                print("Error: --image argument is required for caption mode.")
                return # Exit if required argument is missing

            try:
                # generate_caption handles image loading internally
                caption = cogVLM.generate_caption(args.image, query=args.query)
                
                if caption:
                    print(f"\nCaption:\n{caption}\n")
                else:
                    print("\nFailed to generate caption.\n")
            except FileNotFoundError as e:
                 _logger.error(f"Error loading image for caption: {e}")
                 print(f"Error: Could not load image '{args.image}'.")
            except Exception as e:
                _logger.error(f"Error generating caption: {e}", exc_info=True)
                print("\nAn unexpected error occurred while generating the caption.\n")

        # If --interactive is specified OR if no other mode was specified, run interactive
        elif args.interactive or not (args.json_demo or args.caption):
            if args.interactive:
                _logger.info(
                    "Running interactive chat mode (specified by --interactive)."
                )
            else:
                _logger.info(
                    "No mode specified, defaulting to interactive chat mode."
                )

            chat = cogVLM.create_chat(
                user_name=args.user_name, assistant_name=args.assistant_name
            )

            # If image is provided via args, load it first
            initial_query_after_image = None
            if args.image:
                success, message = chat.open_image(args.image)
                print(message) # Print status to user
                if success and args.query:
                    # If image loaded and query provided, use query as first message
                    initial_query_after_image = args.query

            # Start the interactive loop
            chat.start_cmd_chat()

            # If there was an initial query after loading image, send it now
            # Note: This part is tricky with start_cmd_chat().
            # start_cmd_chat() blocks until exit. A better approach might be
            # to pass the initial query *into* start_cmd_chat if needed,
            # or simply let the user type the query after the image loads.
            # For simplicity, we'll rely on the user typing the query.
            # if initial_query_after_image:
            #    print(f"{chat.user_name}: {initial_query_after_image}")
            #    response = chat.chat(initial_query_after_image)
            #    print(f"{chat.assistant_name}: {response}")


    except Exception as e:
        _logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
        print(f"\nAn critical error occurred: {e}")

if __name__ == "__main__":
    main()
