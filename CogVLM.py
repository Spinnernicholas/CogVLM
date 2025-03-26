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
            end_index = response.rfind("}")    # Find the last closing brace
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
        logging.getLogger(__name__).warning("Warning: num_examples is greater than the length of the data. Returning empty list.")
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
                raise FileNotFoundError(f"Could not load image from path or URL: {image}")
    else:
        raise TypeError(
        "Expected a file path, URL, or PIL Image object, "
        f"but got {type(image)}"
        )

class CogVLMChat:
    def __init__(self, model, user_name = 'USER', assistant_name = 'ASSISTANT'):
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
            history=self.history
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
            return True, f"Successfully opened image: {image_path} ({img.size[0]}x{img.size[1]})"
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
        path_info = f"Path: {self.image_path}" if self.image_path else "Path: Unknown (loaded directly)"
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
            "/reset": "Reset the image and chat history"
        }
        
        while True:
            query = input(f"{self.user_name}: ")
            
            # Handle commands (starting with /)
            if query.startswith('/'):
                cmd_parts = query.split(' ', 1)
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
                    print(f"Unknown command: {cmd}. Type /help for available commands.")
                    continue
            
            # Regular chat message
            print(f"{self.assistant_name}: {self.chat(query)}")

class CogVLM:
    def __init__(self, model_path = 'THUDM/cogvlm2-llama3-chat-19B'):
        self.model_path = model_path
        #Max text tokens including response: 8192

        self.logger = logging.getLogger(__name__)

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16
        self.logger.debug(f" Device: {self.DEVICE}, Torch type: {self.TORCH_TYPE}")

        self.logger.debug(f"Loading model {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.TORCH_TYPE,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            low_cpu_mem_usage=True
        ).eval()
    
    def inference(
            self,
            query,
            system_prmpt = None,
            images = None,
            history = None,
            max_new_tokens = 2048,
            pad_token_id = 128002,
            top_k = 1,
            user_name = 'USER',
            assistant_name = 'ASSISTANT'):
        self.logger.debug(f"Running with args: query: {query}, images: {images}, system_prmpt: {system_prmpt}, history: {history}, max_new_tokens: {max_new_tokens}, pad_token_id: {pad_token_id}, top_k: {top_k}\n\n")

        if images is not None:
            if len(images) > 1:
                self.logger.warning("Only one image is supported at the moment, using the first image.")

            try:
                image = load_image(images[0])
            except Exception as e:
                self.logger.error("Unable to load image. Error: " + e)
                return

            image = image.convert('RGB')
            images = [image]

        if history is None:
            history = []

        history.append((user_name, query))

        old_prompt = "" if system_prmpt is None else system_prmpt + "\n"
        for _, (name, message) in enumerate(history):
            old_prompt += f"{name}: {message}\n"
        query = old_prompt + f"{assistant_name}:"

        self.logger.debug(f"Raw Query: {query}\n\n")

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=history,
            images=images,
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
            "top_k": top_k,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]

        history.append((assistant_name, response))

        self.logger.debug(f"Raw Response: {response}\n\n")
        return response, history
    
    def create_chat(self, user_name = None, assistant_name = None):
        return CogVLMChat(self, user_name=user_name, assistant_name=assistant_name)

    def generate_caption(self, image, query = 'Describe what you see in the image below. Write a concise, descriptive caption at least 10 words long.'):
        if image is None:
            self.logger.error("Image is None")
            return None
        return self.inference(query, images=[image])[0]
    
    def request_json(self, query, image=None, extract=False, schema=None, validate_schema=False, max_retries=0):
        if validate_schema and not extract:
            validate_schema = False
            self.logger.warning("validate_schema can only be True if extract is True, will attempt to return JSON without schema validation.")
        if validate_schema and not schema:
            validate_schema = False
            self.logger.warning("validate_schema can only be true if schema is provided, will attempt to return JSON without schema validation.")

        history = []
        system_prmpt = 'You are a helpful assistant that responds in a json format.'

        if schema:
            try:
                if isinstance(schema, str):
                    schema = json.loads(schema)

                system_prmpt += f"\n Response must conform to the following JSON Schema:\n{json.dumps(schema)}"

            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON schema: {e}")
                system_prmpt += "\nInvalid JSON Schema provided, will attempt to return JSON without schema validation."  # Still try to get JSON

        for attempt in range(max_retries + 1):
            response, history = self.inference(query, images=[image], system_prmpt=system_prmpt, history=history)

            if not extract:
                self.logger.debug("Returning raw response.")
                return response, response
            
            extracted_json = extract_json(response)

            if not extracted_json or len(extracted_json) == 0:
                query = f"The previous JSON response syntax was invalid. Here was your \nprevious response: \"{extracted_json}\""
                continue

            if not validate_schema:
                self.logger.debug("Returning extracted JSON without validation and raw response")
                return extracted_json, response
            
            is_valid, error_message = validate_json_schema(extracted_json, schema)

            if is_valid:
                self.logger.debug("Returning validated JSON and raw response.")
                return extracted_json, response
            
            self.logger.warning(f"Attempt {attempt + 1}: Extracted JSON does not validate against schema: {error_message}")
            query = f"The previous JSON response was invalid.  Error: {error_message}.\nPlease correct the JSON to match the schema."
        
        self.logger.error(f"Max retries reached. Returning None and raw response.")
        return None, response

def run_json_demo(
    cogVLM,
    image=None,
    schema=None,
    query=None,
    retries=3
):
    if not image:
        image = "https://www.contracosta.ca.gov/ImageRepository/Document?documentId=85645"

    if not schema:
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
        "description": "Schema for an object describing an image.  It includes a text description, OCR results, and detected objects.",
    }
        
    if not query:
        query = "write a description of the image. Make sure to fill in all fields."

    response, raw_response = cogVLM.request_json(
        query,
        image=image,
        extract=True,
        schema=schema,
        validate_schema=True,
        max_retries=retries)
    
    print(f"\n[ Raw Response ]\n{raw_response}\n\n[ JSON Response ]\n{response}\n\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description='CogVLM Image Analysis Tool')
    
    # Main operation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--json-demo', action='store_true', help='Run the JSON demo')
    mode_group.add_argument('--interactive', action='store_true', help='Start interactive chat mode')
    mode_group.add_argument('--caption', action='store_true', help='Generate a caption for an image')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, default='THUDM/cogvlm2-llama3-chat-19B', 
                        help='Path to the model')
    
    # Image input
    parser.add_argument('--image', type=str, help='Path or URL to an image')
    
    # JSON demo options
    parser.add_argument('--schema', type=str, help='JSON schema file path')
    parser.add_argument('--query', type=str, help='Query to send to the model')
    parser.add_argument('--retries', type=int, default=3, help='Number of retries for JSON validation')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='count', default=0, 
                        help='Increase verbosity (can be used multiple times)')
    
    # Chat options
    parser.add_argument('--user-name', type=str, default='USER', help='Name for the user in chat')
    parser.add_argument('--assistant-name', type=str, default='ASSISTANT', help='Name for the assistant in chat')
    
    return parser.parse_args()

def setup_logging(verbosity):
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    # If verbosity is higher than the max level, use the highest level
    level = log_levels.get(min(verbosity, max(log_levels.keys())), logging.DEBUG)
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    
    # Initialize the model
    cogVLM = CogVLM(model_path=args.model_path)
    
    # Determine which mode to run
    if args.json_demo:
        # Load schema from file if provided
        schema_data = None
        if args.schema:
            try:
                with open(args.schema, 'r') as f:
                    schema_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                _logger.error(f"Error loading schema file: {e}")
                return
        
        run_json_demo(
            cogVLM,
            image=args.image,
            schema=schema_data,
            query=args.query,
            retries=args.retries
        )
    
    elif args.caption:
        if not args.image:
            _logger.error("Image path or URL is required for caption generation")
            return
        
        try:
            image = load_image(args.image)
            caption = cogVLM.generate_caption(image, args.query if args.query else None)
            print(f"Caption: {caption}")
        except Exception as e:
            _logger.error(f"Error generating caption: {e}")
    
    elif args.interactive:
        chat = cogVLM.create_chat(user_name=args.user_name, assistant_name=args.assistant_name)
        
        # If image is provided, load it first
        if args.image:
            success, message = chat.open_image(args.image)
            print(message)

        chat.start_cmd_chat()
    
    else:
        # Default to JSON demo if no mode specified
        run_json_demo(cogVLM)

if __name__ == '__main__':
    main()
