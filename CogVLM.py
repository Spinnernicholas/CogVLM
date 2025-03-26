import logging
import random

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
    
    def chat(self, query):
        response, self.history = self.model.inference(
            query,
            images=[self.image] if self.image is not None else None,
            history=self.history
        )
        return response
    
    def chat_with_image(self, image, query = 'Describe what you see in the image below. Write a concise, descriptive caption at least 10 words long.'):
        self.image = image
        return self.chat(query)
    
    def start_cmd_chat(self):
        while True:
            query = input(f"{self.user_name}: ")
            if query.lower() == "exit":
                break
            if query.lower().startswith("open "):
                image_path = query[5:].strip()
                try:
                    self.image = Image.open(image_path)
                    self.image = self.image.convert("RGB")
                    print(f"Opened image: {image_path}")
                except FileNotFoundError:
                    logging.error(f"Image file not found: {image_path}")
                except Exception as e:
                    logging.error(f"Error processing image: {e}")
                continue
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

def main():
    logging.basicConfig(level=logging.DEBUG)
    cogVLM = CogVLM()

    images = [
        "https://www.contracosta.ca.gov/ImageRepository/Document?documentId=85645"
    ]

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

    query = "write a description of the image. Make sure to fill in all fields."

    for image in images[:1]:
        response, raw_response = cogVLM.request_json(
            query,
            image=image,
            extract=True,
            schema=schema,
            validate_schema=True,
            max_retries=3)

        #print(f"{image}:\njson: {extract_json(response)}\n\nRaw: {response}")
        return

if __name__ == '__main__':
    main()