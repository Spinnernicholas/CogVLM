# CogVLM Stand Alone Script

This is a basic script you can use to run [CogVLM2](https://github.com/THUDM/CogVLM2) locally. It uses 4-bit quantization by default to minimize VRAM requirements and provides several modes of interaction.

## Features

*   **Multiple Modes:**
    *   Interactive Chat (`--interactive`)
    *   Image Captioning (`--caption`)
    *   JSON Output Demo (`--json-demo`, default)
*   **Interactive Chat:**
    *   Chat History
    *   Load images during chat (`/open`)
    *   View image info (`/image`)
    *   Clear history (`/clear`)
    *   Reset chat and image (`/reset`)
    *   Customizable user/assistant names
    *   Help command (`/help`)
*   **JSON Features:**
    *   Prompt for JSON output.
    *   Extract JSON from model response.
    *   Validate JSON against a provided schema.
    *   Automatic retries with feedback on validation failure.
*   **Configuration:**
    *   Specify model path (`--model-path`).
    *   Control verbosity (`-v`, `-vv`).
    *   Provide image via command line (`--image`).
    *   Provide query via command line (`--query`).

## Quickstart

### Ubuntu

1.  Clone GitHub Repo: `git clone https://github.com/Spinnernicholas/CogVLM.git`
2.  Navigate to directory: `cd CogVLM`
3.  Create Python Environment: `python -m venv .venv`
4.  Activate Environment: `source .venv/bin/activate`
5.  Install Python Requirements: `pip install -r requirements.txt`
6.  Run Examples:
    *   JSON Demo (Default): `python CogVLM.py`
    *   JSON Demo with specific image: `python CogVLM.py --image <path_or_url_to_image>`
    *   Interactive Chat: `python CogVLM.py --interactive`
    *   Interactive Chat starting with an image: `python CogVLM.py --interactive --image <path_or_url_to_image>`
    *   Generate Caption: `python CogVLM.py --caption --image <path_or_url_to_image>`

## Note on Transformers Library Versions

Modifications to the base model classes in [Transformers](https://github.com/huggingface/transformers)>=v4.49.0 broke the [CogVLM2](https://github.com/THUDM/CogVLM2) model classes. This script has been tested and works with [Transformers](https://github.com/huggingface/transformers)==v4.48.3. Ensure you have the correct version installed (as specified in `requirements.txt`).

## Code Reference: CogVLM Interaction Utilities

This script provides utilities for interacting with the CogVLM (Cognitive Visual Language Model), including image loading, chat functionalities, JSON extraction/validation, model inference, and a command-line interface.

### Table of Contents

- [Dependencies](#dependencies)
- [Helper Functions](#helper-functions)
  - [`extract_json(response)`](#extract_jsonresponse)
  - [`validate_json_schema(json_data, schema)`](#validate_json_schemajson_data-schema)
  - [`get_random_examples(data, num_examples=3)`](#get_random_examplesdata-num_examples3)
  - [`load_image(image)`](#load_imageimage)
- [Classes](#classes)
  - [`CogVLMChat`](#cogvlmchat)
    - [`__init__(...)`](#__initself-model-user_nameuser-assistant_nameassistant)
    - [`chat(self, query)`](#chatself-query)
    - [`open_image(self, image_path)`](#open_imageself-image_path)
    - [`get_image_info(self)`](#get_image_infoself)
    - [`clear_history(self)`](#clear_historyself)
    - [`reset(self)`](#resetself)
    - [`start_cmd_chat(self)`](#start_cmd_chatself)
  - [`CogVLM`](#cogvlm)
    - [`__init__(...)`](#__initself-model_paththudmcogvlm2-llama3-chat-19b)
    - [`inference(...)`](#inferenceself-query-system_prmptnone-imagesnone-historynone-max_new_tokens2048-pad_token_id128002-top_k1-user_nameuser-assistant_nameassistant)
    - [`create_chat(self, ...)`](#create_chatself-user_namenone-assistant_namenone)
    - [`generate_caption(self, image, query=...)`](#generate_captionself-image-querydescribe-what-you-see-in-the-image-below-write-a-concise-descriptive-caption-at-least-10-words-long)
    - [`request_json(self, ...)`](#request_jsonself-query-imagenone-extractfalse-schemanone-validate_schemafalse-max_retries0)
- [Command-Line Interface](#command-line-interface)
  - [`parse_arguments()`](#parse_arguments)
  - [`setup_logging(verbosity)`](#setup_loggingverbosity)
  - [`main()`](#main)
  - [`run_json_demo(...)`](#run_json_democogvlm-imagenone-schemanone-querynone-retries3)

### Dependencies

- `logging`
- `random`
- `argparse`
- `json`
- `jsonschema`
- `PIL` (Pillow)
- `io`
- `typing`
- `numpy`
- `requests`
- `torch`
- `transformers` (`AutoModelForCausalLM`, `AutoTokenizer`, `BitsAndBytesConfig`)

### Helper Functions

#### `extract_json(response)`

Extracts a JSON object from a string response, attempting to parse the whole string first, then searching for the first `{` and last `}`.

- **Arguments:**
  - `response` (str): The string potentially containing a JSON object.
- **Returns:**
  - `dict` or `list`: The parsed JSON data if found and valid.
  - `None`: If no JSON is found or if decoding fails. Logs warnings/errors.

#### `validate_json_schema(json_data, schema)`

Validates a Python dictionary (representing JSON data) against a given JSON schema dictionary.

- **Arguments:**
  - `json_data` (dict): The JSON data (as a Python dictionary) to validate.
  - `schema` (dict): The JSON schema (as a Python dictionary) to validate against.
- **Returns:**
  - `tuple`: `(bool, str)`
    - `(True, None)`: If the `json_data` is valid according to the `schema`.
    - `(False, error_message)`: If validation fails, containing the error message string. Logs errors on failure.

#### `get_random_examples(data, num_examples=3)`

Selects a specified number of random examples from a list. (Note: Currently not used by the main script modes).

- **Arguments:**
  - `data` (list): The list of items to sample from.
  - `num_examples` (int, optional): The number of random examples to return. Defaults to `3`.
- **Returns:**
  - `list`: A list containing `num_examples` randomly selected items from `data`. Returns an empty list and logs a warning if `num_examples` is greater than the length of `data`.

#### `load_image(image)`

Loads an image from various sources into a PIL Image object.

- **Arguments:**
  - `image` (Union[str, `PIL.Image.Image`]): The image source. Can be:
    - A local file path (string).
    - A URL (string).
    - An existing `PIL.Image.Image` object.
- **Returns:**
  - `PIL.Image.Image`: The loaded image object.
- **Raises:**
  - `FileNotFoundError`: If the image cannot be loaded from the given path or URL.
  - `TypeError`: If the input `image` is not a string or a `PIL.Image.Image` object.

### Classes

#### `CogVLMChat`

Provides a stateful chat interface wrapper around a `CogVLM` instance, managing conversation history and image context.

##### `__init__(self, model, user_name='USER', assistant_name='ASSISTANT')`

Initializes the chat session.

- **Arguments:**
  - `model` (`CogVLM`): An instance of the `CogVLM` class to use for inference.
  - `user_name` (str, optional): The name representing the user in the chat history. Defaults to `'USER'`.
  - `assistant_name` (str, optional): The name representing the assistant in the chat history. Defaults to `'ASSISTANT'`.
- **Attributes:**
  - `model`: The associated `CogVLM` instance.
  - `user_name`: User's name tag.
  - `assistant_name`: Assistant's name tag.
  - `history`: A list storing the conversation history as `(name, message)` tuples.
  - `image`: The current `PIL.Image.Image` context for the chat (or `None`).
  - `image_path`: The path or URL of the currently loaded image (or `None`).

##### `chat(self, query)`

Sends a user query to the model, using the current history and image context (if any), and updates the history.

- **Arguments:**
  - `query` (str): The user's message.
- **Returns:**
  - `str`: The model's response.

##### `open_image(self, image_path)`

Opens an image from a file path or URL, sets it as the current image context for the chat, and stores its path.

- **Arguments:**
  - `image_path` (str): Path or URL to the image.
- **Returns:**
  - `tuple`: `(bool, str)` where the boolean indicates success and the string provides a status message.

##### `get_image_info(self)`

Returns information about the currently loaded image, including its dimensions, mode, and path/URL.

- **Arguments:** None.
- **Returns:**
  - `str`: A string containing image information or "No image is currently loaded".

##### `clear_history(self)`

Clears the chat conversation history.

- **Arguments:** None.
- **Returns:**
  - `str`: A confirmation message "Chat history cleared".

##### `reset(self)`

Resets the chat session by clearing both the conversation history and the currently loaded image context (including its path).

- **Arguments:** None.
- **Returns:**
  - `str`: A confirmation message "Chat history and image have been reset".

##### `start_cmd_chat(self)`

Starts an interactive command-line chat session.

- **Arguments:** None.
- **Returns:** None.
- **Commands:**
    - `/help`: Show available commands.
    - `/exit`: Exit the chat session.
    - `/open [path_or_url]`: Load an image from a local path or URL.
    - `/clear`: Clear the conversation history.
    - `/image`: Show information about the currently loaded image.
    - `/reset`: Clear history and unload the current image.
    - Any other input is treated as a chat message to the model.

#### `CogVLM`

The main class for loading and interacting with the CogVLM model.

##### `__init__(self, model_path='THUDM/cogvlm2-llama3-chat-19B')`

Initializes the `CogVLM` instance, loading the specified model and tokenizer. Configures device (CUDA if available) and data type. Uses 4-bit quantization by default if CUDA is available and compatible.

- **Arguments:**
  - `model_path` (str, optional): The Hugging Face model identifier or local path to the CogVLM model. Defaults to `'THUDM/cogvlm2-llama3-chat-19B'`.
- **Attributes:**
  - `model_path`: The path/identifier used.
  - `logger`: Logger instance.
  - `DEVICE`: The device ('cuda' or 'cpu').
  - `TORCH_TYPE`: The torch data type (`torch.bfloat16` or `torch.float16`).
  - `tokenizer`: The loaded tokenizer instance.
  - `model`: The loaded model instance (potentially quantized).

##### `inference(self, query, system_prmpt=None, images=None, history=None, max_new_tokens=2048, pad_token_id=128002, top_k=1, user_name='USER', assistant_name='ASSISTANT')`

Performs inference using the loaded CogVLM model.

- **Arguments:**
  - `query` (str): The main text query or prompt.
  - `system_prmpt` (str, optional): A system prompt to prepend to the conversation. Defaults to `None`.
  - `images` (list, optional): A list containing image sources (paths, URLs, or PIL Images). Currently, only the first image is used if multiple are provided. Defaults to `None`.
  - `history` (list, optional): A list of `(name, message)` tuples representing the conversation history. Defaults to `None` (starts a new conversation).
  - `max_new_tokens` (int, optional): Maximum number of tokens to generate. Defaults to `2048`.
  - `pad_token_id` (int, optional): Token ID for padding. Defaults to `128002`.
  - `top_k` (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to `1`.
  - `user_name` (str, optional): Name tag for the user in the current turn. Defaults to `'USER'`.
  - `assistant_name` (str, optional): Name tag for the assistant in the current turn. Defaults to `'ASSISTANT'`.
- **Returns:**
  - `tuple`: `(response, history)`
    - `response` (str): The generated text response from the model.
    - `history` (list): The updated conversation history including the latest query and response.
- **Notes:**
  - Handles image loading and preprocessing if `images` are provided.
  - Formats the input prompt including system prompt, history, and current query.
  - Builds model inputs using `model.build_conversation_input_ids`.
  - Runs generation using `model.generate`.
  - Decodes the output tokens.

##### `create_chat(self, user_name=None, assistant_name=None)`

Factory method to create a `CogVLMChat` instance associated with this `CogVLM` model.

- **Arguments:**
  - `user_name` (str, optional): User name for the chat session. Defaults to `'USER'`.
  - `assistant_name` (str, optional): Assistant name for the chat session. Defaults to `'ASSISTANT'`.
- **Returns:**
  - `CogVLMChat`: A new chat session instance.

##### `generate_caption(self, image, query='Describe what you see in the image below. Write a concise, descriptive caption at least 10 words long.')`

A convenience method to generate a caption for a single image.

- **Arguments:**
  - `image` (Union[str, `PIL.Image.Image`]): The image to caption.
  - `query` (str, optional): The prompt used to request the caption. Defaults to a descriptive prompt.
- **Returns:**
  - `str`: The generated caption.
  - `None`: If the input `image` is `None`.

##### `request_json(self, query, image=None, extract=False, schema=None, validate_schema=False, max_retries=0)`

Requests a response from the model, optionally extracting JSON, validating it against a schema, and retrying on failure.

- **Arguments:**
  - `query` (str): The user's query, intended to elicit a JSON response.
  - `image` (Union[str, `PIL.Image.Image`], optional): An image to provide context. Defaults to `None`.
  - `extract` (bool, optional): If `True`, attempts to extract JSON from the response using `extract_json`. Defaults to `False`.
  - `schema` (dict or str, optional): A JSON schema (as a dictionary or JSON string) to validate against if `validate_schema` is `True`. Also used to guide the model via the system prompt. Defaults to `None`.
  - `validate_schema` (bool, optional): If `True` (and `extract` is `True` and `schema` is provided), validates the extracted JSON against the schema using `validate_json_schema`. Defaults to `False`.
  - `max_retries` (int, optional): The number of times to retry if JSON extraction or validation fails. The model is prompted to correct its previous invalid response on retries. Defaults to `0`.
- **Returns:**
  - `tuple`: `(result, raw_response)`
    - `result`:
      - If `extract` is `False`: The raw string response from the model.
      - If `extract` is `True` and successful (and validation passes, if enabled): The extracted JSON data (dict or list).
      - If `extract` is `True` but fails after retries: `None`.
    - `raw_response` (str): The final raw string response received from the model.
- **Notes:**
  - Sets a system prompt instructing the model to respond in JSON format.
  - If a `schema` is provided, it's added to the system prompt.
  - Implements a retry loop if extraction/validation fails, providing feedback to the model.

### Command-Line Interface

The script uses `argparse` to handle command-line arguments, allowing users to select different modes of operation and configure settings.

#### `parse_arguments()`

Parses command-line arguments using `argparse`.

- **Returns:**
  - `argparse.Namespace`: An object containing the parsed arguments.
- **Arguments:**
    - `--json-demo`: Run the JSON demo (default if no other mode is specified).
    - `--interactive`: Start interactive chat mode.
    - `--caption`: Generate a caption for an image (requires `--image`).
    - `--model-path`: Path to the model (default: `THUDM/cogvlm2-llama3-chat-19B`).
    - `--image`: Path or URL to an image (used by `--caption`, `--interactive`, `--json-demo`).
    - `--schema`: Path to a JSON schema file (used by `--json-demo`).
    - `--query`: Query to send to the model (used by `--caption`, `--json-demo`).
    - `--retries`: Number of retries for JSON validation (used by `--json-demo`, default: 3).
    - `--verbose`/`-v`: Increase logging verbosity (can use `-v` or `-vv`).
    - `--user-name`: Name for the user in chat (default: `USER`).
    - `--assistant-name`: Name for the assistant in chat (default: `ASSISTANT`).

#### `setup_logging(verbosity)`

Configures the root logger based on the verbosity level provided by command-line arguments.

- **Arguments:**
  - `verbosity` (int): The level of verbosity (0: WARNING, 1: INFO, 2+: DEBUG).

#### `main()`

The main entry point of the script. It parses arguments, sets up logging, initializes the `CogVLM` model, and runs the selected mode (`json-demo`, `interactive`, or `caption`).

#### `run_json_demo(cogVLM, image=None, schema=None, query=None, retries=3)`

Runs the JSON demonstration mode. It uses default values for image, schema, and query if none are provided. It calls `cogVLM.request_json` with extraction and validation enabled.

- **Arguments:**
  - `cogVLM` (`CogVLM`): The initialized CogVLM model instance.
  - `image` (str, optional): Path or URL to the image.
  - `schema` (dict, optional): The JSON schema to use for validation.
  - `query` (str, optional): The query to send to the model.
  - `retries` (int, optional): Number of retries for validation.
