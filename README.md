# CogVLM Stand Alone Script

This is a basic script you can use to run [CogVLM2](https://github.com/THUDM/CogVLM2) locally. It uses 4-bit quantization by default (on CUDA devices) to minimize VRAM requirements and provides several modes of interaction, with **interactive chat as the default mode**.

## Features ✨

1.  **Chat About Your Images! 🗣️🖼️**
    *   **Interactive Conversation:** Talk back and forth with the AI about your pictures.
    *   **Context Aware:** It remembers the chat history for follow-up questions.
    *   **Load Images Easily:** Add pictures from your computer or the web during the chat.

2.  **Get Instant Image Descriptions ✍️**
    *   **Automatic Captions:** Instantly generates text descriptions for your images.
    *   **Practical Use:** Great for summaries, alt text, and quick understanding.

3.  **Extract Specific Details into JSON 📊**
    *   **Structured JSON Output:** Get organized information (like lists of objects or text) from images formatted as JSON, which is easy for computers to read and use.
    *   **Targeted Recognition:** Ask it to find specific things like text or particular objects to include in the JSON.
    *   **Format Correction:** Automatically tries to fix the JSON output if it's not valid or doesn't match requirements.


## Quickstart

### Ubuntu

1.  Clone GitHub Repo: `git clone https://github.com/Spinnernicholas/CogVLM.git`
2.  Navigate to directory: `cd CogVLM`
3.  Create Python Environment: `python -m venv .venv`
4.  Activate Environment: `source .venv/bin/activate`
5.  Install Python Requirements: `pip install -r requirements.txt`
6.  Run Examples:
    *   **Interactive Chat (Default):** `python CogVLM.py`
    *   Interactive Chat starting with an image: `python CogVLM.py --image <path_or_url_to_image>`
    *   JSON Demo: `python CogVLM.py --json-demo`
    *   JSON Demo with specific image and 1 retry: `python CogVLM.py --json-demo --image <path_or_url_to_image> --retries 1`
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
    - [`inference(...)`](#inferenceself-query-system_prmptnone-imagesnone-historynone-max_new_tokens2048-pad_token_id128002-top_k1-user_nameuser-assistant_nameassistant-seed_response)
    - [`create_chat(self, ...)`](#create_chatself-user_namenone-assistant_namenone)
    - [`generate_caption(self, image, query=...)`](#generate_captionself-image-querydescribe-what-you-see-in-the-image-below-write-a-concise-descriptive-caption-at-least-10-words-long)
    - [`request_json(self, ...)`](#request_jsonself-query-imagenone-extractfalse-schemanone-validate_schemafalse-max_retries0)
- [Command-Line Interface](#command-line-interface)
  - [`parse_arguments()`](#parse_arguments)
  - [`setup_logging(verbosity)`](#setup_loggingverbosity)
  - [`main()`](#main)
  - [`run_json_demo(...)`](#run_json_democogvlm-imagenone-schemanone-querynone-retries0)

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

Extracts a JSON object from a string response. It first tries to load the entire string as JSON. If that fails, it searches for the first `{` and last `}` and attempts to parse the content between them.

- **Arguments:**
  - `response` (str): The string potentially containing a JSON object.
- **Returns:**
  - `dict` or `list`: The parsed JSON data if found and valid.
  - `None`: If no JSON is found or if decoding fails. Logs warnings/errors.

#### `validate_json_schema(json_data, schema)`

Validates a Python dictionary (representing JSON data) against a given JSON schema dictionary using the `jsonschema` library.

- **Arguments:**
  - `json_data` (dict): The JSON data (as a Python dictionary) to validate.
  - `schema` (dict): The JSON schema (as a Python dictionary) to validate against.
- **Returns:**
  - `tuple`: `(bool, str)`
    - `(True, None)`: If the `json_data` is valid according to the `schema`.
    - `(False, error_message)`: If validation fails, containing the error message string from `ValidationError`. Logs errors on failure.

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
  - `FileNotFoundError`: If the image cannot be loaded from the given path or URL after trying both.
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

Sends a user query to the model using the `inference` method, incorporating the current history and image context (if any). Updates the internal history with the query and the model's response.

- **Arguments:**
  - `query` (str): The user's message.
- **Returns:**
  - `str`: The model's response.

##### `open_image(self, image_path)`

Opens an image from a file path or URL using `load_image`, converts it to RGB, sets it as the current image context for the chat, and stores its path.

- **Arguments:**
  - `image_path` (str): Path or URL to the image.
- **Returns:**
  - `tuple`: `(bool, str)` where the boolean indicates success and the string provides a status message (including image dimensions). Handles `FileNotFoundError`, `requests.exceptions.RequestException`, and other potential errors during loading/processing.

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

Starts an interactive command-line chat session. This is the default mode of operation if no other mode is specified. Handles user input, commands, and model interaction within a loop.

- **Arguments:** None.
- **Returns:** None.
- **Behavior:**
    - Prompts the user for input using `user_name`.
    - Parses input starting with `/` as commands.
    - Sends non-command input to the `chat` method and prints the response.
    - Handles `EOFError` (e.g., Ctrl+D) to exit gracefully.
    - Prints error messages for unknown commands or inference errors.
- **Commands:**
    - `/help`: Show available commands.
    - `/exit`: Exit the chat session.
    - `/open [path_or_url]`: Load an image from a local path or URL.
    - `/clear`: Clear the conversation history.
    - `/image`: Show information about the currently loaded image.
    - `/reset`: Clear history and unload the current image.

#### `CogVLM`

The main class for loading and interacting with the CogVLM model.

##### `__init__(self, model_path='THUDM/cogvlm2-llama3-chat-19B')`

Initializes the `CogVLM` instance, loading the specified model and tokenizer. Configures device (CUDA if available, else CPU) and data type (`bfloat16` on compute capability >= 8, else `float16`). Uses 4-bit quantization via `BitsAndBytesConfig` by default if CUDA is available. Logs device, dtype, and loading progress.

- **Arguments:**
  - `model_path` (str, optional): The Hugging Face model identifier or local path to the CogVLM model. Defaults to `'THUDM/cogvlm2-llama3-chat-19B'`.
- **Attributes:**
  - `model_path`: The path/identifier used.
  - `logger`: Logger instance.
  - `DEVICE`: The device ('cuda' or 'cpu').
  - `TORCH_TYPE`: The torch data type (`torch.bfloat16` or `torch.float16`).
  - `tokenizer`: The loaded `AutoTokenizer` instance.
  - `model`: The loaded `AutoModelForCausalLM` instance (potentially quantized).

##### `inference(self, query, system_prmpt=None, images=None, history=None, max_new_tokens=2048, pad_token_id=128002, top_k=1, user_name='USER', assistant_name='ASSISTANT', seed_response="")`

Performs inference using the loaded CogVLM model, handling text, optional images, and conversation history.

- **Arguments:**
  - `query` (str): The main text query or prompt for the current turn.
  - `system_prmpt` (str, optional): A system prompt to prepend to the conversation context. Defaults to `None`.
  - `images` (list, optional): A list containing image sources (paths, URLs, or PIL Images). **Only the first image is used** if multiple are provided. Defaults to `None`.
  - `history` (list, optional): A list of `(name, message)` tuples representing the conversation history *before* the current turn. If `None`, a new history is started. The provided list is modified in place by appending the current user query and the assistant's response. Defaults to `None`.
  - `max_new_tokens` (int, optional): Maximum number of new tokens to generate in the response. Defaults to `2048`.
  - `pad_token_id` (int, optional): Token ID for padding during generation. Defaults to `128002`.
  - `top_k` (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering during generation. Defaults to `1`.
  - `user_name` (str, optional): Name tag for the user in the current turn. Defaults to `'USER'`.
  - `assistant_name` (str, optional): Name tag for the assistant in the current turn. Defaults to `'ASSISTANT'`.
  - `seed_response` (str, optional): A string to prepend to the model's generated output. This string is also included after the `ASSISTANT:` tag when building the input prompt. Defaults to `""`.
- **Returns:**
  - `tuple`: `(response, history)`
    - `response` (str): The generated text response from the model (with `seed_response` prepended), stripped of EOS tokens, or an error message string on failure.
    - `history` (list): The updated conversation history including the latest user query and the assistant's response (or the history *before* the failed turn if an exception occurred).
- **Notes:**
  - Handles image loading (`load_image`) and RGB conversion if `images` are provided. Logs a warning if multiple images are given.
  - Appends the current `(user_name, query)` to the `history` list *before* inference.
  - Formats the input prompt string including system prompt (if any), all history turns, and the current assistant tag (`f"{assistant_name}:{seed_response}"`).
  - Builds model inputs using `model.build_conversation_input_ids` (handles text and image token interleaving).
  - Runs generation using `model.generate` within `torch.no_grad()`.
  - Decodes the generated output tokens, prepends `seed_response`, and cleans the result.
  - Appends the successful `(assistant_name, response)` to the `history` list *after* successful inference.
  - Includes error handling and logging for image processing and model inference steps.

##### `create_chat(self, user_name=None, assistant_name=None)`

Factory method to create a `CogVLMChat` instance associated with this `CogVLM` model.

- **Arguments:**
  - `user_name` (str, optional): User name for the chat session. Defaults to `'USER'`.
  - `assistant_name` (str, optional): Assistant name for the chat session. Defaults to `'ASSISTANT'`.
- **Returns:**
  - `CogVLMChat`: A new chat session instance.

##### `generate_caption(self, image, query='Describe what you see in the image below. Write a concise, descriptive caption at least 10 words long.')`

A convenience method to generate a caption for a single image by calling the `inference` method with no history (used by `--caption` mode).

- **Arguments:**
  - `image` (Union[str, `PIL.Image.Image`]): The image to caption.
  - `query` (str, optional): The prompt used to request the caption. Defaults to a descriptive prompt asking for at least 10 words.
- **Returns:**
  - `str`: The generated caption.
  - `None`: If the input `image` is `None` or an error occurs during inference.

##### `request_json(self, query, image=None, extract=False, schema=None, validate_schema=False, max_retries=0)`

Requests a response from the model, specifically aiming for JSON output. Optionally extracts the JSON, validates it against a schema, and retries with feedback on failure (used by `--json-demo` mode).

- **Arguments:**
  - `query` (str): The user's query, intended to elicit a JSON response.
  - `image` (Union[str, `PIL.Image.Image`], optional): An image to provide context. Defaults to `None`.
  - `extract` (bool, optional): If `True`, attempts to extract JSON from the response using `extract_json`. Defaults to `False`.
  - `schema` (dict or str, optional): A JSON schema (as a dictionary or JSON string) to validate against if `validate_schema` is `True`. If provided, it's also added to the system prompt to guide the model. Defaults to `None`.
  - `validate_schema` (bool, optional): If `True` (and `extract` is `True` and `schema` is provided), validates the extracted JSON against the schema using `validate_json_schema`. Automatically disabled with a warning if `extract` is `False` or `schema` is not provided. Defaults to `False`.
  - `max_retries` (int, optional): The number of times to retry if JSON extraction or validation fails. Defaults to `0`.
- **Returns:**
  - `tuple`: `(result, raw_response)`
    - `result`:
      - If `extract` is `False`: The raw string response from the model.
      - If `extract` is `True` and successful (and validation passes, if enabled): The extracted JSON data (dict or list).
      - If `extract` is `True` but fails after `max_retries`: `None`.
    - `raw_response` (str): The final raw string response received from the model during the last attempt.
- **Notes:**
  - Sets a system prompt: "You are a helpful assistant that responds in a valid JSON format."
  - If a valid `schema` is provided, it's loaded (if string) and appended to the system prompt within a JSON code block. Invalid schemas disable validation.
  - Calls `inference` with the constructed system prompt and `seed_response="\n```json\n{"` to encourage JSON output.
  - Maintains conversation history across retries.
  - Implements a retry loop:
    - If extraction fails, retries with a modified query asking for strict JSON format, potentially including the start of the previous invalid response.
    - If validation fails, retries with a modified query including the schema validation error message, asking for correction.
  - Logs warnings/errors during extraction, validation, and retries.

### Command-Line Interface

The script uses `argparse` to handle command-line arguments, allowing users to select different modes of operation and configure settings.

#### `parse_arguments()`

Parses command-line arguments using `argparse`.

- **Returns:**
  - `argparse.Namespace`: An object containing the parsed arguments.
- **Arguments:**
    - Mode Selection (mutually exclusive):
        - `--json-demo`: Run the JSON demo.
        - `--interactive`: Start interactive chat mode. **This is the default behavior if no mode is specified.**
        - `--caption`: Generate a caption for an image (requires `--image`).
    - Configuration:
        - `--model-path`: Path or HuggingFace ID of the model (default: `THUDM/cogvlm2-llama3-chat-19B`).
        - `--image`: Path or URL to an image (used by `--caption`, `--interactive`, `--json-demo`).
        - `--schema`: Path to a JSON schema file (used by `--json-demo`).
        - `--query`: Query to send to the model (used by `--caption`, `--json-demo`).
        - `--retries`: Number of retries for JSON extraction/validation (used by `--json-demo`, default: `0`).
        - `--verbose`/`-v`: Increase logging verbosity (0: WARNING, 1: INFO, 2: DEBUG, default: 0).
        - `--user-name`: Name for the user in chat (default: `USER`).
        - `--assistant-name`: Name for the assistant in chat (default: `ASSISTANT`).

#### `setup_logging(verbosity)`

Configures the root logger based on the verbosity level provided by command-line arguments (`--verbose`). Suppresses overly verbose logs from dependencies unless verbosity is set to DEBUG (`-vv`).

- **Arguments:**
  - `verbosity` (int): The level of verbosity (0: WARNING, 1: INFO, 2: DEBUG).

#### `main()`

The main entry point of the script.
1.  Parses arguments using `parse_arguments()`.
2.  Sets up logging using `setup_logging()`.
3.  Initializes the `CogVLM` model.
4.  Runs the selected mode:
    *   If `args.json_demo`: Loads schema (if provided), then calls `run_json_demo()`. Handles schema file loading errors.
    *   If `args.caption`: Checks for required `--image` argument, then calls `cogVLM.generate_caption()` and prints the result. Handles image loading errors.
    *   If `args.interactive` OR if no other mode was specified: Creates a `CogVLMChat` instance, optionally loads the initial image specified by `--image`, and starts the interactive loop by calling `chat.start_cmd_chat()`.
5.  Includes top-level exception handling to catch and log unexpected errors.

#### `run_json_demo(cogVLM, image=None, schema=None, query=None, retries=0)`

Runs the JSON demonstration mode **when explicitly requested via the `--json-demo` argument**. It uses default values for image, schema, and query if none are provided via command-line arguments or if loading fails. It calls `cogVLM.request_json` with extraction and validation enabled, using the specified number of retries. Prints the raw response and the final extracted/validated JSON (or None if failed).

- **Arguments:**
  - `cogVLM` (`CogVLM`): The initialized CogVLM model instance.
  - `image` (str, optional): Path or URL to the image. Uses a default URL if not provided.
  - `schema` (dict, optional): The JSON schema to use for validation. Uses a default schema if not provided.
  - `query` (str, optional): The query to send to the model. Uses a default query if not provided.
  - `retries` (int, optional): Number of retries for JSON extraction/validation. Defaults to `0`.
