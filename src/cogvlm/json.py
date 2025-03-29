import logging
import json
import re # Added for placeholder extract_json
from typing import Any, Dict, Optional, Tuple, Union, List, Iterator, Callable

# Attempt to import ICogVLM from core, provide dummy if not found
# Note: In a real scenario, this dependency handling might be different
# depending on how json.py is intended to be used (standalone vs part of cogvlm)
try:
    from cogvlm.core import ICogVLM
    # Import Image type if needed for type hints, assuming it's used similarly
    from PIL import Image
except ImportError:
    logging.warning("Could not import ICogVLM/Image from cogvlm.core/PIL, using dummy types.")
    from abc import ABC, abstractmethod
    class ICogVLM(ABC):
        @abstractmethod
        def inference(
            self,
            query: str,
            system_prmpt: Optional[str] = None,
            images: Optional[List[Union[str, Any]]] = None, # Use Any for dummy Image
            history: Optional[List[Tuple[str, str]]] = None,
            max_new_tokens: int = 2048,
            top_k: int = 1,
            user_name: str = "USER",
            assistant_name: str = "ASSISTANT",
            seed_response: str = "",
            stream: bool = False,
        ) -> Union[Tuple[str, List[Tuple[str, str]]], Tuple[Iterator[str], Callable]]:
            raise NotImplementedError
    # Dummy Image type if PIL isn't available where this is run
    class Image:
        Image = type("DummyPILImage", (), {})


_logger = logging.getLogger(__name__)

# --- Existing Helper Functions (extract_json, validate_json_schema) ---

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Placeholder: Attempts to extract the first JSON object found within ```json ... ``` blocks.
    A more robust implementation is recommended.
    """
    # Simple regex to find ```json ... ``` blocks
    # Handle potential whitespace and newlines around the JSON content
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            _logger.warning(f"Failed to decode JSON within ```json block: {e}")
            # Fall through to attempt finding JSON outside blocks

    # Fallback: Try finding any JSON object starting with { and ending with }
    # This is less reliable and might grab incorrect structures.
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            potential_json = text[start:end+1]
            # Basic sanity check: does it look like JSON?
            if potential_json.count('{') >= potential_json.count('}') and potential_json.count('[') >= potential_json.count(']'):
                 return json.loads(potential_json)
        return None
    except json.JSONDecodeError:
        return None


def validate_json_schema(
    instance: Dict[str, Any], schema: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Placeholder: Validates JSON against a schema.
    Use the 'jsonschema' library for a proper implementation.
    Example: pip install jsonschema
    """
    try:
        # --- Placeholder Logic ---
        # Example using jsonschema (install first: pip install jsonschema)
        # try:
        #     from jsonschema import validate, ValidationError
        #     validate(instance=instance, schema=schema)
        #     logging.info("jsonschema validation successful.")
        #     return True, None
        # except ValidationError as e:
        #     logging.warning(f"jsonschema validation failed: {e.message}")
        #     return False, e.message
        # except ImportError:
        #     logging.warning("jsonschema library not found, using basic placeholder validation.")
        #     # Fall through to basic placeholder if jsonschema is not installed

        # --- Basic Placeholder Validation (if jsonschema isn't used) ---
        if not isinstance(instance, dict):
             return False, "Instance is not a dictionary."
        required_props = schema.get("required", [])
        for prop in required_props:
            if prop not in instance:
                return False, f"Missing required property: {prop}"

        properties = schema.get("properties", {})
        for key, rules in properties.items():
            if key in instance:
                expected_type = rules.get("type")
                actual_value = instance[key]
                # Very basic type check
                if expected_type == "string" and not isinstance(actual_value, str):
                    return False, f"Property '{key}' should be a string, got {type(actual_value).__name__}"
                if expected_type == "number" and not isinstance(actual_value, (int, float)):
                     return False, f"Property '{key}' should be a number, got {type(actual_value).__name__}"
                if expected_type == "integer" and not isinstance(actual_value, int):
                     return False, f"Property '{key}' should be an integer, got {type(actual_value).__name__}"
                if expected_type == "boolean" and not isinstance(actual_value, bool):
                     return False, f"Property '{key}' should be a boolean, got {type(actual_value).__name__}"
                if expected_type == "array" and not isinstance(actual_value, list):
                     return False, f"Property '{key}' should be an array, got {type(actual_value).__name__}"
                if expected_type == "object" and not isinstance(actual_value, dict):
                     return False, f"Property '{key}' should be an object, got {type(actual_value).__name__}"
        # --- End Placeholder Logic ---
        logging.info("Placeholder schema validation successful.")
        return True, None
    except Exception as e: # Catch any unexpected error during validation
        logging.error(f"Schema validation failed unexpectedly: {e}", exc_info=True)
        return False, f"Unexpected validation error: {e}"

# --- Existing request_json function ---

def request_json(
    model: ICogVLM, # Changed parameter name for clarity
    query: str,
    image: Optional[Union[str, Image.Image]] = None,
    extract: bool = True, # Default to True as it's the main purpose
    schema: Optional[Union[Dict[str, Any], str]] = None,
    validate_schema: bool = False,
    max_retries: int = 1, # Default to 1 retry
    system_prmpt_override: Optional[str] = None,
    **kwargs: Any # Pass extra args to inference
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Requests a response from the CogVLM model, aiming for a JSON output.

    Handles system prompt generation (including schema), optional extraction,
    optional schema validation, and retries on failure.

    Args:
        model: An instance implementing the ICogVLM interface.
        query: The user's query.
        image: Optional image input (path, URL, PIL Image).
        extract: If True, attempt to extract JSON from the response.
                 If False, return the raw model response string as the first element.
        schema: Optional JSON schema (as dict or JSON string) to guide the model
                and optionally validate against.
        validate_schema: If True (and schema is provided and extract is True),
                         validate the extracted JSON against the schema.
        max_retries: Number of retries if JSON extraction or validation fails.
        system_prmpt_override: Optionally override the default system prompt.
        **kwargs: Additional keyword arguments passed directly to model.inference().

    Returns:
        A tuple containing:
        - The extracted and validated JSON object (if successful),
          the extracted JSON object (if validation not requested/fails but extraction succeeds),
          or None (if extraction fails after retries). If extract=False, this is the raw response string.
        - The final raw response string from the model.
    """
    if validate_schema and not extract:
        validate_schema = False
        _logger.warning(
            "validate_schema requires extract=True. Disabling schema validation."
        )
    if validate_schema and not schema:
        validate_schema = False
        _logger.warning(
            "validate_schema requires a schema to be provided. Disabling schema validation."
        )

    current_history = [] # Maintain history across retries
    loaded_schema: Optional[Dict[str, Any]] = None

    # --- System Prompt Construction ---
    if system_prmpt_override:
        system_prmpt = system_prmpt_override
    else:
        system_prmpt = "You are a helpful assistant. Respond *only* with valid JSON that adheres to the provided schema (if any)."
        if schema:
            try:
                # Ensure schema is a dict
                if isinstance(schema, str):
                    schema_dict = json.loads(schema)
                elif isinstance(schema, dict):
                    schema_dict = schema
                else:
                    raise TypeError("Schema must be a dict or a JSON string.")

                system_prmpt += f"\n\nThe required JSON Schema is:\n```json\n{json.dumps(schema_dict, indent=2)}\n```\nStrictly adhere to this schema in your JSON output."
                loaded_schema = schema_dict # Use the loaded dict for validation

            except (json.JSONDecodeError, TypeError) as e:
                _logger.error(f"Invalid JSON schema provided: {e}. Proceeding without schema constraint in prompt.")
                # Proceed without schema constraint in prompt, disable validation
                loaded_schema = None
                validate_schema = False
        else:
            system_prmpt += "\nRespond using valid JSON format."
            loaded_schema = None # No schema provided

    current_query = query # Start with the original query
    last_raw_response = "" # Store the last raw response

    for attempt in range(max_retries + 1):
        _logger.info(f"JSON Request - Attempt {attempt + 1}/{max_retries + 1}")
        try:
            response, current_history = model.inference(
                query=current_query,
                images=[image] if image else None,
                system_prmpt=system_prmpt,
                history=current_history, # Pass the evolving history
                # Seed response encourages starting with JSON structure
                seed_response="\n```json\n{\n",
                stream=False, # This function uses non-streaming inference
                **kwargs # Pass through other inference args
            )
            last_raw_response = response # Update last raw response

        except Exception as e:
            _logger.error(f"Inference call failed during attempt {attempt + 1}: {e}", exc_info=True)
            if attempt < max_retries:
                _logger.warning("Retrying after inference failure...")
                # No need to modify query here, the inference itself failed.
                continue # Retry
            else:
                _logger.error("Max retries reached after inference failure.")
                return None, f"Inference failed after {max_retries + 1} attempts: {e}"


        if not extract:
            _logger.debug("Extraction not requested. Returning raw response.")
            # Return raw response as the first element when extract=False
            return response, response

        extracted_json = extract_json(response)

        if not extracted_json:
            _logger.warning(f"Attempt {attempt + 1}: Failed to extract JSON from response.")
            if attempt < max_retries:
                # Modify query for retry, providing context
                current_query = (
                    f"Your previous response could not be parsed as JSON or did not contain a JSON block. "
                    f"Please provide a response strictly in the required JSON format. "
                    f"Ensure the JSON is enclosed in ```json ... ``` markers. "
                    f"Previous response snippet: '{response[:150]}...'"
                )
                # Add user's original query back for context if it's not the first attempt
                if attempt > 0:
                     current_query += f"\nOriginal user query was: {query}"
                continue # Retry
            else:
                _logger.error("Max retries reached. Failed to extract JSON.")
                return None, last_raw_response # Failed after retries

        # JSON extracted, now validate if requested
        if not validate_schema or not loaded_schema:
            _logger.info(
                "JSON extracted successfully. Schema validation not requested or no schema provided."
            )
            return extracted_json, last_raw_response # Success (no validation needed)

        _logger.debug("Validating extracted JSON against schema...")
        is_valid, error_message = validate_json_schema(
            extracted_json, loaded_schema
        )

        if is_valid:
            _logger.info(
                "JSON extracted and schema validation successful."
            )
            return extracted_json, last_raw_response # Success (validation passed)
        else:
            _logger.warning(
                f"Attempt {attempt + 1}: Extracted JSON failed schema validation: {error_message}"
            )
            if attempt < max_retries:
                 # Modify query for retry, providing context
                current_query = (
                    f"Your previous JSON response failed schema validation. "
                    f"Error: '{error_message}'. "
                    f"Please correct the JSON structure and content to match the required schema. "
                    f"Ensure the JSON is enclosed in ```json ... ``` markers."
                 )
                 # Add user's original query back for context if it's not the first attempt
                if attempt > 0:
                     current_query += f"\nOriginal user query was: {query}"
                continue # Retry
            else:
                _logger.error(
                    "Max retries reached. Failed to get schema-valid JSON."
                )
                # Return the invalid JSON and the raw response
                return extracted_json, last_raw_response

    # Fallback if loop finishes unexpectedly (shouldn't happen with current logic)
    _logger.error("Exited retry loop unexpectedly.")
    return None, last_raw_response


# --- NEW request_json_stream function ---

def request_json_stream(
    model: ICogVLM,
    query: str,
    image: Optional[Union[str, Image.Image]] = None,
    schema: Optional[Union[Dict[str, Any], str]] = None,
    system_prmpt_override: Optional[str] = None,
    **kwargs: Any # Pass extra args to inference
) -> Tuple[Iterator[str], Callable]:
    """
    Requests a streaming response from the CogVLM model, aiming for JSON output.

    Sets up the system prompt for JSON generation (including schema if provided)
    and calls the model's streaming inference endpoint.

    Note: This function does *not* perform JSON extraction or validation itself,
          as these can only be reliably done on the *complete* response.
          The caller should consume the iterator and then use the finalizer
          function to get the complete response string, which can then be
          passed to `extract_json` and `validate_json_schema` if needed.
          Retries based on invalid JSON content are not handled by this streaming
          function; the caller would need to implement retry logic by calling
          this function again if the finalized response is unsatisfactory.

    Args:
        model: An instance implementing the ICogVLM interface.
        query: The user's query.
        image: Optional image input (path, URL, PIL Image).
        schema: Optional JSON schema (as dict or JSON string) to guide the model.
        system_prmpt_override: Optionally override the default system prompt.
        **kwargs: Additional keyword arguments passed directly to model.inference().

    Returns:
        A tuple containing:
        - An iterator yielding response tokens as they arrive.
        - A finalizer function (callable with no arguments) that, once the
          iterator is exhausted, returns the final aggregated response string
          and the updated history list: `Tuple[str, List[Tuple[str, str]]]`.
          Calling the finalizer is necessary to confirm the stream completed
          successfully and retrieve the full context.
    """
    _logger.info("Initiating streaming JSON request.")
    loaded_schema: Optional[Dict[str, Any]] = None

    # --- System Prompt Construction ---
    if system_prmpt_override:
        system_prmpt = system_prmpt_override
    else:
        system_prmpt = "You are a helpful assistant. Respond *only* with valid JSON that adheres to the provided schema (if any)."
        if schema:
            try:
                # Ensure schema is a dict
                if isinstance(schema, str):
                    schema_dict = json.loads(schema)
                elif isinstance(schema, dict):
                    schema_dict = schema
                else:
                    raise TypeError("Schema must be a dict or a JSON string.")

                system_prmpt += f"\n\nThe required JSON Schema is:\n```json\n{json.dumps(schema_dict, indent=2)}\n```\nStrictly adhere to this schema in your JSON output."
                loaded_schema = schema_dict # Keep track if schema was loaded

            except (json.JSONDecodeError, TypeError) as e:
                _logger.error(f"Invalid JSON schema provided: {e}. Proceeding without schema constraint in prompt.")
                loaded_schema = None
        else:
            system_prmpt += "\nRespond using valid JSON format."
            loaded_schema = None # No schema provided

    _logger.debug(f"Using system prompt for streaming: {system_prmpt[:200]}...")

    # --- Call Streaming Inference ---
    try:
        # History is typically managed by the caller between stream requests.
        # Pass None here unless explicitly provided via kwargs.
        history = kwargs.pop("history", None)

        token_iterator, finalizer = model.inference(
            query=query,
            images=[image] if image else None,
            system_prmpt=system_prmpt,
            history=history,
            # Seed response encourages starting with JSON structure
            seed_response="\n```json\n{\n",
            stream=True, # Explicitly request streaming
            **kwargs # Pass through other inference args
        )
        _logger.info("Streaming inference initiated successfully.")
        return token_iterator, finalizer

    except Exception as e:
        _logger.error(f"Failed to initiate streaming inference call: {e}", exc_info=True)
        # Re-raise the exception to the caller
        raise


# --- Example Usage (Placeholder - requires CogVLMClient/Model instance) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _logger.info("Running json.py example usage...")

    # This example requires a running CogVLM server and the CogVLMClient
    # Or a direct model instance implementing ICogVLM
    try:
        # Assume client.py is in the parent directory or PYTHONPATH
        from cogvlm.client import CogVLMClient
        # Replace with your server URL if different
        client = CogVLMClient(base_url="http://localhost:8000")

        # Check health before proceeding
        health = client.check_health()
        if not health or health.get("status") != "ok":
             _logger.error(f"Server not healthy: {health}. Aborting example.")
             exit(1)
        if not health.get("model_loaded", False):
             _logger.error(f"Server healthy, but model not loaded: {health}. Aborting example.")
             exit(1)

        _logger.info("CogVLMClient connected successfully.")
        model_instance: ICogVLM = client # Use the client as the model instance

    except ImportError:
        _logger.error("Could not import CogVLMClient. Skipping live examples.")
        _logger.info("You can run the client/server separately to test.")
        model_instance = None # No live model available
    except Exception as e:
        _logger.error(f"Failed to connect to CogVLM server: {e}")
        model_instance = None


    # --- Test Data ---
    test_query = "Describe the weather today in JSON format with fields 'city', 'temperature_celsius', and 'condition'."
    test_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temperature_celsius": {"type": "number"},
            "condition": {"type": "string", "enum": ["sunny", "cloudy", "rainy", "snowy"]},
        },
        "required": ["city", "temperature_celsius", "condition"]
    }

    if model_instance:
        # --- Test request_json (Non-streaming) ---
        print("\n--- Testing request_json (Non-Streaming) ---")
        try:
            # Test with schema validation
            json_result, raw_resp = request_json(
                model=model_instance,
                query=test_query,
                schema=test_schema,
                validate_schema=True,
                max_retries=1
            )
            print(f"Raw Response:\n{raw_resp}")
            if json_result:
                print(f"\nExtracted & Validated JSON:\n{json.dumps(json_result, indent=2)}")
            else:
                print("\nFailed to get valid JSON after retries.")

            # Test without extraction
            print("\n--- Testing request_json (No Extraction) ---")
            raw_resp_only, raw_resp_again = request_json(
                 model=model_instance,
                 query="Tell me a short story.",
                 extract=False
            )
            print(f"Raw Response (extract=False):\n{raw_resp_only}")


        except Exception as e:
            _logger.error(f"Error during request_json test: {e}", exc_info=True)


        # --- Test request_json_stream (Streaming) ---
        print("\n--- Testing request_json_stream (Streaming) ---")
        try:
            stream_iterator, stream_finalizer = request_json_stream(
                model=model_instance,
                query=test_query,
                schema=test_schema,
                # Example of passing inference args:
                # max_new_tokens=150
            )

            print("Streaming Response Tokens:")
            streamed_content = ""
            for token in stream_iterator:
                print(token, end="", flush=True)
                streamed_content += token
            print("\n--- Stream Ended ---")

            # Finalize the stream to get the full response
            final_response, final_history = stream_finalizer()
            print(f"\nFinal Aggregated Response (from finalizer):\n{final_response}")
            # print(f"Final History: {final_history}") # Optional

            # Now, try extracting/validating the *final* streamed response
            print("\n--- Post-Stream Extraction & Validation ---")
            final_json = extract_json(final_response)
            if final_json:
                print(f"Extracted JSON from streamed response:\n{json.dumps(final_json, indent=2)}")
                is_valid, msg = validate_json_schema(final_json, test_schema)
                if is_valid:
                    print("Schema validation PASSED.")
                else:
                    print(f"Schema validation FAILED: {msg}")
            else:
                print("Failed to extract JSON from the final streamed response.")

        except Exception as e:
            _logger.error(f"Error during request_json_stream test: {e}", exc_info=True)

    else:
        print("\nSkipping live model tests as no model instance is available.")

    # --- Test helper functions directly (optional) ---
    print("\n--- Testing helper functions ---")
    test_text_ok = "Some text before ```json\n{\"key\": \"value\", \"num\": 123}\n``` and after."
    test_text_bad_json = "```json\n{\"key\": \"value\", \n```"
    test_text_no_markers = "Plain text {\"key\": \"value\"}"
    test_text_nested = "```json\n{\"outer\": {\"inner\": true}}\n```"

    print(f"Extracting from: '{test_text_ok}' -> {extract_json(test_text_ok)}")
    print(f"Extracting from: '{test_text_bad_json}' -> {extract_json(test_text_bad_json)}")
    print(f"Extracting from: '{test_text_no_markers}' -> {extract_json(test_text_no_markers)}")
    print(f"Extracting from: '{test_text_nested}' -> {extract_json(test_text_nested)}")

    valid_inst = {"city": "London", "temperature_celsius": 15, "condition": "cloudy"}
    invalid_inst_missing = {"city": "Paris", "condition": "sunny"}
    invalid_inst_type = {"city": "Tokyo", "temperature_celsius": "hot", "condition": "sunny"}

    print(f"Validating OK instance: {validate_json_schema(valid_inst, test_schema)}")
    print(f"Validating missing field instance: {validate_json_schema(invalid_inst_missing, test_schema)}")
    print(f"Validating wrong type instance: {validate_json_schema(invalid_inst_type, test_schema)}")

    print("\n--- JSON Module Example Done ---")
