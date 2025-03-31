import sys
import logging
import json
import re
from typing import Any, Dict, Optional, Tuple, Union, List, Iterator, Callable

import jsonschema
from jsonschema.exceptions import ValidationError, SchemaError
from cogvlm.core import ICogVLM
from PIL import Image

from cogvlm.client import CogVLMClient
from cogvlm.conversation_manager import conversation_manager

_logger = logging.getLogger(__name__)

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Placeholder: Attempts to extract the first JSON object found within ```json ... ``` blocks.
    A more robust implementation is recommended.
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            _logger.warning(f"Failed to decode JSON within ```json block: {e}")

    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            potential_json = text[start:end+1]
            if potential_json.count('{') >= potential_json.count('}') and potential_json.count('[') >= potential_json.count(']'):
                 return json.loads(potential_json)
        return None
    except json.JSONDecodeError:
        return None

def validate_json_schema(
    instance: Dict[str, Any], schema: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validates a dictionary instance against a JSON schema.

    Args:
        instance: The dictionary (JSON object) to validate.
        schema: The dictionary representing the JSON schema.

    Returns:
        A tuple containing:
        - bool: True if validation is successful, False otherwise.
        - Optional[str]: None if validation is successful, or an error
                         message string if validation fails (due to either
                         instance validation error or schema error).
    """
    try:
        jsonschema.validate(instance=instance, schema=schema)
        return True, None
    except ValidationError as e:
        # Instance validation failed
        # e.message often provides a concise error description
        return False, e.message
    except SchemaError as e:
        # The schema itself is invalid
        return False, f"Schema Error: {e.message}"
    except Exception as e:
        # Catch other potential unexpected errors during validation
        return False, f"An unexpected error occurred: {str(e)}"

def request_json(
    model: ICogVLM, # Changed parameter name for clarity
    query: str,
    image: Optional[Union[str, Image.Image]] = None,
    extract: bool = True, # Default to True as it's the main purpose
    schema: Optional[Union[Dict[str, Any], str]] = None,
    validate_schema: bool = False,
    max_retries: int = 1, # Default to 1 retry
    system_prmpt_override: Optional[str] = None,
    seed_response: Optional[str] = "\n```json\n{\n",
    **kwargs: Any # Pass extra args to inference
) -> Iterator[Dict[str, Any]]:
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

    conv_manager = conversation_manager(model)
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
    raw_response = "" # Store the last raw response

    yield {'event': 'system_prompt', 'data': system_prmpt}

    for attempt in range(max_retries + 1):
        _logger.info(f"JSON Request - Attempt {attempt + 1}/{max_retries + 1}")
        try:
            yield {'event': 'message', 'data': (conv_manager.user_name, current_query)}
            for event in conv_manager.send_message(
                current_query,
                images=[image] if image else None,
                system_prompt=system_prmpt,
                seed_response=seed_response
            ):
                yield event
                if event['event'] == 'message':
                    raw_response = event['data'][1]
                else:
                    continue

        except Exception as e:
            _logger.error(f"Inference call failed during attempt {attempt + 1}: {e}", exc_info=True)
            if attempt < max_retries:
                _logger.warning("Retrying after inference failure...")
                # No need to modify query here, the inference itself failed.
                continue # Retry
            else:
                _logger.error("Max retries reached after inference failure.")
                yield {'event': 'failure', 'error': f"Inference failed after {max_retries + 1} attempts: {e}"}
                return


        if not extract:
            _logger.debug("Extraction not requested. Returning raw response.")
            # Return raw response as the first element when extract=False
            yield {'event': 'success', 'response': raw_response}
            return

        extracted_json = extract_json(raw_response)

        if not extracted_json:
            _logger.warning(f"Attempt {attempt + 1}: Failed to extract JSON from response.")
            if attempt < max_retries:
                # Modify query for retry, providing context
                current_query = (
                    f"Your previous response could not be parsed as JSON or did not contain a JSON block. "
                    f"Please provide a response strictly in the required JSON format. "
                    f"Ensure the JSON is enclosed in ```json ... ``` markers. "
                    f"Previous response snippet: '{raw_response[:150]}...'"
                )
                # Add user's original query back for context if it's not the first attempt
                if attempt > 0:
                     current_query += f"\nOriginal user query was: {query}"
                continue # Retry
            else:
                _logger.error("Max retries reached. Failed to extract JSON.")
                yield {'event': 'failure', 'error': f"Max retries reached. Failed to extract JSON after {max_retries + 1} attempts."}
                return

        # JSON extracted, now validate if requested
        if not validate_schema or not loaded_schema:
            _logger.info(
                "JSON extracted successfully. Schema validation not requested or no schema provided."
            )
            yield {'event': 'success', 'response': raw_response, 'json': extracted_json}
            return

        _logger.debug("Validating extracted JSON against schema...")
        is_valid, error_message = validate_json_schema(
            extracted_json, loaded_schema
        )

        if is_valid:
            _logger.info(
                "JSON extracted and schema validation successful."
            )
            yield {'event': 'success', 'response': raw_response, 'json': extracted_json}
            return
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
            yield {'event': 'failure', 'response': raw_response, 'json': extracted_json, 'error': f"Max retries reached. Failed to extract JSON after {max_retries + 1} attempts."}
            return

    # Fallback if loop finishes unexpectedly (shouldn't happen with current logic)
    _logger.error("Exited retry loop unexpectedly.")
    yield {'event': 'failure', 'error': 'Exited retry loop unexpectedly.'}

# --- Main Demo Function ---
def main():
    """Simple demo for the request_json function."""
    # Setup basic logging to see output
    # Note: If mocks or request_json log at import time, basicConfig should
    # be called even earlier, ideally right at the start of the script execution.
    # Here, we assume logging is mainly used *during* the function calls.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        stream=sys.stdout # Print logs to console
    )
    # Ensure the logger instance used by request_json and mocks is configured.
    # If _logger is defined globally, this basicConfig call should configure it.
    # If not, ensure _logger is obtained *after* basicConfig is called.
    global _logger # Assuming _logger is used globally in the other functions
    _logger = logging.getLogger(__name__)


    # 1. Create a mock model instance
    # Assumes MockCogVLM is defined above
    model = CogVLMClient()

    # 2. Define the user query
    user_query = "Describe the product 'WidgetMaster 5000' in JSON format."

    # 3. Define an example schema (optional, but good for demo)
    example_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Product name"},
            "id": {"type": "string", "description": "Product ID code"},
            "stock": {"type": "integer", "minimum": 0},
            "features": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "id", "stock"]
    }

    # 4. Call request_json
    print("\n--- Calling request_json ---")
    # Assumes request_json is defined above
    json_request_iterator = request_json(
        model=model,
        query=user_query,
        image=None, # No image in this demo
        extract=True,
        schema=example_schema,
        validate_schema=True,
        max_retries=9
    )

    # 5. Process the results from the iterator
    print("\n--- Processing Events ---")
    final_result = None
    try:
        for event in json_request_iterator:
            print(f"Received Event: {event}")
            if event['event'] == 'message':
                print(f"\n{event['data'][0]}:\n----------------\n{event['data'][1]}\n----------------\n")
            elif event['event'] == 'system_prompt':
                print(f"\nSystem Prompt:\n----------------\n{event['data'][0]}\n----------------\n")
            # Store the last event to check the final outcome
            final_result = event
    except Exception as e:
        print(f"\n--- An error occurred during iteration: {e} ---")
        _logger.exception("Error processing request_json iterator")


    # 6. Print the final outcome
    print("\n--- Final Result ---")
    if final_result:
        print(f"Event Type: {final_result.get('event')}")
        if final_result.get('event') == 'success':
            print("Status: Success!")
            print("Extracted JSON:")
            # Assumes json module is imported above
            print(json.dumps(final_result.get('json'), indent=2))
            print("\nRaw Response Snippet:")
            print(final_result.get('response', '')[:200] + "...")
        elif final_result.get('event') == 'failure':
            print("Status: Failed!")
            print(f"Error: {final_result.get('error')}")
            if 'json' in final_result:
                print("\nExtracted JSON (may be invalid):")
                # Assumes json module is imported above
                print(json.dumps(final_result.get('json'), indent=2))
            if 'response' in final_result:
                 print("\nRaw Response Snippet:")
                 print(final_result.get('response', '')[:200] + "...")
        else:
            print("Status: Unknown final event type.")
            print(f"Final Event Data: {final_result}")
    else:
        print("No final result event received.")

    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    main()
