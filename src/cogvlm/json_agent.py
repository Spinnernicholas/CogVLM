import logging
import json
import re
from typing import Any, Dict, Optional, Tuple, Union, List, Iterator, Callable, Protocol
from PIL import Image

from cogvlm.core import ICogVLM
from cogvlm.conversation_manager import agent, conversation_manager

_logger = logging.getLogger(__name__)

DEFAULT_USER_NAME: str = "USER"
DEFAULT_ASSISTANT_NAME: str = "ASSISTANT"

# --- Helper Functions (extract_json, validate_json_schema) ---

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
    Placeholder: Validates JSON against a schema.
    Use the 'jsonschema' library for a proper implementation.
    """
    try:
        # --- Basic Placeholder Validation ---
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
        logging.info("Placeholder schema validation successful.")
        return True, None
    except Exception as e:
        logging.error(f"Schema validation failed unexpectedly: {e}", exc_info=True)
        return False, f"Unexpected validation error: {e}"

# --- JsonRequestAgent Class ---

class JsonRequestAgent(agent):
    """
    An agent that interacts with a CogVLM model via a conversation_manager
    to request and potentially validate JSON output based on a query and schema.
    """
    original_query: str
    image: Optional[Union[str, Image.Image]]
    extract: bool
    schema: Optional[Union[Dict[str, Any], str]]
    validate_schema: bool
    max_retries: int
    system_prmpt_override: Optional[str]
    kwargs: Dict[str, Any] # Store extra args, though conversation_manager might not use them

    # Internal state
    attempt_count: int
    current_query: str
    system_prmpt: str
    loaded_schema: Optional[Dict[str, Any]]
    last_raw_response: str
    final_json_result: Optional[Dict[str, Any]]
    _is_finished: bool
    _user_name: str # Store the user name for sending messages

    def __init__(
        self,
        query: str,
        image: Optional[Union[str, Image.Image]] = None,
        extract: bool = True,
        schema: Optional[Union[Dict[str, Any], str]] = None,
        validate_schema: bool = False,
        max_retries: int = 1,
        system_prmpt_override: Optional[str] = None,
        user_name: str = DEFAULT_USER_NAME, # Added user_name
        **kwargs: Any
    ):
        self.original_query = query
        self.image = image # Note: conversation_manager needs modification to handle images
        self.extract = extract
        self.schema = schema
        self.validate_schema = validate_schema
        self.max_retries = max_retries
        self.system_prmpt_override = system_prmpt_override
        self.kwargs = kwargs
        self._user_name = user_name

        # Initialize state
        self.attempt_count = 0
        self.last_raw_response = ""
        self.final_json_result = None
        self._is_finished = False
        self.loaded_schema = None

        # --- Input Validation ---
        if self.validate_schema and not self.extract:
            self.validate_schema = False
            _logger.warning(
                "validate_schema requires extract=True. Disabling schema validation."
            )
        if self.validate_schema and not self.schema:
            self.validate_schema = False
            _logger.warning(
                "validate_schema requires a schema to be provided. Disabling schema validation."
            )

        # --- System Prompt Construction ---
        if self.system_prmpt_override:
            self.system_prmpt = self.system_prmpt_override
        else:
            self.system_prmpt = "You are a helpful assistant. Respond *only* with valid JSON that adheres to the provided schema (if any)."
            if self.schema:
                try:
                    if isinstance(self.schema, str):
                        schema_dict = json.loads(self.schema)
                    elif isinstance(self.schema, dict):
                        schema_dict = self.schema
                    else:
                        raise TypeError("Schema must be a dict or a JSON string.")

                    self.system_prmpt += f"\n\nThe required JSON Schema is:\n```json\n{json.dumps(schema_dict, indent=2)}\n```\nStrictly adhere to this schema in your JSON output."
                    self.loaded_schema = schema_dict

                except (json.JSONDecodeError, TypeError) as e:
                    _logger.error(f"Invalid JSON schema provided: {e}. Proceeding without schema constraint.")
                    self.loaded_schema = None
                    self.validate_schema = False # Can't validate without loaded schema
            else:
                self.system_prmpt += "\nRespond using valid JSON format."
                self.loaded_schema = None

        # Set the initial query for the first turn
        # We prepend the system prompt here as conversation_manager doesn't have a separate system prompt arg
        # A better approach would be to modify conversation_manager to accept system prompts.
        self.current_query = f"{self.system_prmpt}\n\n{self.original_query}"
        _logger.info(f"JsonRequestAgent initialized. Max retries: {self.max_retries}")
        _logger.debug(f"Initial system prompt part: {self.system_prmpt}")


    def start(self) -> Tuple[Optional[str], Optional[str]]:
        """Initiates the conversation by sending the first query."""
        if self._is_finished:
            return None, None # Already finished
        _logger.info(f"Agent starting. Attempt {self.attempt_count + 1}/{self.max_retries + 1}")
        # The conversation_manager expects (role, message)
        # Role is who is speaking now (the user)
        # Message is the content for the LLM
        return self._user_name, self.current_query

    def process(self, message: Any) -> Tuple[Optional[str], Optional[str]]:
        """Processes the model's response and decides whether to retry or finish."""
        if self._is_finished:
            return None, None # Already finished

        # Expecting message format from conversation_manager's 'end_message' event data
        # which is (assistant_name, response_text)
        if not isinstance(message, tuple) or len(message) != 2:
             _logger.error(f"Agent received unexpected message format: {message}. Stopping.")
             self._is_finished = True
             return None, None

        assistant_name, response_text = message
        self.last_raw_response = response_text
        self.attempt_count += 1
        _logger.debug(f"Agent received response (attempt {self.attempt_count}):\n{response_text}")


        # --- Case 1: No extraction needed ---
        if not self.extract:
            _logger.info("Extraction not requested. Storing raw response and finishing.")
            # Store raw response as the 'result' when not extracting
            self.final_json_result = response_text # Type mismatch, but follows original logic
            self._is_finished = True
            return None, None # Signal conversation end

        # --- Case 2: Extraction needed ---
        extracted_json = extract_json(response_text)

        if not extracted_json:
            _logger.warning(f"Attempt {self.attempt_count}: Failed to extract JSON.")
            if self.attempt_count <= self.max_retries:
                _logger.info("Retrying...")
                # Construct retry query
                self.current_query = (
                    f"{self.system_prmpt}\n\n" # Re-include system prompt instructions
                    f"Your previous response could not be parsed as JSON or did not contain a JSON block. "
                    f"Please provide a response strictly in the required JSON format. "
                )
                return self._user_name, self.current_query # Send retry query
            else:
                _logger.error("Max retries reached. Failed to extract JSON.")
                self.final_json_result = None # Indicate failure
                self._is_finished = True
                return None, None # Signal conversation end

        # --- Case 3: Extraction successful, check validation ---
        _logger.info(f"Attempt {self.attempt_count}: JSON extracted successfully.")
        if not self.validate_schema or not self.loaded_schema:
            _logger.info("Schema validation not requested or no schema provided. Finishing.")
            self.final_json_result = extracted_json
            self._is_finished = True
            return None, None # Signal conversation end

        # --- Case 4: Validation needed ---
        _logger.debug("Validating extracted JSON against schema...")
        is_valid, error_message = validate_json_schema(
            extracted_json, self.loaded_schema
        )

        if is_valid:
            _logger.info("JSON extracted and schema validation successful. Finishing.")
            self.final_json_result = extracted_json
            self._is_finished = True
            return None, None # Signal conversation end
        else:
            _logger.warning(f"Attempt {self.attempt_count}: Extracted JSON failed schema validation: {error_message}")
            if self.attempt_count <= self.max_retries:
                _logger.info("Retrying due to validation failure...")
                # Construct retry query
                self.current_query = (
                    f"Your previous JSON response failed schema validation. "
                    f"Error: '{error_message}'. "
                    f"Please correct the JSON structure and content to match the required schema. "
                )
                return self._user_name, self.current_query # Send retry query
            else:
                _logger.error("Max retries reached. Failed to get schema-valid JSON.")
                # Store the last invalid JSON as the result, mirroring original function
                self.final_json_result = extracted_json
                self._is_finished = True
                return None, None # Signal conversation end

        # Should not be reached
        _logger.error("Agent process logic error. Stopping.")
        self._is_finished = True
        return None, None


    def stop(self, message: Any) -> None:
        """Called when the conversation loop ends."""
        _logger.info(f"Agent stopping. Final state reached after {self.attempt_count} attempt(s).")
        # Can perform cleanup here if needed
        pass

    def get_result(self) -> Tuple[Optional[Union[Dict[str, Any], str]], str]:
        """
        Returns the final result after the agent has finished processing.

        Returns:
            A tuple containing:
            - The extracted and validated JSON object (if successful),
              the extracted JSON object (if validation not requested/fails but extraction succeeds),
              the raw response string (if extract=False),
              or None (if extraction fails after retries).
            - The final raw response string received from the model.
        """
        if not self._is_finished:
            _logger.warning("get_result() called before agent finished processing.")
        return self.final_json_result, self.last_raw_response

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _logger.info("Running json.py agent example usage...")

    model_instance: Optional[ICogVLM] = None
    try:
        from cogvlm.client import CogVLMClient
        client = CogVLMClient(base_url="http://localhost:8000")
        health = client.check_health()
        if not health or health.get("status") != "ok" or not health.get("model_loaded", False):
             _logger.error(f"Server not healthy or model not loaded: {health}. Aborting example.")
             exit(1)
        _logger.info("CogVLMClient connected successfully.")
        model_instance = client
    except ImportError:
        _logger.error("Could not import CogVLMClient. Skipping live examples.")
    except Exception as e:
        _logger.error(f"Failed to connect to CogVLM server: {e}")

    # --- Test Data ---
    test_query = "Describe the weather today in JSON format with fields 'city', 'temperature_celsius', and 'condition'."
    test_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temperature_celsius": {"type": "number"},
            "condition": {"type": "string", "enum": ["sunny", "cloudy", "rainy", "snowy", "foggy", "windy"]}, # Added more options
        },
        "required": ["city", "temperature_celsius", "condition"]
    }
    invalid_query = "Generate JSON for a user profile with name (string) and age (integer)."
    invalid_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "string"}}, # Intentionally wrong type for age
        "required": ["name", "age"]
    }


    if model_instance:
        cm = conversation_manager(model_instance)

        print("\n\n--- Test 1: Valid JSON Request with Schema Validation ---")
        json_agent_valid = JsonRequestAgent(
            query=test_query,
            schema=test_schema,
            validate_schema=True,
            max_retries=1
        )
        # Run the conversation
        for event in cm.run(json_agent_valid):
            if event['event'] == 'token':
                print(event['data'], end="", flush=True)
            elif event['event'] == 'end_message':
                 print(f"\n--- End {event['data'][0]}: {event['data'][1]} ---")
            elif event['event'] == 'start_message':
                 print(f"\n--- Start {event['data']} ---")
            else:
                 print(f"\nEVENT: {event}") # Print other events like errors

        # Get the final result from the agent
        final_json, final_raw = json_agent_valid.get_result()
        print("\n--- Agent Finished ---")
        print(f"Final Raw Response:\n{final_raw}")
        if isinstance(final_json, dict):
            print(f"\nFinal Extracted & Validated JSON:\n{json.dumps(final_json, indent=2)}")
        elif final_json is None:
             print("\nAgent failed to get valid JSON.")
        else: # Should be string if extract=False
             print(f"\nAgent returned raw string (extract=False): {final_json}")


        print("\n\n--- Test 2: Invalid JSON Response (Schema Mismatch) with Retries ---")
        # This test relies on the model *initially* failing the schema
        json_agent_invalid = JsonRequestAgent(
            query=invalid_query,
            schema=invalid_schema, # Schema expects age as string
            validate_schema=True,
            max_retries=1 # Allow one retry
        )
        for event in cm.run(json_agent_invalid):
            if event['event'] == 'token':
                print(event['data'], end="", flush=True)
            elif event['event'] == 'end_message':
                 print(f"\n--- End {event['data'][0]}: {event['data'][1]} ---")
            elif event['event'] == 'start_message':
                 print(f"\n--- Start {event['data']} ---")
            else:
                 print(f"\nEVENT: {event}") # Print other events like errors

        final_json_invalid, final_raw_invalid = json_agent_invalid.get_result()
        print("\n--- Agent Finished ---")
        print(f"Final Raw Response:\n{final_raw_invalid}")
        if isinstance(final_json_invalid, dict):
            print(f"\nFinal Extracted JSON (may be invalid):\n{json.dumps(final_json_invalid, indent=2)}")
            # Check if it *actually* became valid after retry
            is_valid_final, _ = validate_json_schema(final_json_invalid, invalid_schema)
            print(f"Is final JSON valid against schema? {is_valid_final}")
        elif final_json_invalid is None:
             print("\nAgent failed to get JSON (extraction failed).")
        else:
             print(f"\nAgent returned raw string: {final_json_invalid}")


        print("\n\n--- Test 3: Request without Extraction ---")
        json_agent_no_extract = JsonRequestAgent(
            query="Tell me a short story about a robot learning to paint.",
            extract=False,
        )
        for event in cm.run(json_agent_no_extract):
            if event['event'] == 'token':
                print(event['data'], end="", flush=True)
            elif event['event'] == 'end_message':
                 print(f"\n--- End {event['data'][0]}: {event['data'][1]} ---")
            elif event['event'] == 'start_message':
                 print(f"\n--- Start {event['data']} ---")
            else:
                 print(f"\nEVENT: {event}") # Print other events like errors

        final_res_noex, final_raw_noex = json_agent_no_extract.get_result()
        print("\n--- Agent Finished ---")
        print(f"Final Raw Response:\n{final_raw_noex}")
        print(f"\nResult from get_result() (should be same as raw): {final_res_noex}")


    else:
        print("\nSkipping live model tests as no model instance is available.")

    print("\n--- JSON Agent Example Done ---")