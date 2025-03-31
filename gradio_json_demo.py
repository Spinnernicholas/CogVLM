# fmt: off
import sys
import logging
import json
import pandas as pd
import time
from datetime import datetime
from typing import Optional, Dict, Any, Iterator, List, Tuple, Union

# --- Basic Configuration ---
APP_TITLE = "CogVLM JSON Request Demo"

# --- Logging Setup (Do this FIRST) ---
logging.basicConfig(
    level=logging.INFO, # Use logging.DEBUG for more verbose chat history logs
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    stream=sys.stdout,
)
_logger = logging.getLogger(__name__)
_logger.info(f"Starting {APP_TITLE}...")

# --- Import Core Libraries ---
try:
    import gradio as gr
    from PIL import Image
    _logger.info("Imported Gradio and PIL.")
except ImportError as e:
    _logger.error(f"Missing core dependency: {e}. Please install Gradio and Pillow.")
    sys.exit(1)

# --- Import CogVLM Components ---
try:
    from cogvlm.core import ICogVLM
    from cogvlm.client import CogVLMClient
    from cogvlm.json import request_json, extract_json, validate_json_schema
    from cogvlm.conversation_manager import conversation_manager

    _logger.info("Successfully imported CogVLM components.")
    COGVLM_AVAILABLE = True
except ImportError as e:
    _logger.error(f"Failed to import CogVLM components: {e}. App cannot run.")
    COGVLM_AVAILABLE = False
    class CogVLMClient: pass
    def request_json(*args, **kwargs): raise ImportError("CogVLM components failed to import")
except Exception as e:
    _logger.exception(f"An unexpected error occurred during CogVLM import: {e}")
    COGVLM_AVAILABLE = False
    class CogVLMClient: pass
    def request_json(*args, **kwargs): raise ImportError("Unexpected error during CogVLM import")


# --- Main Application Logic ---
def process_request(
    query: str,
    image_input: Optional[Image.Image],
    schema_str: Optional[str],
    validate_schema_flag: bool,
    max_retries_input: int,
    system_prompt_str: Optional[str],
    extract_flag: bool,
    seed_response_str: Optional[str],
) -> Iterator[Dict[str, gr.update]]:
    """
    Handles the request lifecycle triggered by the Gradio button.
    Streams chat messages (using flat list 'messages' format) including the
    system prompt, and yields updates.
    """
    _logger.info("Processing new request...")

    # 1. Initialize state
    event_log_data: List[Dict[str, Any]] = []
    chat_history: List[Dict[str, str]] = [] # Flat list 'messages' format
    status = "Starting..."
    json_output = ""
    raw_output = ""
    last_event_data = None
    USER_ROLE_NAME = "user"
    ASSISTANT_ROLE_NAME = "assistant"
    SYSTEM_ROLE_NAME = "system"
    initial_user_message_added = False # Flag

    # Helper to yield updates
    def yield_update(chat=None, log_df=None, stat=None, json_val=None, raw_val=None):
        updates = {}
        if chat is not None:
            # Add extra logging/validation before yielding chat update
            _logger.debug(f"Yielding chat_history update: {chat}")
            # Basic validation (can be expanded)
            if not isinstance(chat, list) or any(not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg for msg in chat):
                 _logger.error(f"INVALID chat_history structure detected before yield: {chat}")
                 # Decide how to handle: skip update, yield empty, raise error?
                 # For now, let's skip the chat update to avoid crashing Gradio
                 pass # Don't add chat update if invalid
            else:
                 updates[out_chat_log] = gr.update(value=chat)

        if log_df is not None: updates[out_event_log] = gr.update(value=log_df if not log_df.empty else pd.DataFrame(columns=["Timestamp", "Event", "Raw Event"]))
        if stat is not None: updates[out_final_status] = gr.update(value=stat)
        if json_val is not None: updates[out_final_json] = gr.update(value=json_val)
        if raw_val is not None: updates[out_final_raw] = gr.update(value=raw_val)
        return updates

    # Initial clear/status update
    initial_log_df = pd.DataFrame(columns=["Timestamp", "Event", "Raw Event"])
    yield yield_update(chat=[], log_df=initial_log_df, stat=status, json_val="", raw_val="")

    # Helper to log events
    def add_log_entry(event_name: str, raw_event_dict: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            raw_event_str = json.dumps(raw_event_dict, default=str)
        except Exception as json_err:
            _logger.error(f"Failed to serialize event for logging: {json_err}")
            raw_event_str = f"Error serializing event: {raw_event_dict}"
        event_log_data.append({"Timestamp": timestamp, "Event": event_name, "Raw Event": raw_event_str})
        return pd.DataFrame(event_log_data)

    # 2. Check prerequisites
    if not COGVLM_AVAILABLE:
        status = "Error: CogVLM components failed to import."
        _logger.error(status)
        current_log_df = add_log_entry("Import Error", {"error": status})
        yield yield_update(log_df=current_log_df, stat=status)
        return
    if not query:
        status = "Error: Query cannot be empty."
        _logger.warning(status)
        current_log_df = add_log_entry("Input Error", {"error": status})
        yield yield_update(log_df=current_log_df, stat=status)
        return

    # 3. Prepare inputs
    if schema_str:
        try:
            json.loads(schema_str)
            _logger.info("Schema string appears valid.")
        except json.JSONDecodeError as e:
            schema_error = f"Invalid Schema JSON: {e}."
            _logger.warning(schema_error)
            current_log_df = add_log_entry("Schema Warning", {"warning": schema_error})
            yield yield_update(log_df=current_log_df)

    max_retries = max(0, int(max_retries_input))
    system_override = system_prompt_str if system_prompt_str else None
    seed_response_arg = seed_response_str if seed_response_str is not None else None

    # 4. Instantiate Model Client
    try:
        _logger.info("Instantiating CogVLMClient...")
        model_client = CogVLMClient()
        _logger.info("CogVLMClient instantiated.")
    except Exception as e:
        status = f"Error: Failed to initialize CogVLMClient: {e}"
        _logger.exception(status)
        current_log_df = add_log_entry("Init Error", {"error": str(e)})
        yield yield_update(log_df=current_log_df, stat=status)
        return

    # 5. Call request_json and stream results
    status = "Requesting response from model..."
    yield yield_update(stat=status)
    try:
        _logger.info(f"Calling request_json with seed_response: {seed_response_arg!r}")
        request_iterator = request_json(
            model=model_client,
            query=query,
            image=image_input,
            extract=extract_flag,
            schema=schema_str,
            validate_schema=validate_schema_flag,
            max_retries=max_retries,
            system_prmpt_override=system_override,
            seed_response=seed_response_arg,
        )

        current_assistant_message_content = "" # Track streaming content

        for event in request_iterator:
            last_event_data = event
            event_name = event.get('event', 'N/A')
            _logger.debug(f"Received event: {event_name}, Raw: {event}")

            # Update Event Log first
            current_log_df = add_log_entry(event_name, event)

            # --- Update Chat Log based on event ---
            chat_updated = False # Flag to check if chat needs yielding

            if event_name == 'system_prompt':
                system_prompt_text = event.get('data', '')
                # Ensure it's added only once at the beginning
                if not chat_history or chat_history[0]['role'] != SYSTEM_ROLE_NAME:
                    _logger.info("Adding system prompt to chat history.")
                    chat_history.insert(0, {"role": SYSTEM_ROLE_NAME, "content": system_prompt_text})
                    chat_updated = True
                else:
                     _logger.debug("System prompt event received, but already present in chat history.")


            elif event_name == 'message':
                event_data = event.get('data')
                if isinstance(event_data, (list, tuple)) and len(event_data) == 2:
                    role, content = event_data
                    role = role.lower() # Normalize role

                    if role == USER_ROLE_NAME:
                        # Add the user message if it's the first one or a retry
                        if not initial_user_message_added or content != query:
                            _logger.info(f"Adding user message to chat history (Initial: {not initial_user_message_added}, Content: '{content[:50]}...')")
                            chat_history.append({"role": USER_ROLE_NAME, "content": content})
                            initial_user_message_added = True # Set flag after adding the first one
                            chat_updated = True
                        else:
                             _logger.debug("User message event received, but content matches initial query and already added.")

                    elif role == ASSISTANT_ROLE_NAME:
                        # Add placeholder and stream content
                        _logger.info("Adding assistant message placeholder.")
                        current_assistant_message_content = ""
                        assistant_message_dict = {"role": ASSISTANT_ROLE_NAME, "content": ""}
                        chat_history.append(assistant_message_dict)
                        chat_updated = True # Initial placeholder added

                        # Ensure the placeholder was added correctly before streaming
                        if not chat_history or chat_history[-1]['role'] != ASSISTANT_ROLE_NAME:
                             _logger.error("Chat history state error: Failed to add assistant message placeholder correctly.")
                             # Attempt to recover or skip streaming for this message
                             chat_history.pop() # Remove potentially wrong placeholder
                             chat_updated = False # Don't yield update for this broken state
                        else:
                            _logger.debug("Streaming assistant response...")
                            # Stream into the last message dict's content
                            for char_index, char in enumerate(content):
                                current_assistant_message_content += char
                                chat_history[-1]['content'] = current_assistant_message_content
                                # Yield frequently during streaming
                                # Avoid logging every single char update unless debugging heavily
                                # _logger.debug(f"Streaming char {char_index}")
                                yield yield_update(chat=chat_history, log_df=current_log_df, stat=status)
                                time.sleep(0.01)

                            _logger.debug("Finished streaming assistant response.")
                            # No need for chat_updated=True here, already yielded during loop
                            chat_updated = False # Reset flag as updates were yielded in loop
                            raw_output = content # Store complete raw response
                    else:
                         _logger.warning(f"Received message with unrecognized role: {role}")
                else:
                     _logger.warning(f"Received 'message' event with unexpected data format: {event_data}")

            # --- Yield Update ---
            # Yield updates for log, status, and potentially chat if modified outside streaming loop
            if chat_updated:
                 yield yield_update(chat=chat_history, log_df=current_log_df, stat=status)
            else:
                 # If only log/status changed, yield that
                 yield yield_update(log_df=current_log_df, stat=status)


        _logger.info("Finished iterating through request_json events.")

    except Exception as e:
        status = f"Runtime Error: {e}"
        _logger.exception("Error during request processing:")
        current_log_df = add_log_entry("Runtime Error", {"error": str(e)})
        if last_event_data is None or last_event_data.get('event') not in ('success', 'failure'):
            last_event_data = {'event': 'failure', 'error': status}
            current_log_df = add_log_entry("failure", last_event_data)
        yield yield_update(log_df=current_log_df, stat=status)
        return

    # 6. Process final result
    _logger.info("Processing final result...")
    final_log_df = pd.DataFrame(event_log_data)

    if last_event_data:
        final_event_name = last_event_data.get('event')
        if final_event_name == 'success':
            status = "Success"
            extracted_json = last_event_data.get('json')
            if extracted_json:
                try:
                    json_output = json.dumps(extracted_json, indent=2)
                except Exception as e:
                    _logger.error(f"Failed to serialize final JSON: {e}")
                    json_output = f"Error: Could not display JSON - {e}"
            raw_output = last_event_data.get('response', raw_output)
            _logger.info(f"Request finished successfully. JSON extracted: {bool(extracted_json)}")
        elif final_event_name == 'failure':
            error_msg = last_event_data.get('error', 'Unknown failure reason')
            status = f"Failed: {error_msg}"
            _logger.warning(f"Request failed: {error_msg}")
            extracted_json = last_event_data.get('json')
            if extracted_json:
                 try:
                    json_output = json.dumps(extracted_json, indent=2)
                 except Exception as e:
                    _logger.error(f"Failed to serialize final JSON (on failure): {e}")
                    json_output = f"Error: Could not display JSON - {e}"
            raw_output = last_event_data.get('response', raw_output)
        else:
            status = f"Finished with unexpected final event: {final_event_name}"
            _logger.warning(f"Request ended with non-terminal event: {final_event_name}")
    else:
        status = "Error: No events received from request function."
        _logger.error(status)
        if not any(entry['Event'] == 'Error' and entry['Raw Event'] == json.dumps({"error": status}, default=str) for entry in event_log_data):
             final_log_df = add_log_entry("Iterator Error", {"error": status})

    # Final update
    # Ensure the final chat state is yielded correctly
    _logger.debug(f"Final yield with chat_history: {chat_history}")
    yield yield_update(chat=chat_history, log_df=final_log_df, stat=status, json_val=json_output, raw_val=raw_output)
    _logger.info(f"Request processing complete. Final status: {status}")


# --- Gradio UI Definition ---
_logger.info("Defining Gradio UI...")
with gr.Blocks(theme=gr.themes.Soft(), title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(
        "Interact with the CogVLM model. View streamed chat (including system prompt), detailed events, and final JSON output."
    )

    with gr.Row():
        # Inputs Column
        with gr.Column(scale=1):
            gr.Markdown("## Inputs")
            inp_query = gr.Textbox(
                label="Query", lines=3,
                placeholder="Example: Describe the main object in the image as JSON.",
            )
            inp_image = gr.Image(label="Image (Optional)", type="pil")

            gr.Markdown("### JSON Options")
            inp_schema = gr.Code(
                label="JSON Schema (Optional)", language="json", lines=8,
                value='{\n  "type": "object",\n  "properties": {\n    "name": {"type": "string", "description": "Product name"},\n    "id": {"type": "string", "description": "Product ID code"},\n    "stock": {"type": "integer", "minimum": 0},\n    "features": {\n      "type": "array",\n      "items": {"type": "string"}\n    }\n  },\n  "required": ["name", "id", "stock"]\n}',
            )
            with gr.Row():
                inp_extract = gr.Checkbox(label="Extract JSON", value=True, info="Attempt to find JSON in the response.")
                inp_validate_schema = gr.Checkbox(label="Validate Schema", value=True, info="Validate extracted JSON against the schema (if provided).")

            gr.Markdown("### Advanced Options")
            inp_max_retries = gr.Number(label="Max Retries", value=1, minimum=0, step=1, precision=0, info="Attempts if JSON extraction or validation fails.")
            inp_system_prompt = gr.Textbox(label="System Prompt Override (Optional)", lines=2, placeholder="Leave blank for default JSON-focused prompt.")
            inp_seed_response = gr.Textbox(label="Seed Response (Optional)", lines=2, placeholder="Example: \\n```json\\n{\\n", info="Text to prepend to the model's response generation.", value="\n```json\n{\n")

            submit_btn = gr.Button("Submit Request", variant="primary")

        # Outputs Column
        with gr.Column(scale=2):
            gr.Markdown("## Results")
            out_final_status = gr.Textbox(label="Final Status", interactive=False)
            with gr.Tabs():
                with gr.TabItem("Chat Stream", id="chat"):
                    out_chat_log = gr.Chatbot(
                        label="Chat Log",
                        bubble_full_width=False,
                        height=500,
                        type="messages" # Ensure type is set
                    )
                with gr.TabItem("Event Log"):
                    out_event_log = gr.DataFrame(
                        label="Processing Events Log", headers=["Timestamp", "Event", "Raw Event"],
                        datatype=["str", "str", "str"], interactive=False,
                        row_count=(15, "dynamic"), wrap=True
                    )
                with gr.TabItem("Final Extracted JSON"):
                    out_final_json = gr.Code(label="Extracted JSON", language="json", lines=20, interactive=False)
                with gr.TabItem("Final Raw Response"):
                    out_final_raw = gr.Textbox(label="Complete Raw Model Response", lines=20, interactive=False)

    # --- Connect UI to Logic ---
    _logger.info("Connecting UI components to processing function...")
    submit_btn.click(
        fn=process_request,
        inputs=[
            inp_query, inp_image, inp_schema, inp_validate_schema,
            inp_max_retries, inp_system_prompt, inp_extract, inp_seed_response,
        ],
        outputs=[
            out_chat_log, out_event_log, out_final_status,
            out_final_json, out_final_raw,
        ],
    )

_logger.info("Gradio UI definition complete.")

# --- Launch the Application ---
if __name__ == "__main__":
    _logger.info(f"Launching {APP_TITLE}...")
    demo.launch()
    _logger.info(f"{APP_TITLE} has shut down.")
# fmt: on
