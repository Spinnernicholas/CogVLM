import gradio as gr
import logging
import json
from PIL import Image
import io # Needed for image handling if sending PIL object directly
from typing import List, Tuple, Optional, Dict, Any

# --- Assuming client.py and json.py are accessible ---
# If they are in subdirectories, adjust the import paths accordingly
# e.g., from ..cogvlm.client import CogVLMClient
# e.g., from ..dogvlm.json import request_json_stream, extract_json, validate_json_schema
try:
    from cogvlm import CogVLMClient, ICogVLM
    # Import the specific streaming function and helpers
    from cogvlm.json import request_json_stream, extract_json, validate_json_schema
    # Import the interface for type hinting (optional but good practice)
except ImportError as e:
    print(f"ERROR: Could not import necessary modules: {e}")
    print("Please ensure client.py and json.py are in the correct path.")
    exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8000" # CHANGE IF YOUR SERVER IS ELSEWHERE
DEFAULT_SCHEMA = """{
  "type": "object",
  "properties": {
    "summary": {
      "type": "string",
      "description": "A brief summary of the image content or answer."
    },
    "details": {
      "type": "object",
      "description": "Specific details extracted or generated.",
      "properties": {
         "keywords": {
            "type": "array",
            "items": {"type": "string"}
         }
      }
    }
  },
  "required": ["summary"]
}"""

# --- Initialize Client (or create on demand) ---
# Creating it once might be slightly more efficient if the app runs long
try:
    client = CogVLMClient(base_url=SERVER_URL)
    # Perform an initial health check
    health = client.check_health()
    if not health or health.get("status") != "ok" or not health.get("model_loaded"):
        _logger.error(f"CogVLM server not ready at {SERVER_URL}. Health: {health}")
        # Optionally raise an error or exit, or let Gradio handle connection errors later
        # raise RuntimeError(f"CogVLM server not ready: {health}")
        print(f"WARNING: CogVLM server not ready at {SERVER_URL}. Health: {health}")
        print("The application might not function correctly.")
    else:
        _logger.info(f"CogVLM client connected successfully to {SERVER_URL}")
except Exception as e:
    _logger.error(f"Failed to initialize CogVLMClient: {e}", exc_info=True)
    print(f"ERROR: Failed to initialize CogVLMClient targeting {SERVER_URL}.")
    print("Ensure the server is running and the URL is correct.")
    # Set client to None to handle gracefully later, or exit
    client = None
    # exit(1) # Or handle this more gracefully in the chat function


# --- Gradio Chat Function ---
def process_chat(
    user_query: str,
    image_input: Optional[Image.Image], # Gradio provides PIL Image
    schema_input: str,
    chat_history: List[Tuple[str, str]]
):
    """
    Handles the chat interaction, calls the streaming API, and updates history.
    Yields updates to the chatbot UI during streaming.
    """
    if not client:
        yield chat_history + [
            (user_query, "**Error:** CogVLM Client not initialized. Cannot connect to server.")
        ], chat_history
        return

    _logger.info(f"Processing query: '{user_query[:50]}...', Image: {'Yes' if image_input else 'No'}, Schema: {'Yes' if schema_input else 'No'}")

    # --- Input Preparation ---
    # Append user message to history immediately
    user_message_display = user_query
    if image_input:
        # You could potentially display the image using Markdown later,
        # but for simplicity, just note its presence.
        user_message_display += " (Image provided)"
    chat_history.append((user_message_display, None)) # Placeholder for assistant response
    yield chat_history, chat_history # Update UI to show user query

    parsed_schema: Optional[Dict[str, Any]] = None
    if schema_input:
        try:
            parsed_schema = json.loads(schema_input)
            _logger.info("Schema provided and parsed successfully.")
        except json.JSONDecodeError as e:
            _logger.warning(f"Invalid JSON schema provided: {e}")
            # Append error message to chat instead of proceeding with bad schema
            chat_history[-1] = (chat_history[-1][0], f"**Error:** Invalid JSON schema provided. Please correct it.\n\nDetails: `{e}`")
            yield chat_history, chat_history
            return # Stop processing this turn

    # --- Streaming Call ---
    assistant_response = ""
    finalizer = None # To store the finalizer function

    try:
        # Use the imported request_json_stream function
        token_iterator, finalizer = request_json_stream(
            model=client, # Pass the client instance as the model
            query=user_query,
            image=image_input, # Pass the PIL image directly
            schema=parsed_schema, # Pass the parsed schema dict
            # Add any other inference params if needed via kwargs, e.g.:
            # max_new_tokens=1024
        )

        _logger.debug("Streaming started...")
        for token in token_iterator:
            assistant_response += token
            # Update the last element (assistant's turn) in history
            chat_history[-1] = (chat_history[-1][0], assistant_response)
            yield chat_history, chat_history # Yield history for Gradio UI update

        _logger.debug("Streaming finished.")

    except Exception as e:
        _logger.error(f"Error during streaming inference call: {e}", exc_info=True)
        error_message = f"**Error during streaming:**\n\n`{e}`"
        # Update the last message with the error
        if chat_history: # Ensure history is not empty
             chat_history[-1] = (chat_history[-1][0], error_message)
        else: # Should not happen if we append user query first, but safety check
             chat_history.append((user_query, error_message))
        yield chat_history, chat_history
        return # Stop processing this turn
    finally:
        # --- Finalization and Post-Stream Processing ---
        if finalizer:
            try:
                _logger.debug("Calling finalizer...")
                # The finalizer from client.inference returns (final_response_str, final_history_list)
                # Note: request_json_stream currently just returns the finalizer from client.inference
                final_response_str, _ = finalizer() # We manage history in Gradio state
                _logger.info(f"Finalized response length: {len(final_response_str)}")

                # Ensure the final response is reflected in the chat history
                # (Usually the streamed content should match, but this is safer)
                if assistant_response.strip() != final_response_str.strip():
                    _logger.warning("Streamed content differs slightly from finalized response. Updating chat history.")
                    chat_history[-1] = (chat_history[-1][0], final_response_str)
                    assistant_response = final_response_str # Use finalized version
                    yield chat_history, chat_history # Yield final update

                # --- Optional: Post-stream JSON validation ---
                if parsed_schema:
                    _logger.info("Attempting post-stream JSON extraction and validation...")
                    extracted = extract_json(assistant_response)
                    validation_msg = ""
                    if extracted:
                        is_valid, msg = validate_json_schema(extracted, parsed_schema)
                        if is_valid:
                            validation_msg = "\n\n*(JSON extracted and schema validation PASSED)*"
                            _logger.info("Post-stream validation successful.")
                        else:
                            validation_msg = f"\n\n*(JSON extracted but schema validation FAILED: {msg})*"
                            _logger.warning(f"Post-stream validation failed: {msg}")
                    else:
                        validation_msg = "\n\n*(Failed to extract JSON from the final response)*"
                        _logger.warning("Post-stream JSON extraction failed.")

                    # Append validation status to the assistant's message
                    chat_history[-1] = (chat_history[-1][0], assistant_response + validation_msg)
                    yield chat_history, chat_history

            except Exception as e:
                _logger.error(f"Error during stream finalization or post-processing: {e}", exc_info=True)
                # Append error to the current assistant message
                error_msg = f"\n\n**Error during finalization:** `{e}`"
                chat_history[-1] = (chat_history[-1][0], assistant_response + error_msg)
                yield chat_history, chat_history


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # CogVLM Streaming JSON Chat Demo
    Interact with the CogVLM model using the `request_json_stream` function.
    The assistant aims to respond in JSON format, optionally guided by a schema.
    Server URL: `{SERVER_URL}`
    """)

    # State to store the conversation history
    chatbot_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            chatbot_display = gr.Chatbot(
                label="Conversation",
                bubble_full_width=False,
                height=500,
            )
            user_query_textbox = gr.Textbox(
                label="Your Message",
                placeholder="Type your query here...",
                lines=3,
            )
            with gr.Row():
                 clear_button = gr.Button("Clear Chat")
                 submit_button = gr.Button("Send", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## Inputs")
            image_input_box = gr.Image(label="Upload Image (Optional)", type="pil")
            schema_textbox = gr.Textbox(
                label="JSON Schema (Optional)",
                placeholder="Paste your JSON schema here...",
                lines=10,
                value=DEFAULT_SCHEMA # Pre-fill with default
            )

    # --- Event Handlers ---
    submit_button.click(
        fn=process_chat,
        inputs=[user_query_textbox, image_input_box, schema_textbox, chatbot_state],
        outputs=[chatbot_display, chatbot_state],
        # Note: Clearing inputs happens after fn returns/yields last value
    ).then(
        lambda: gr.update(value=""), # Clear query box after processing
        outputs=[user_query_textbox]
    )

    user_query_textbox.submit(
         fn=process_chat,
        inputs=[user_query_textbox, image_input_box, schema_textbox, chatbot_state],
        outputs=[chatbot_display, chatbot_state],
    ).then(
        lambda: gr.update(value=""), # Clear query box after processing
        outputs=[user_query_textbox]
    )

    def clear_chat_history():
        return [], [] # Clear display and state

    clear_button.click(
        fn=clear_chat_history,
        outputs=[chatbot_display, chatbot_state],
        queue=False # No need to queue simple clear
    )


# --- Launch the App ---
if __name__ == "__main__":
    demo.queue() # Enable queuing for handling multiple users/requests
    demo.launch(debug=True) # Debug=True provides more logs
