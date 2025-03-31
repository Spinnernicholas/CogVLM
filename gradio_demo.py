import gradio as gr
import time # Optional: for simulating delay if needed

# Assume conversation_manager.py is in the same directory or accessible
from cogvlm.conversation_manager import conversation_manager
from cogvlm.client import CogVLMClient # Assuming CogVLMClient is here
# If ICogVLM is needed directly (it's used as a type hint)
# from cogvlm.core import ICogVLM

# --- Configuration ---
USER_NAME = "USER"
ASSISTANT_NAME = "ASSISTANT"

# --- Gradio App Logic ---

def chat_interface(messages_input: str, chat_history: list, state_manager: conversation_manager | None):
    """
    Handles one or more turns of conversation based on newline-separated input.

    Args:
        messages_input: A string potentially containing multiple messages
                        separated by newlines.
        chat_history: The current chat history displayed in the Gradio chatbot.
        state_manager: The conversation_manager instance stored in Gradio state.

    Yields:
        Updates to the Gradio components (chatbot, state).
    """
    if not messages_input or messages_input.strip() == "":
        yield chat_history, state_manager
        return

    # Split input into a list of questions, removing empty lines
    questions = [q.strip() for q in messages_input.split('\n') if q.strip()]

    if not questions: # If only whitespace was entered
        yield chat_history, state_manager
        return

    if state_manager is None:
        print("Initializing CogVLMClient and conversation_manager...")
        try:
            cogvlm_client = CogVLMClient()
            state_manager = conversation_manager(
                cogvlm_client, user_name=USER_NAME, assistant_name=ASSISTANT_NAME
            )
            print("Initialization complete.")
        except Exception as e:
            print(f"Error initializing CogVLMClient: {e}")
            # Add error message to chat history for the first question attempt
            chat_history.append([questions[0], f"Error: Could not initialize model. {e}"])
            yield chat_history, state_manager
            return

    # Use the conversation_manager's ask_questions generator with the list
    event_generator = state_manager.ask_questions(questions)

    current_turn_index = -1
    assistant_response_started = False

    # Process events for all questions in the list sequentially
    for event in event_generator:
        event_type = event.get('event')
        event_data = event.get('data')

        if event_type == 'end_message' and event_data[0] == state_manager.user_name:
            # A user message from the list is confirmed by the manager.
            # Add it to Gradio's history.
            chat_history.append([event_data[1], ""]) # User msg, empty assistant placeholder
            current_turn_index = len(chat_history) - 1 # Update index for this new turn
            yield chat_history, state_manager # Update chatbot display

        elif event_type == 'start_message' and event_data == state_manager.assistant_name:
            # Assistant starts responding for the current turn.
            assistant_response_started = True

        elif event_type == 'token':
            # Append token to the assistant's part of the *current* turn
            if assistant_response_started and current_turn_index != -1 and current_turn_index < len(chat_history):
                 # Ensure index is valid before accessing
                chat_history[current_turn_index][1] += event_data
                yield chat_history, state_manager # Stream token update

        elif event_type == 'end_message' and event_data[0] == state_manager.assistant_name:
            # Assistant message finished streaming for the current turn.
            final_response = event_data[1]
            if current_turn_index != -1 and current_turn_index < len(chat_history):
                 # Final check/update if needed
                 if chat_history[current_turn_index][1] != final_response:
                     chat_history[current_turn_index][1] = final_response
                     yield chat_history, state_manager # Ensure final state
            assistant_response_started = False # Reset for next potential turn in the list

        elif event_type == 'history':
            # Internal history updated in the manager. No UI action needed here.
            pass

    # Final yield to ensure state is correctly passed back
    yield chat_history, state_manager


def clear_chat(chat_history, state_manager):
    """Clears the chat history and resets the conversation manager state."""
    print("Clearing chat and resetting state...")
    new_manager = None
    try:
        cogvlm_client = CogVLMClient()
        new_manager = conversation_manager(
            cogvlm_client, user_name=USER_NAME, assistant_name=ASSISTANT_NAME
        )
        print("State reset.")
        return [], new_manager
    except Exception as e:
        print(f"Error re-initializing CogVLMClient during clear: {e}")
        error_msg = f"Error resetting chat: {e}"
        return [[None, error_msg]], state_manager


# --- Build Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# CogVLM Chat Demo")
    gr.Markdown("Enter one or more messages below, separated by newlines. Press Enter (Shift+Enter for newline in textbox) to send.")

    manager_state = gr.State(None)

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        label="Conversation",
        bubble_full_width=False,
        height=600
    )

    with gr.Row():
        # Use a Textbox with multiple lines enabled
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter message(s) here (one per line)...",
            container=False,
            lines=3, # Start with 3 visible lines, can grow
        )

    clear_btn = gr.Button("Clear Chat History")

    # --- Event Handling ---
    # Submit happens when Enter is pressed (unless Shift+Enter for newline)
    txt_msg = txt.submit(
        chat_interface,
        inputs=[txt, chatbot, manager_state],
        outputs=[chatbot, manager_state],
        queue=True
    ).then(lambda: gr.Textbox(value=""), None, [txt], queue=False) # Clear textbox

    clear_btn.click(
        clear_chat,
        inputs=[chatbot, manager_state],
        outputs=[chatbot, manager_state],
        queue=False
    )


if __name__ == "__main__":
    print("Starting Gradio App...")
    demo.launch()
