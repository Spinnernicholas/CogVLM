import logging
from typing import List, Tuple, Optional, Iterator, Callable, Dict, Any

# Assuming ICogVLM is defined elsewhere (e.g., in cogvlm.core)
from cogvlm.core import ICogVLM

DEFAULT_USER_NAME: str = "USER"
DEFAULT_ASSISTANT_NAME: str = "ASSISTANT"

_logger = logging.getLogger(__name__)

class conversation_manager:
    """
    Manages a conversation session with a CogVLM model, handling history,
    system prompts, and message streaming.
    """
    model: ICogVLM
    system_prompt: Optional[str]
    user_name: str
    assistant_name: str
    history: List[Tuple[str, str]] # Explicitly type history

    def __init__(self,
        model: ICogVLM,
        system_prompt: Optional[str] = None, # Added system_prompt here
        user_name: str = DEFAULT_USER_NAME,
        assistant_name: str = DEFAULT_ASSISTANT_NAME,
        history: Optional[List[Tuple[str, str]]] = None # Default to None, init as []
    ):
        """
        Initializes the conversation manager.

        Args:
            model: An object conforming to the ICogVLM protocol.
            system_prompt: An optional initial instruction or context for the model.
            user_name: The identifier for the user role.
            assistant_name: The identifier for the assistant role.
            history: An optional initial conversation history.
        """
        self.model = model
        self.system_prompt = system_prompt # Store system_prompt
        self.user_name = user_name
        self.assistant_name = assistant_name
        # Initialize history as an empty list if None is provided
        self.history = list(history) if history is not None else []
        _logger.info(f"ConversationManager initialized. System Prompt: '{system_prompt}'")

    def send_message(
        self,
        message: str,
        images: Optional[List[Any]] = None, # Added images parameter
        seed_response: str = "", # Added seed_response parameter
        role: Optional[str] = None,
        receiver_role: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048, # Allow overriding model defaults
        top_k: int = 1, # Allow overriding model defaults
    ) -> Iterator[Dict[str, Any]]:
        """
        Sends a message to the model and yields events for the streaming response.

        Args:
            message: The user's message text.
            images: Optional list of images (paths, URLs, PIL Images) for VLM input.
            seed_response: Optional text to guide the beginning of the model's response.
            role: The role sending the message (defaults to user_name).
            max_new_tokens: Max tokens for this specific response generation.
            top_k: Top-k sampling for this specific response generation.

        Yields:
            Dictionaries representing streaming events:
            - {'event': 'start_message', 'data': assistant_name}: Indicates response start.
            - {'event': 'token', 'data': token}: A generated response token.
            - {'event': 'message', 'data': (assistant_name, full_response)}: The final message.
        """
        if not role:
            role = self.user_name
        if not receiver_role:
            role = self.assistant_name

        if system_prompt:
            self.system_prompt = system_prompt

        _logger.info(f"Sending message from '{role}'. Seed: '{seed_response[:50]}...'")
        _logger.debug(f"Current history length: {len(self.history)}")

        # Ensure history contains tuples of (str, str)
        # This might be redundant if __init__ and finalizer guarantee it, but safer
        validated_history: List[Tuple[str, str]] = [
            (str(r), str(m)) for r, m in self.history
        ]

        try:
            # Pass system_prompt and seed_response to the model's inference method
            token_iterator: Iterator[str]
            stream_finalizer: Callable[[], Tuple[str, List[Tuple[str, str]]]]
            token_iterator, stream_finalizer = self.model.inference(
                query=message,
                system_prmpt=self.system_prompt, # Pass stored system prompt
                images=images, # Pass images
                history=validated_history, # Pass current history
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                user_name=self.user_name, # Ensure model knows user role name
                assistant_name=self.assistant_name, # Ensure model knows assistant role name
                seed_response=seed_response, # Pass seed response
                stream=True
            )

            # Signal the start of the assistant's response stream
            yield {'event': 'start_message', 'data': self.assistant_name}

            # Stream the tokens
            token: str
            for token in token_iterator:
                yield {'event': 'token', 'data': token}

            # Finalize the stream to get the full response and updated history
            _logger.info("Stream finished, finalizing...")
            full_response: str
            updated_history: List[Tuple[str, str]]
            full_response, updated_history = stream_finalizer()

            # Update the conversation manager's history
            self.history = updated_history
            _logger.info(f"History updated. New length: {len(self.history)}")
            _logger.debug(f"Final response length: {len(full_response)}")

            # Yield the final complete message from the assistant
            # Note: The finalizer in CogVLM core should return the history
            # including the latest user query and the full assistant response.
            # We yield the assistant part here.
            yield {'event': 'message', 'data': (self.assistant_name, full_response)}

        except Exception as e:
            _logger.error(f"Error during model inference or streaming: {e}", exc_info=True)
            # Yield an error event (optional, depends on how you want to handle errors)
            yield {'event': 'error', 'data': f"Failed to get response: {e}"}


    def get_system_prompt(self) -> Optional[str]:
        """Returns the configured system prompt."""
        return self.system_prompt

    def set_system_prompt(self, system_prompt: Optional[str]):
        """Updates the system prompt."""
        _logger.info(f"Setting system prompt to: '{system_prompt}'")
        self.system_prompt = system_prompt

    def get_history(self) -> List[Tuple[str, str]]:
        """Returns the current conversation history."""
        return list(self.history) # Return a copy

    def clear_history(self):
        """Clears the conversation history."""
        _logger.info("Clearing conversation history.")
        self.history = []
