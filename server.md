# CogVLM Server API Specification

This document outlines the API endpoints for the CogVLM server, which provides visual language model inference capabilities.

## Base URL

```
http://{host}:{port}
```

Replace `{host}` and `{port}` with your server's host and port (default: `localhost:8000`).

## Endpoints

### 1. Inference

Processes a text query with optional image input and conversation history.

**URL:** `/inference`

**Method:** `POST`

**Content-Type:** `application/json`

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | The user's text query |
| images | array of strings | No | List of image URLs, local file paths, or base64 data URIs (currently only the first image is used) |
| system_prmpt | string | No | Optional system prompt to guide the model's behavior |
| history | array of objects | No | Previous conversation turns |
| max_new_tokens | integer | No | Maximum number of tokens to generate (default: 2048) |
| top_k | integer | No | Top-k sampling parameter (default: 1) |
| user_name | string | No | Name for the user role in history (default: "USER") |
| assistant_name | string | No | Name for the assistant role in history (default: "ASSISTANT") |
| seed_response | string | No | Optional text to seed the assistant's response (default: "") |

**History Object:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| role | string | Yes | Role of the speaker ("USER" or "ASSISTANT") |
| content | string | Yes | The message content |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| response | string | The generated text response from the model |
| history | array of objects | The updated conversation history including the new response |
| error | string or null | Error message if processing failed |

**Status Codes:**

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters or image processing error) |
| 500 | Internal Server Error (model inference error) |
| 503 | Service Unavailable (model not loaded) |

**Example Request:**

```json
{
  "query": "What can you see in this image?",
  "images": ["https://example.com/image.jpg"],
  "history": [
    {"role": "USER", "content": "Hello, I have an image to show you."},
    {"role": "ASSISTANT", "content": "Hello! I'd be happy to look at your image. What would you like to know about it?"}
  ],
  "max_new_tokens": 1024,
  "top_k": 1
}
```

**Example Response:**

```json
{
  "response": "In this image, I can see a scenic mountain landscape with snow-capped peaks against a clear blue sky. There appears to be a forest of evergreen trees in the foreground and a small lake reflecting the mountains.",
  "history": [
    {"role": "USER", "content": "Hello, I have an image to show you."},
    {"role": "ASSISTANT", "content": "Hello! I'd be happy to look at your image. What would you like to know about it?"},
    {"role": "USER", "content": "What can you see in this image?"},
    {"role": "ASSISTANT", "content": "In this image, I can see a scenic mountain landscape with snow-capped peaks against a clear blue sky. There appears to be a forest of evergreen trees in the foreground and a small lake reflecting the mountains."}
  ],
  "error": null
}
```

### 2. Health Check

Checks if the server is running and the model is loaded.

**URL:** `/health`

**Method:** `GET`

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| status | string | "ok" if the server is healthy, "error" otherwise |
| model_loaded | boolean | Whether the model is loaded successfully |
| model_path | string | The path or identifier of the loaded model (if model_loaded is true) |
| detail | string | Error details (if model_loaded is false) |

**Status Codes:**

| Status Code | Description |
|-------------|-------------|
| 200 | Success (even if model is not loaded, the endpoint itself is working) |

**Example Response (Success):**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "THUDM/cogvlm2-llama3-chat-19B"
}
```

**Example Response (Model Not Loaded):**

```json
{
  "status": "error",
  "model_loaded": false,
  "detail": "Model not loaded."
}
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure. In case of an error, the response body will contain an error message explaining what went wrong.

## Notes

1. The server currently only processes the first image in the `images` array.
2. Base64-encoded images should be prefixed with the data URI scheme (e.g., `data:image/jpeg;base64,/9j/4AAQSkZJRg...`).
3. The model may take some time to generate a response for complex queries or large images.
4. The `max_new_tokens` parameter controls the maximum length of the generated response.
5. Setting `top_k` > 1 enables sampling for more diverse responses.