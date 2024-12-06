from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from aiohttp import ClientSession, ClientTimeout
from typing import Dict, List, AsyncGenerator, Any, Union, Optional
import json
import os

from copilot_more.token import get_cached_copilot_token
from copilot_more.logger import logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
API_URL = "https://api.individual.githubcopilot.com/chat/completions"
TIMEOUT = ClientTimeout(total=300)
EDITOR_VERSION = "vscode/1.95.3"

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")

def validate_api_key(authorization: Optional[str] = Header(None), api_key: Optional[str] = Header(None)) -> str:
    """
    Validate the API key from either Authorization or api-key header.
    
    Args:
        authorization: Authorization header value
        api_key: api-key header value
        
    Returns:
        str: The validated API key
        
    Raises:
        HTTPException: If no valid API key is found
    """
    if not API_KEY:
        raise HTTPException(500, "API key not configured on server")

    # Check Authorization header
    if authorization:
        if authorization.startswith("Bearer "):
            key = authorization[7:]
            if key == API_KEY:
                return key
        elif authorization == API_KEY:
            return authorization

    # Check api-key header
    if api_key and api_key == API_KEY:
        return api_key

    raise HTTPException(401, "Invalid API key")

def preprocess_request_body(request_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess the request body to handle array content in messages.
    
    Args:
        request_body: The original request body
        
    Returns:
        Dict[str, Any]: Processed request body with flattened messages
        
    Raises:
        HTTPException: If content type is not supported
    """
    if not request_body.get("messages"):
        return request_body

    processed_messages: List[Dict[str, str]] = []

    for message in request_body["messages"]:
        if not isinstance(message.get("content"), list):
            processed_messages.append(message)
            continue

        for content_item in message["content"]:
            if content_item.get("type") != "text":
                raise HTTPException(400, "Only text type is supported in content array")

            processed_messages.append({
                "role": message["role"],
                "content": content_item["text"]
            })

    return {
        **request_body,
        "messages": processed_messages
    }

async def create_stream_response(request_body: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
    """
    Creates a streaming response for chat completions.
    
    Args:
        request_body: The processed request body
        api_key: The validated API key
        
    Yields:
        bytes: Chunks of the response
        
    Raises:
        HTTPException: If API returns an error
    """
    try:
        token = await get_cached_copilot_token()
        async with ClientSession(timeout=TIMEOUT) as session:
            async with session.post(
                API_URL,
                json=request_body,
                headers={
                    "Authorization": f"Bearer {token['token']}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "editor-version": EDITOR_VERSION,
                },
            ) as response:
                if response.status != 200:
                    error_message = await response.text()
                    logger.error(f"API error: {error_message}")
                    raise HTTPException(
                        response.status,
                        f"API error: {error_message}"
                    )

                async for chunk in response.content.iter_chunks():
                    if chunk:
                        chunk_str = chunk[0].decode('utf-8')
                        if chunk_str.startswith('data: '):
                            try:
                                # Parse the JSON data
                                data = json.loads(chunk_str[6:])
                                if 'choices' in data and isinstance(data['choices'], list):
                                    # Add index to each choice
                                    for i, choice in enumerate(data['choices']):
                                        choice['index'] = i
                                # Add completion_tokens to usage if present
                                if 'usage' in data:
                                    if not isinstance(data['usage'], dict):
                                        data['usage'] = {}
                                    if 'completion_tokens' not in data['usage'] and 'prompt_tokens' in data['usage'] and 'total_tokens' in data['usage']:
                                        data['usage']['completion_tokens'] = data['usage']['total_tokens'] - data['usage']['prompt_tokens']
                                    elif 'completion_tokens' not in data['usage']:
                                        data['usage']['completion_tokens'] = 0  # Initialize to 0
                                # Add the object field and reconstruct
                                data['object'] = 'chat.completion'
                                modified_chunk = f'data: {json.dumps(data)}\n\n'
                                yield modified_chunk.encode('utf-8')
                            except json.JSONDecodeError:
                                # If we can't parse the JSON, add object field directly
                                modified_chunk = 'data: {"object": "chat.completion",' + chunk_str[6:].lstrip('{')
                                yield modified_chunk.encode('utf-8')
                        elif chunk_str.startswith('{'):
                            try:
                                # Parse the JSON data
                                data = json.loads(chunk_str)
                                if 'choices' in data and isinstance(data['choices'], list):
                                    # Add index to each choice
                                    for i, choice in enumerate(data['choices']):
                                        choice['index'] = i
                                # Add completion_tokens to usage if present
                                if 'usage' in data:
                                    if not isinstance(data['usage'], dict):
                                        data['usage'] = {}
                                    if 'completion_tokens' not in data['usage'] and 'prompt_tokens' in data['usage'] and 'total_tokens' in data['usage']:
                                        data['usage']['completion_tokens'] = data['usage']['total_tokens'] - data['usage']['prompt_tokens']
                                    elif 'completion_tokens' not in data['usage']:
                                        data['usage']['completion_tokens'] = 0  # Initialize to 0
                                # Add the object field and reconstruct
                                data['object'] = 'chat.completion'
                                modified_chunk = json.dumps(data)
                                yield modified_chunk.encode('utf-8')
                            except json.JSONDecodeError:
                                # If we can't parse the JSON, add object field directly
                                modified_chunk = '{"object": "chat.completion",' + chunk_str.lstrip('{')
                                yield modified_chunk.encode('utf-8')
                        else:
                            yield chunk[0]

    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        yield json.dumps({"object": "chat.completion", "error": str(e)}).encode("utf-8")

async def handle_chat_completion(request: Request) -> StreamingResponse:
    """
    Common handler for chat completion requests.
    
    Args:
        request: The incoming FastAPI request
        
    Returns:
        StreamingResponse: Server-sent events response
        
    Raises:
        HTTPException: If request preprocessing fails
    """
    try:
        # Validate API key from headers
        validate_api_key(
            request.headers.get("Authorization"),
            request.headers.get("api-key")
        )

        request_body = await request.json()
        logger.info(f"Received request: {json.dumps(request_body, indent=2)}")
        processed_body = preprocess_request_body(request_body)
        
        return StreamingResponse(
            create_stream_response(processed_body),
            media_type="text/event-stream",
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(400, f"Error processing request: {str(e)}")



@app.post("/chat/completions")
async def proxy_chat_completions(request: Request) -> StreamingResponse:
    """Proxies chat completion requests with SSE support."""
    return await handle_chat_completion(request)

@app.get("/api/tags")
async def get_tags():
    """Returns available model tags, similar to Ollama's API."""
    return {
        "models": [
            {
                "name": "gpt-4o",
                "modified_at": "",
                "size": 0,
                "digest": "",
                "details": {
                    "format": "gguf",
                    "family": "github-copilot",
                    "parameter_size": "unknown",
                    "quantization_level": "none"
                }
            },
            {
                "name": "claude-3-5-sonnet",
                "modified_at": "",
                "size": 0,
                "digest": "",
                "details": {
                    "format": "gguf",
                    "family": "anthropic",
                    "parameter_size": "unknown",
                    "quantization_level": "none"
                }
            }
        ]
    }

@app.post("/v1/chat/completions")
async def v1_proxy_chat_completions(request: Request) -> StreamingResponse:
    """Proxies v1 chat completion requests with SSE support."""
    return await handle_chat_completion(request)
