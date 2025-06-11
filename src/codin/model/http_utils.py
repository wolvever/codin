"""
Utility functions for handling HTTP requests and responses, particularly for
interacting with LLM APIs, including Server-Sent Event (SSE) stream processing.
"""
from __future__ import annotations

import json
import logging
import typing as _t

if _t.TYPE_CHECKING:
    from httpx import Response as HttpxResponse
    from ..client import Client # Assuming Client is httpx.AsyncClient or wraps it

logger = logging.getLogger(__name__)

# Custom Exception Hierarchy
class ModelAPIError(Exception):
    """Base class for exceptions raised by model API interactions."""
    pass

class ContentExtractionError(ModelAPIError): # Inherits from ModelAPIError
    """Raised when content cannot be extracted from a response, often due to unexpected structure
       or if an extractor function explicitly signals not found (e.g. returns None)."""
    pass

class StreamProcessingError(ModelAPIError): # Inherits from ModelAPIError
    """Raised during an issue with processing a stream, such as parsing errors or delta extraction failures."""
    pass

class ModelResponseParsingError(ModelAPIError): # Inherits from ModelAPIError
    """Raised for issues when the response structure is not as expected (e.g., missing keys, wrong types)."""
    pass


async def make_post_request(
    client: Client,
    endpoint: str,
    json_payload: dict,
    error_message_prefix: str = "API request failed"
) -> HttpxResponse:
    """
    Makes a POST request using the provided client and handles initial response validation.

    Args:
        client: The HTTP client instance.
        endpoint: The API endpoint to call.
        json_payload: The JSON payload for the POST request.
        error_message_prefix: Prefix for error messages.

    Returns:
        The httpx.Response object.

    Raises:
        Exception: If the request fails or returns a non-2xx status code.
    """
    try:
        response = await client.post(endpoint, json=json_payload)
        response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
        return response
    except Exception as e:
        logger.error(f"{error_message_prefix}: {e} (Payload: {json_payload.get('messages', json_payload)})") # Log part of payload for context
        raise


def extract_content_from_json(
    response_data: dict,
    extractor: _t.Callable[[dict], _t.Optional[str]],
    error_message_prefix: str = "Failed to extract content"
) -> str:
    """
    Extracts string content from parsed JSON data using a model-specific extractor.

    Args:
        response_data: The parsed JSON response as a dictionary.
        extractor: A callable that takes the JSON data and returns the string content.
        error_message_prefix: Prefix for error messages.

    Returns:
        The extracted string content.

    Raises:
        ContentExtractionError: If the extractor fails to find content or returns None.
    """
    try:
        content = extractor(response_data)
        if content is None: # Extractor signals not found with None by returning None
            # This case is specific: the structure might be valid, but the extractor logic didn't find the content.
            # It's often due to an empty field, like an empty 'content' in a message.
            raise ContentExtractionError(f"{error_message_prefix}: Extractor returned None, indicating content not found as expected. Data: {response_data}")
        return content
    except ContentExtractionError: # Re-raise if it's already the correct type
        raise
    except (KeyError, IndexError, TypeError) as e_parse: # Specific parsing/structure errors
        logger.error(f"{error_message_prefix}: Parsing error - {type(e_parse).__name__}: {e_parse}. Data: {response_data}")
        raise ModelResponseParsingError(f"{error_message_prefix}: Parsing error - {type(e_parse).__name__}: {e_parse}") from e_parse
    except Exception as e_other: # Catch any other unexpected error from the extractor
        logger.error(f"{error_message_prefix}: Unexpected error during extraction - {type(e_other).__name__}: {e_other}. Data: {response_data}")
        raise ContentExtractionError(f"{error_message_prefix}: Unexpected error - {type(e_other).__name__}: {e_other}") from e_other


async def process_sse_stream(
    response: HttpxResponse,
    delta_extractor: _t.Callable[[dict], _t.Optional[str]],
    stop_marker: str = "[DONE]", # OpenAI specific, others might use different signals or rely on stream end
    event_filter: _t.Optional[_t.Callable[[str, dict], bool]] = None, # Optional filter based on event type and data
    error_message_prefix: str = "SSE stream processing failed"
) -> _t.AsyncIterator[str]:
    """
    Processes a Server-Sent Events (SSE) stream and yields content deltas.

    Args:
        response: The httpx.Response object from a streaming request.
        delta_extractor: A callable that takes a parsed SSE data chunk (dict)
                         and returns the text delta, or None to skip.
        stop_marker: The string indicating the end of the stream in a data line.
        event_filter: An optional callable to filter SSE events by event type and data.
                      It should return True if the event should be processed, False otherwise.
                      The callable receives event_type (str, e.g. "message_delta") and data (dict).
        error_message_prefix: Prefix for error messages.

    Yields:
        String chunks of content from the stream.

    Raises:
        StreamProcessingError: If there's an issue with JSON decoding or delta extraction.
    """
    current_event_type = None # For multi-line event types if they occur
    try:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("event: "):
                current_event_type = line[len("event: "):].strip()
                continue # Wait for data line

            if line.startswith("data: "):
                data_str = line[len("data: "):].strip()
                if data_str == stop_marker:
                    break

                try:
                    data_chunk = json.loads(data_str)
                except json.JSONDecodeError as e_json:
                    logger.warning(f"{error_message_prefix}: Failed to parse SSE JSON: '{data_str}'. Error: {e_json}")
                    # Optionally, re-raise as a type of StreamProcessingError if strict parsing is needed.
                    # For now, skipping malformed lines is often preferred in streaming.
                    continue

                if event_filter and not event_filter(current_event_type or "", data_chunk):
                    current_event_type = None # Reset after use
                    continue

                try:
                    delta_content = delta_extractor(data_chunk)
                    if delta_content:
                        yield delta_content
                except Exception as e_extract:
                    logger.error(f"{error_message_prefix}: Error extracting delta: {e_extract}. Data: {data_chunk}")
                    # Raising the error makes issues more visible.
                    # If specific extractors need to be lenient, they can handle their own errors.
                    raise StreamProcessingError(f"Delta extraction failed for chunk {data_chunk}: {e_extract}") from e_extract

                current_event_type = None # Reset after processing data for this event

    except Exception as e:
        logger.error(f"{error_message_prefix}: {e}")
        raise StreamProcessingError(f"{error_message_prefix}: An unexpected error occurred - {e}") from e
    finally:
        await response.aclose()
