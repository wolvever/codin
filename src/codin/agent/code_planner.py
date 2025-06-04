import asyncio
import json
import logging
import uuid
import typing as _t
from datetime import datetime
from dataclasses import dataclass

from a2a.types import Message, Role, TextPart

from .types import ToolCall
from .base import Planner
from .types import Step, StepType, ThinkStep, MessageStep, ToolCallStep, FinishStep, State
from ..model.base import BaseLLM
from ..model.factory import LLMFactory
from ..tool.base import to_tool_definitions
from ..tool.registry import ToolRegistry
from ..prompt import prompt_run

__all__ = [
    "CodePlanner",
    "CodePlannerConfig",
]

logger = logging.getLogger("codin.agent.code_planner")


@dataclass
class CodePlannerConfig:
    """Configuration for CodePlanner."""
    model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    max_tool_calls_per_turn: int = 10
    thinking_enabled: bool = True
    streaming_enabled: bool = True
    rules: str | None = None


class CodePlanner(Planner):
    """Planner that uses LLM to generate execution steps, adapted from CodeAgent."""
    
    def __init__(
        self,
        config: CodePlannerConfig | None = None,
        llm: BaseLLM | None = None,
        tool_registry: ToolRegistry | None = None,
        debug: bool = False
    ):
        """Initialize the CodePlanner.
        
        Args:
            config: Planner configuration
            llm: LLM instance (will create if not provided)
            tool_registry: Tool registry for available tools
            debug: Enable debug logging
        """
        self.config = config or CodePlannerConfig()
        self.llm = llm or LLMFactory.create_llm(model=self.config.model)
        self.tool_registry = tool_registry or ToolRegistry()
        self.debug = debug
        
        logger.info(f"Initialized CodePlanner with {len(self.tool_registry.get_tools())} tools")
    
    async def next(self, state: State) -> _t.AsyncGenerator[Step, None]:
        """Generate execution steps based on current state."""
        step_count = 0
        
        # Build the prompt variables from state
        variables = await self._build_prompt_variables(state)
        
        if self.debug:
            logger.debug(f"Planning for session {state.session_id}, turn {state.turn_count}")
            logger.debug(f"Variables: {variables}")
        
        try:
            # Get LLM response using prompt_run
            response = await prompt_run(
                "code_agent_loop",
                variables=variables,
                tools=variables.get("tools", []),
                stream=False
            )
            
            # Parse the structured response 
            parsed_response = self._parse_structured_response(response)
            
            # Emit thinking step if enabled and thinking is present
            if self.config.thinking_enabled and parsed_response.get("thinking"):
                yield ThinkStep(
                    step_id=str(uuid.uuid4()),
                    thinking=parsed_response["thinking"],
                    timestamp=datetime.now(),
                    metadata={"turn": state.turn_count}
                )
                step_count += 1
            
            # Emit tool call steps if any
            tool_calls = parsed_response.get("tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    yield ToolCallStep(
                        step_id=str(uuid.uuid4()),
                        tool_call=tool_call,
                        timestamp=datetime.now(),
                        metadata={"turn": state.turn_count}
                    )
                    step_count += 1
                    
                    # Respect max tool calls limit
                    if step_count >= self.config.max_tool_calls_per_turn:
                        break
            
            # Emit message step if there's a message
            message_content = parsed_response.get("message", "")
            if message_content:
                message = Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=message_content)],
                    contextId=state.session_id,
                    kind="message"
                )
                
                yield MessageStep(
                    step_id=str(uuid.uuid4()),
                    message=message,
                    is_streaming=self.config.streaming_enabled,
                    timestamp=datetime.now(),
                    metadata={"turn": state.turn_count}
                )
                step_count += 1
            
            # Check if task should finish
            should_continue = parsed_response.get("should_continue", True)
            if not should_continue or not tool_calls:
                # Task is complete
                final_message = None
                if message_content:
                    final_message = Message(
                        messageId=str(uuid.uuid4()),
                        role=Role.agent,
                        parts=[TextPart(text=message_content)],
                        contextId=state.session_id,
                        kind="message"
                    )
                
                yield FinishStep(
                    step_id=str(uuid.uuid4()),
                    final_message=final_message,
                    reason="Task completed based on LLM decision",
                    timestamp=datetime.now(),
                    metadata={
                        "turn": state.turn_count,
                        "should_continue": should_continue,
                        "had_tool_calls": len(tool_calls) > 0
                    }
                )
            
        except Exception as e:
            logger.error(f"Error in planner execution: {e}", exc_info=True)
            
            # Emit error message step
            error_message = Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text=f"I encountered an error while planning: {str(e)}")],
                contextId=state.session_id,
                kind="message"
            )
            
            yield MessageStep(
                step_id=str(uuid.uuid4()),
                message=error_message,
                timestamp=datetime.now(),
                metadata={"turn": state.turn_count, "error": str(e)}
            )
            
            # Finish with error
            yield FinishStep(
                step_id=str(uuid.uuid4()),
                final_message=error_message,
                reason=f"Planning error: {str(e)}",
                timestamp=datetime.now(),
                metadata={"turn": state.turn_count, "error": str(e)}
            )
    
    async def _build_prompt_variables(self, state: State) -> dict[str, _t.Any]:
        """Build variables for the prompt from current state."""
        # Get tool definitions
        tool_definitions_objects = to_tool_definitions(self.tool_registry.get_tools())
        tool_definitions = [
            {
                "name": td.name,
                "description": td.description,
                "parameters": self._clean_parameters(td.parameters)
            }
            for td in tool_definitions_objects
        ]
        
        # Format conversation history
        history_messages = []
        for msg in state.history[:-1]:  # All but the last message (if it's the current input)
            history_messages.append({
                "role": "user" if msg.role == Role.user else "assistant",
                "content": self._extract_text_from_message(msg)
            })
        
        # Get current user input (last message if it's from user)
        current_input = ""
        if state.history and state.history[-1].role == Role.user:
            current_input = self._extract_text_from_message(state.history[-1])
        
        # Format tool results from previous turn if any
        tool_results_text = ""
        if state.last_tool_results:
            tool_results_text = self._format_tool_results_for_conversation(state.last_tool_results)
        
        return {
            "agent_name": "CodePlanner",
            "task_id": state.session_id,
            "turn_count": state.turn_count,
            "has_tools": len(tool_definitions) > 0,
            "tools": tool_definitions,
            "has_history": len(history_messages) > 0,
            "history_text": self._format_history_for_prompt(history_messages),
            "user_input": current_input if current_input else None,
            "tool_results": bool(tool_results_text),
            "tool_results_text": tool_results_text,
            "task_list": state.task_list,
            "rules": self.config.rules
        }
    
    def _parse_structured_response(self, response) -> dict[str, _t.Any]:
        """Parse the structured response from prompt_run, adapted from CodeAgent."""
        # Extract content from response
        content = ""
        if hasattr(response, 'message') and response.message:
            # Extract text from A2A message parts
            for part in response.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    content += part.root.text
                elif hasattr(part, 'text'):
                    content += part.text
        elif hasattr(response, 'content'):
            content = str(response.content)
        else:
            content = str(response)
        
        # Initialize default response structure
        parsed_response = {
            "thinking": "",
            "task_list": {"completed": [], "pending": []},
            "tool_calls": [],
            "message": "",
            "should_continue": True,
            "raw_content": content
        }
        
        # Try to extract JSON from markdown code blocks first
        import re
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        response_data = None
        if json_matches:
            # Use the first JSON block found
            try:
                response_data = json.loads(json_matches[0])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from markdown block: {e}")
        
        # If no JSON block found, try parsing the entire content as JSON
        if not response_data:
            try:
                response_data = json.loads(content.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract information from text
                logger.debug("No valid JSON found, extracting from text content")
                parsed_response["message"] = content
                parsed_response["tool_calls"] = self._parse_tool_calls_from_text(content)
                return parsed_response
        
        # Extract fields from parsed JSON
        if response_data and isinstance(response_data, dict):
            parsed_response["thinking"] = response_data.get("thinking", "")
            parsed_response["message"] = response_data.get("message", "")
            parsed_response["should_continue"] = response_data.get("should_continue", True)
            
            # Extract task list
            task_list = response_data.get("task_list", {})
            if isinstance(task_list, dict):
                parsed_response["task_list"]["completed"] = task_list.get("completed", [])
                parsed_response["task_list"]["pending"] = task_list.get("pending", [])
            
            # Extract tool calls and convert to ToolCall objects
            tool_calls_data = response_data.get("tool_calls", [])
            if isinstance(tool_calls_data, list):
                for call_data in tool_calls_data:
                    if isinstance(call_data, dict) and "name" in call_data:
                        tool_call = ToolCall(
                            call_id=str(uuid.uuid4()),
                            name=call_data["name"],
                            arguments=call_data.get("arguments", {})
                        )
                        parsed_response["tool_calls"].append(tool_call)
        
        return parsed_response
    
    def _parse_tool_calls_from_text(self, content: str) -> list[ToolCall]:
        """Parse tool calls from text content when JSON parsing fails."""
        tool_calls = []
        
        # Pattern for function call tags
        import re
        pattern = r'<function_call name="([^"]+)">\s*(\{.*?\})\s*</function_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for tool_name, args_str in matches:
            try:
                arguments = json.loads(args_str)
                tool_call = ToolCall(
                    call_id=str(uuid.uuid4()),
                    name=tool_name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse arguments for tool {tool_name}: {args_str}")
        
        return tool_calls
    
    def _clean_parameters(self, parameters: dict) -> dict:
        """Clean parameters to remove Undefined values and make them JSON serializable."""
        def clean_value(value):
            """Recursively clean a value to make it JSON serializable."""
            if hasattr(value, '__class__') and value.__class__.__name__ == 'Undefined':
                return None
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items() 
                       if not (hasattr(v, '__class__') and v.__class__.__name__ == 'Undefined')}
            elif isinstance(value, (list, tuple)):
                return [clean_value(item) for item in value 
                       if not (hasattr(item, '__class__') and item.__class__.__name__ == 'Undefined')]
            elif isinstance(value, (str, int, float, bool, type(None))):
                return value
            else:
                # Try to convert to string for other types
                try:
                    json.dumps(value)  # Test if it's JSON serializable
                    return value
                except (TypeError, ValueError):
                    return str(value)
        
        return clean_value(parameters)
    
    def _extract_text_from_message(self, message: Message) -> str:
        """Extract text content from a Message object."""
        text_parts = []
        for part in message.parts:
            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                text_parts.append(part.root.text)
            elif hasattr(part, 'text'):
                text_parts.append(part.text)
        return "\n".join(text_parts)
    
    def _format_history_for_prompt(self, history_messages: list[dict]) -> str:
        """Format conversation history for the prompt."""
        if not history_messages:
            return ""
        
        formatted = []
        for msg in history_messages:
            role = msg["role"].title()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n\n".join(formatted)
    
    def _format_tool_results_for_conversation(self, tool_results: list) -> str:
        """Format tool results for inclusion in conversation."""
        if not tool_results:
            return ""
        
        formatted = []
        for result in tool_results:
            if hasattr(result, 'success'):
                # It's a ToolResult
                status = "✅ Success" if result.success else "❌ Failed"
                formatted.append(f"**Tool Call {result.call_id}** {status}")
                if result.output:
                    formatted.append(f"Output: {result.output}")
                if result.error:
                    formatted.append(f"Error: {result.error}")
            else:
                # Generic result
                formatted.append(f"Result: {str(result)}")
            formatted.append("")  # Empty line between results
        
        return "\n".join(formatted)
    
    def get_config(self) -> dict[str, _t.Any]:
        """Get planner configuration."""
        return {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "max_tool_calls_per_turn": self.config.max_tool_calls_per_turn,
            "thinking_enabled": self.config.thinking_enabled,
            "streaming_enabled": self.config.streaming_enabled,
            "rules": self.config.rules
        }
    
    async def cleanup(self) -> None:
        """Cleanup planner resources."""
        # LLM cleanup is handled by the LLM factory/client
        logger.debug("CodePlanner cleanup completed") 