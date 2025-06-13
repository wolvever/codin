from ..planner import BasicPlanner


class CodingAssistantPlanner(BasicPlanner):
    """Specialized planner for coding tasks."""

    def _default_system_prompt(self) -> str:
        return (
            "You are an expert coding assistant. Respond with JSON actions for programming tasks.\n\n"
            "CODING WORKFLOW:\n"
            "1. Read relevant files using tools\n"
            "2. Analyze code structure\n"
            "3. Make necessary changes\n"
            "4. Run tests to verify\n\n"
            "TOOLS: {available_tools}\n"
            "HISTORY: {recent_history}\n"
        )
