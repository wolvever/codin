"""Tests for the codin.prompt.run.prompt_render function."""

import pytest
from typing import Any

from codin.prompt import prompt_render
from codin.prompt.registry import get_registry, PromptRegistry
from codin.prompt.base import PromptTemplate, PromptVariant

# The env_setup fixture from tests/conftest.py will be automatically used
# as it's defined there and sets up PROMPT_TEMPLATE_DIR.

@pytest.mark.asyncio
async def test_prompt_render_basic_render(env_setup):
    """Test basic rendering with a manually registered template."""
    registry = get_registry() # Gets the global registry affected by env_setup

    # Manually register a simple template for this test
    simple_template_text = "Hello {{ name }} from basic test!"
    template = PromptTemplate(name="prompt_render_basic", text=simple_template_text) # Updated template name
    registry.register(template)

    result = await prompt_render("prompt_render_basic", variables={"name": "Tester"}) # Updated template name
    assert result == "Hello Tester from basic test!"

@pytest.mark.asyncio
async def test_prompt_render_from_fixture_file(env_setup):
    """Test rendering a template loaded from the fixture directory."""
    # This test relies on 'test.jinja2' being in 'tests/fixtures/prompts/'
    # and env_setup correctly pointing the registry to it.
    # Content of test.jinja2:
    # You are a helpful assistant.
    #
    # I need help with {{ topic }}.
    #
    # Please provide a detailed response in {{ format }} format.

    result = await prompt_render(
        "test", # Name of the template file (without .jinja2)
        variables={"topic": "fixture file loading", "format": "plain text"}
    )

    assert "I need help with fixture file loading." in result
    assert "Please provide a detailed response in plain text format." in result

@pytest.mark.asyncio
async def test_prompt_render_template_not_found(env_setup):
    """Test that prompt_render raises KeyError for a non-existent template."""
    # Updated template name in match string for clarity, though not strictly necessary if only one such test.
    with pytest.raises(KeyError, match="Template 'non_existent_template_for_prompt_render' version 'latest' not found"):
        await prompt_render("non_existent_template_for_prompt_render")

@pytest.mark.asyncio
async def test_prompt_render_with_version_from_fixture(env_setup):
    """Test rendering a specific version of a template from fixture files."""
    # Content of tests/fixtures/prompts/test.v1.jinja2:
    # This is version 1 of the test prompt.
    #
    # Please help me with {{ topic }} and explain it like I'm {{ audience }}.

    result = await prompt_render(
        "test",
        version="v1",
        variables={"topic": "versioned prompts", "audience": "a five year old"}
    )

    expected_text = (
        "This is version 1 of the test prompt.\n\n"
        "Please help me with versioned prompts and explain it like I'm a five year old."
    )
    # Normalize whitespace/trailing newlines for comparison if necessary,
    # but Jinja rendering should be exact.
    assert result.strip() == expected_text.strip()


@pytest.mark.asyncio
async def test_prompt_render_with_conditions(env_setup):
    """Test prompt_render with conditions selecting template variants."""
    registry = get_registry()

    template = PromptTemplate(name="prompt_render_conditional") # Updated template name
    variant_A_text = "Variant A for {{ user }} on channel A."
    variant_B_text = "Variant B for {{ user }} on channel B."

    template.add_variant(PromptVariant(text=variant_A_text, conditions={"channel": "A"}))
    template.add_variant(PromptVariant(text=variant_B_text, conditions={"channel": "B"}))
    registry.register(template)

    result_A = await prompt_render(
        "prompt_render_conditional", # Updated template name
        conditions={"channel": "A"},
        variables={"user": "Alice"}
    )
    assert result_A == "Variant A for Alice on channel A."

    result_B = await prompt_render(
        "prompt_render_conditional", # Updated template name
        conditions={"channel": "B"},
        variables={"user": "Bob"}
    )
    assert result_B == "Variant B for Bob on channel B."

    # Test fallback to the first registered variant if conditions don't match strongly
    # or if no conditions are provided to prompt_render where variants have conditions
    result_fallback = await prompt_render(
        "prompt_render_conditional", # Updated template name
        conditions={"other_condition": "X"}, # No variant explicitly matches this
        variables={"user": "Charles"}
    )
    # Default behavior is to return the first variant if no specific match is better
    assert result_fallback == "Variant A for Charles on channel A."

    result_no_condition = await prompt_render(
        "prompt_render_conditional", # Updated template name
        variables={"user": "David"} # No conditions passed to prompt_render
    )
    # Should also default to the first variant
    assert result_no_condition == "Variant A for David on channel A."
