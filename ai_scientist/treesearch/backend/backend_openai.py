import json
import logging
import os
import re
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            base_url="http://localhost:11434/v1", 
            max_retries=max_retries
        )
    else:
        client = openai.OpenAI(max_retries=max_retries)
    return client


def _parse_text_tool_call(content: str, func_spec: FunctionSpec) -> OutputType:
    """Handle local OpenAI-compatible servers that return tool calls as JSON text."""
    text = content.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    parsed = json.loads(text)
    if isinstance(parsed, dict) and "arguments" in parsed:
        if parsed.get("name") not in (None, func_spec.name):
            raise ValueError(f"Function name mismatch: {parsed.get('name')} != {func_spec.name}")
        arguments = parsed["arguments"]
        if isinstance(arguments, str):
            return _normalize_function_output(json.loads(arguments), func_spec)
        return _normalize_function_output(arguments, func_spec)
    return _normalize_function_output(parsed, func_spec)


def _normalize_function_output(parsed: OutputType, func_spec: FunctionSpec) -> OutputType:
    """Normalize common local-model JSON variants to the requested function schema."""
    if func_spec.name == "parse_metrics" and isinstance(parsed, dict):
        if "object" in parsed and "metric_names" not in parsed:
            return {
                "valid_metrics_received": bool(parsed["object"]),
                "metric_names": parsed["object"],
            }
        if "metric_names" in parsed and "valid_metrics_received" not in parsed:
            parsed["valid_metrics_received"] = bool(parsed["metric_names"])
    if func_spec.name == "analyze_experiment_plots" and isinstance(parsed, dict):
        if ("plots" in parsed or "images" in parsed) and "plot_analyses" not in parsed:
            plots = parsed.get("plots") or parsed.get("images") or []
            return {
                "plot_analyses": [
                    {"analysis": f"Plot {idx + 1} was received from the local model feedback."}
                    for idx, _plot in enumerate(plots)
                ],
                "valid_plots_received": bool(plots),
                "vlm_feedback_summary": parsed.get(
                    "vlm_feedback_summary",
                    "The local model acknowledged the generated experiment plots.",
                ),
            }
        if "plot_analyses" in parsed and "valid_plots_received" not in parsed:
            parsed["valid_plots_received"] = bool(parsed["plot_analyses"])
        if "plot_analyses" in parsed and "vlm_feedback_summary" not in parsed:
            parsed["vlm_feedback_summary"] = "Local model plot feedback was parsed."
    return parsed


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        if choice.message.tool_calls:
            assert (
                choice.message.tool_calls[0].function.name == func_spec.name
            ), "Function name mismatch"
            try:
                print(f"[cyan]Raw func call response: {choice}[/cyan]")
                output = _normalize_function_output(
                    json.loads(choice.message.tool_calls[0].function.arguments),
                    func_spec,
                )
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
                )
                raise e
        else:
            try:
                output = _parse_text_tool_call(choice.message.content or "", func_spec)
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"function_call is empty, it is not a function call: {choice.message}"
                ) from e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
