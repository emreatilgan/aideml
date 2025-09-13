"""Backend for OpenRouter API"""

import json
import logging
import os
import time

from funcy import notnone, once, select_values
import openai

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    backoff_create,
    opt_messages_to_list,
)

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openrouter_client():
    global _client
    _client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=0,
    )

def _append_json_mode_instruction(messages: list[dict[str, str]], func_spec: FunctionSpec) -> list[dict[str, str]]:
    instr = (
        f"You must produce ONLY a single JSON object for function '{func_spec.name}' "
        f"that matches this JSON Schema exactly (no extra keys, no prose, no markdown):\n"
        f"{json.dumps(func_spec.json_schema)}"
    )
    # Append as a user message to be most universally honored
    return messages + [{"role": "user", "content": instr}]

def _extract_json(s: str) -> dict:
    # direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # try to grab the largest {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start : end + 1]
        return json.loads(snippet)
    # last resort: raise
    raise json.JSONDecodeError("Could not parse JSON from model output", s, 0)

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openrouter_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(
        system_message, user_message, convert_system_to_user=convert_system_to_user
    )

    use_tools = func_spec is not None
    if use_tools:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    def do_request(msgs: list[dict[str, str]], kwargs: dict):
        return backoff_create(
            _client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=msgs,
            extra_body={
                "provider": {
                    "order": ["Fireworks"],
                    "ignore": ["Together", "DeepInfra", "Hyperbolic"],
                },
            },
            **kwargs,
        )

    t0 = time.time()
    completion = None

    try:
        completion = do_request(messages, filtered_kwargs)
    except (openai.NotFoundError, openai.BadRequestError) as e:
        # Likely: tools not supported by selected provider/model
        logger.info(f"Tools likely unsupported by provider/model; falling back to JSON mode: {e}")
        use_tools = False

    # If the first attempt didn't succeed or tools are disabled after catching, try JSON-mode fallback
    if completion is None and func_spec is not None and not use_tools:
        json_messages = _append_json_mode_instruction(messages, func_spec)
        # Remove tool kwargs if present
        filtered_no_tools = {k: v for k, v in filtered_kwargs.items() if k not in ("tools", "tool_choice")}
        completion = do_request(json_messages, filtered_no_tools)

    # If completion is still None, bubble up whatever error occurred in do_request
    if completion is None:
        # backoff_create returns False on retryable exceptions; treat as error here to surface clearly
        raise RuntimeError("OpenRouter request failed (completion is None/False)")

    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output: OutputType = choice.message.content
    else:
        # If tools worked, parse tool call; otherwise parse JSON content
        if getattr(choice.message, "tool_calls", None):
            try:
                assert choice.message.tool_calls[0].function.name == func_spec.name
                output = json.loads(choice.message.tool_calls[0].function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding tool function arguments: {choice.message.tool_calls[0].function.arguments}")
                raise e
        else:
            # JSON-mode fallback (no tools)
            try:
                output = _extract_json(choice.message.content or "")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON in fallback mode: {choice.message.content}")
                raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
