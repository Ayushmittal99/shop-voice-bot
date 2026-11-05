
"""
Shopping agent: Groq LLM + MCP tools.

Features:
- MCP tool calling
- Tool start/completion/failure logging
- Per-tool execution timings
- Reduced conversation history and tool output size
- Product IDs for frontend cards
- Cart change detection
- Graceful Groq API error handling
"""

import asyncio
import json
import re
import socket
import time
from contextlib import AsyncExitStack
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from config import (
    AGENT_MAX_STEPS,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    MAX_HISTORY_TURNS,
    MCP_URL,
    REPLY_LANGUAGE,
    SYSTEM_PROMPT,
)
from generator import _get_client
from logger_setup import get_logger, log_event

logger = get_logger()

# Example product line:
# 1. [P012] Product Name | price ...
PRODUCT_LINE_PATTERN = re.compile(
    r"^\s*\d+\.\s*\[([^\]]+)\]\s*([^|\n]+?)\s*\|",
    re.MULTILINE,
)

PRODUCT_PATTERN = re.compile(
    r"\[([^\]]+)\]\s*([^|\n]+?)\s*\|"
)

MAX_HISTORY_MESSAGE_CHARS = 1000
MAX_TOOL_RESULT_CHARS = 2500
MAX_QUESTION_CHARS = 2000


class MCPUnavailable(RuntimeError):
    """Raised when the MCP server cannot be reached."""


def _shown_ids(tool_outputs: list[tuple[str, str]]) -> list[str]:
    """Return product IDs from the latest local search/compare result."""

    for name, result in reversed(tool_outputs):
        if name not in {"search_products", "compare_products"}:
            continue

        matches = PRODUCT_PATTERN.findall(result)

        if matches:
            return list(
                dict.fromkeys(
                    product_id.strip()
                    for product_id, _ in matches
                )
            )

    return []


def _shown_note(tool_outputs: list[tuple[str, str]]) -> str:
    """Save the latest displayed products in conversation history."""

    for name, result in reversed(tool_outputs):
        if name not in {"search_products", "compare_products"}:
            continue

        matches = PRODUCT_LINE_PATTERN.findall(result)

        if matches:
            products = [
                f"{index}. {product_id.strip()} {title.strip()}"
                for index, (product_id, title)
                in enumerate(matches[:8], start=1)
            ]

            return " (Products I showed: " + "; ".join(products) + ")"

        if name == "compare_products":
            matches = PRODUCT_PATTERN.findall(result)

            if matches:
                products = [
                    f"{product_id.strip()} {title.strip()}"
                    for product_id, title in matches[:4]
                ]

                return " (Products I showed: " + "; ".join(products) + ")"

    return ""


def _trim_history(history: list[dict] | None) -> list[dict]:
    """
    Keep a small number of recent user/assistant messages.

    Tool-call history is excluded because its corresponding tool results
    may not be present in the saved conversation history.
    """

    if not history:
        return []

    max_messages = max(0, int(MAX_HISTORY_TURNS)) * 2

    if max_messages == 0:
        return []

    recent = history[-max_messages:]
    trimmed = []

    for item in recent:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content", "")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        trimmed.append(
            {
                "role": role,
                "content": content[-MAX_HISTORY_MESSAGE_CHARS:],
            }
        )

    # Avoid beginning the conversation with an assistant message.
    while trimmed and trimmed[0]["role"] != "user":
        trimmed.pop(0)

    return trimmed


def _trim_tool_result(
    result: str,
    limit: int = MAX_TOOL_RESULT_CHARS,
) -> str:
    """Limit large tool responses sent back to the LLM."""

    if len(result) <= limit:
        return result

    return (
        result[:limit]
        + "\n[Tool output shortened to reduce request size.]"
    )


def _print_tool_log(
    status: str,
    name: str,
    duration: float | None = None,
    details: str = "",
) -> None:
    """Print readable tool execution information in the terminal."""

    timestamp = time.strftime("%H:%M:%S")

    message = f"[{timestamp}] [TOOL {status}] {name}"

    if duration is not None:
        message += f" | Duration: {duration:.3f}s"

    if details:
        message += f" | {details}"

    print("\n" + "=" * 65, flush=True)
    print(message, flush=True)
    print("=" * 65, flush=True)


class ShopAgent:
    """Use with: async with ShopAgent() as agent."""

    def __init__(self, url: str = MCP_URL):
        self.url = url
        self.system = SYSTEM_PROMPT.format(language=REPLY_LANGUAGE)
        self.tools: list[dict] = []
        self.session: ClientSession | None = None
        self._stack = AsyncExitStack()

    def _preflight(self):
        """Check whether the MCP server port is reachable."""

        parsed = urlparse(self.url)

        try:
            with socket.create_connection(
                (parsed.hostname, parsed.port or 80),
                timeout=2,
            ):
                pass

        except OSError as exc:
            raise MCPUnavailable(
                f"Cannot reach MCP server at {self.url}. "
                "Start it using: python mcp_server.py"
            ) from exc

    async def __aenter__(self):
        self._preflight()

        try:
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self.url)
            )

            self.session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )

            await self.session.initialize()
            listed = await self.session.list_tools()

        except BaseException as exc:
            try:
                await self._stack.aclose()
            except Exception:
                pass

            task = asyncio.current_task()

            externally_cancelled = (
                isinstance(exc, asyncio.CancelledError)
                and task is not None
                and getattr(task, "cancelling", lambda: 0)()
            )

            if (
                isinstance(exc, (KeyboardInterrupt, SystemExit))
                or externally_cancelled
            ):
                raise

            raise MCPUnavailable(
                f"Could not initialize MCP server at {self.url}. "
                f"Error type: {type(exc).__name__}. "
                "Check that python mcp_server.py is running."
            ) from exc

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in listed.tools
        ]

        log_event(
            logger,
            "Agent connected",
            url=self.url,
            tools=[
                item["function"]["name"]
                for item in self.tools
            ],
        )

        print("\n[AGENT READY] Connected to MCP server.", flush=True)
        print(
            "[AVAILABLE TOOLS] "
            + ", ".join(
                item["function"]["name"]
                for item in self.tools
            ),
            flush=True,
        )

        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def _call_tool(self, name: str, args: dict) -> str:
        """Execute an MCP tool and return its text result."""

        if self.session is None:
            return "Error: MCP session is not initialized."

        try:
            result = await self.session.call_tool(name, args)

            output = "\n".join(
                item.text
                for item in result.content
                if getattr(item, "text", None)
            )

            if getattr(result, "isError", False):
                output = "Tool error: " + (
                    output or "Unknown tool error."
                )

            return _trim_tool_result(output or "(no output)")

        except Exception as exc:
            logger.exception("MCP tool execution failed: %s", name)

            return (
                f"Error: tool {name} failed "
                f"({type(exc).__name__}). "
                "Try again with different arguments."
            )

    async def ask(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> dict:
        """
        Returns:
        answer, tools_used, product_ids, cart_changed,
        history_entries and timings.
        """

        request_start = time.perf_counter()

        clean_question = (question or "").strip()[:MAX_QUESTION_CHARS]

        if not clean_question:
            return {
                "answer": "Please enter a product question.",
                "tools_used": [],
                "product_ids": [],
                "cart_changed": False,
                "history_entries": [],
                "timings": {
                    "llm_seconds": 0.0,
                    "tools_seconds": 0.0,
                    "total_seconds": 0.0,
                },
            }

        messages = [
            {"role": "system", "content": self.system},
            *_trim_history(history),
            {"role": "user", "content": clean_question},
        ]

        tools_used: list[str] = []
        outputs: list[tuple[str, str]] = []

        llm_seconds = 0.0
        tools_seconds = 0.0
        answer = ""

        client = _get_client()
        max_steps = max(1, int(AGENT_MAX_STEPS))

        for step in range(max_steps):
            llm_start = time.perf_counter()

            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=GROQ_MODEL,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=GROQ_MAX_TOKENS,
                    temperature=0.2,
                )

            except Exception as exc:
                llm_seconds += time.perf_counter() - llm_start

                logger.exception("Groq request failed")

                error_text = str(exc).lower()

                if (
                    "413" in error_text
                    or "request too large" in error_text
                    or "tokens per minute" in error_text
                ):
                    answer = (
                        "The AI request exceeded the current token limit. "
                        "Try a shorter question. If the issue continues, "
                        "the system prompt or tool definitions also need "
                        "to be shortened."
                    )

                elif (
                    "429" in error_text
                    or "rate_limit" in error_text
                ):
                    answer = (
                        "The AI service is temporarily rate-limited. "
                        "Please wait a moment and try again."
                    )

                else:
                    answer = (
                        "Sorry, the AI service request failed. "
                        "Check the backend terminal for the error."
                    )

                print(
                    f"\n[LLM ERROR] {type(exc).__name__}: {answer}",
                    flush=True,
                )

                break

            llm_seconds += time.perf_counter() - llm_start

            if not response.choices:
                answer = "Sorry, the AI model returned no response."
                break

            message = response.choices[0].message
            calls = message.tool_calls or []

            # The LLM has produced its final text response.
            if not calls:
                answer = (message.content or "").strip()
                break

            assistant_tool_message = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments or "{}",
                        },
                    }
                    for call in calls
                ],
            }

            messages.append(assistant_tool_message)

            for call in calls:
                tool_name = call.function.name

                try:
                    arguments = json.loads(
                        call.function.arguments or "{}"
                    ) or {}

                    if not isinstance(arguments, dict):
                        arguments = {}

                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                tool_start = time.perf_counter()

                _print_tool_log(
                    "STARTED",
                    tool_name,
                    details="Executing MCP tool",
                )

                print(
                    "[TOOL INPUT] "
                    + json.dumps(
                        arguments,
                        ensure_ascii=False,
                        default=str,
                    )[:1000],
                    flush=True,
                )

                log_event(
                    logger,
                    "Tool started",
                    tool=tool_name,
                    args=arguments,
                )

                # _call_tool catches MCP execution errors and returns
                # an error string, so the LLM can receive the result.
                tool_output = await self._call_tool(
                    tool_name,
                    arguments,
                )

                tool_duration = time.perf_counter() - tool_start
                tools_seconds += tool_duration

                tools_used.append(tool_name)
                outputs.append((tool_name, tool_output))

                is_error = tool_output.startswith("Error:")

                _print_tool_log(
                    "FAILED" if is_error else "COMPLETED",
                    tool_name,
                    duration=tool_duration,
                    details=(
                        "Tool returned an error"
                        if is_error
                        else f"Output: {len(tool_output)} characters"
                    ),
                )

                print(
                    "[TOOL OUTPUT] "
                    + tool_output[:500].replace("\n", " "),
                    flush=True,
                )

                log_event(
                    logger,
                    "Tool completed" if not is_error else "Tool returned error",
                    tool=tool_name,
                    duration_seconds=round(tool_duration, 3),
                    result_chars=len(tool_output),
                    is_error=is_error,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_output,
                    }
                )

            # Before another LLM request, keep only the latest exchange.
            # This prevents old tool outputs from accumulating each round.
            if step < max_steps - 1:
                latest_exchange = messages[-(len(calls) + 1):]

                messages = [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": clean_question},
                    *latest_exchange,
                ]

        if not answer:
            answer = (
                "Sorry, I couldn't complete the request. "
                "Please try a more specific product query."
            )

        total_seconds = time.perf_counter() - request_start

        timings = {
            "llm_seconds": round(llm_seconds, 3),
            "tools_seconds": round(tools_seconds, 3),
            "total_seconds": round(total_seconds, 3),
        }

        log_event(
            logger,
            "Answer timing",
            question=clean_question,
            tools=tools_used,
            **timings,
        )

        print("\n" + "-" * 65, flush=True)
        print(f"[REQUEST COMPLETE] {clean_question}", flush=True)
        print(
            "[TOOLS USED] "
            + (", ".join(tools_used) if tools_used else "None"),
            flush=True,
        )
        print(
            f"[LATENCY] LLM: {timings['llm_seconds']:.3f}s | "
            f"Tools: {timings['tools_seconds']:.3f}s | "
            f"Total: {timings['total_seconds']:.3f}s",
            flush=True,
        )
        print("-" * 65, flush=True)

        note = _shown_note(outputs)

        return {
            "answer": answer,
            "tools_used": tools_used,
            "product_ids": _shown_ids(outputs),
            "cart_changed": any(
                name in {"add_to_cart", "remove_from_cart"}
                for name in tools_used
            ),
            "history_entries": [
                {"role": "user", "content": clean_question},
                {
                    "role": "assistant",
                    "content": answer + note,
                },
            ],
            "timings": timings,
        }
