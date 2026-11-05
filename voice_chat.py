"""Voice shopping assistant: SPEAK or TYPE what you want, hear the answer.

Usage (two terminals):
    python mcp_server.py      # tools + product search (keep running)
    python voice_chat.py

  - Type a request and press Enter          -> typed request
  - Press Enter on an empty line            -> speak (Enter again to stop)
  - 'mute' / 'unmute'                       -> turn spoken answers off / on
  - 'clear' forgets the conversation, 'exit' quits.

Try: "I need running shoes under 3000 rupees" ... "add the second one to my cart" ... "what's in my cart?"

If the microphone, speech-to-text or speaker fails, the bot says so and keeps working in
text mode. Timing for every step goes to logs/pipeline.log.
"""
import asyncio
import sys
import time

from agent import MCPUnavailable, ShopAgent
from config import MAX_HISTORY_TURNS
from logger_setup import get_logger, log_event

logger = get_logger()


def get_question() -> tuple[str, float]:
    """Returns (question, speech_to_text_seconds).
    Typed input always works; empty Enter tries the microphone and falls back to typing."""
    typed = input("You (type, or Enter to speak): ").strip()
    if typed:
        return typed, 0.0

    try:
        from voice import record_until_enter, speech_to_text

        wav = record_until_enter()
        if not wav:
            print("  Didn't hear anything. Try again, or type your request.\n")
            return "", 0.0
        t = time.perf_counter()
        question = speech_to_text(wav)
        seconds = round(time.perf_counter() - t, 3)
        if not question:
            print("  Couldn't understand that. Try again, or type your request.\n")
            return "", seconds
        print(f"  You said: {question}")
        return question, seconds
    except Exception as e:
        logger.exception("Voice input failed")
        print(f"  Voice input isn't working ({type(e).__name__}). Please type your request.\n")
        return "", 0.0


async def main():
    speak = True
    history: list[dict] = []

    try:
        async with ShopAgent() as agent:
            print("\nShopping assistant ready. Type a request, or press Enter to speak.")
            print("Commands: mute, unmute, clear, exit\n")
            log_event(logger, "Voice chat session started")

            while True:
                try:
                    # blocking input/recording/speaking is fine: one customer, one turn at a time
                    question, stt_seconds = get_question()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                low = question.lower()
                if low in {"exit", "quit"}:
                    break
                if low == "clear":
                    history.clear()
                    print("Bot: Conversation cleared.\n")
                    continue
                if low == "mute":
                    speak = False
                    print("Bot: Spoken answers off (text only).\n")
                    continue
                if low == "unmute":
                    speak = True
                    print("Bot: Spoken answers on.\n")
                    continue
                if not question:
                    continue

                try:
                    result = await agent.ask(question, history)
                except Exception:
                    logger.exception("Shopping agent failed")
                    print("Bot: Something went wrong. Check logs/pipeline.log.\n")
                    continue

                # the text answer is always shown first, so it works even if speaking fails
                print(f"\nBot: {result['answer']}")
                if result["tools_used"]:
                    print("Tools: " + ", ".join(result["tools_used"]))

                tts_seconds = 0.0
                if speak:
                    try:
                        from voice import text_to_speech

                        tts_seconds = text_to_speech(result["answer"])
                    except Exception as e:
                        logger.exception("Text to speech failed")
                        speak = False
                        print(f"  (Couldn't speak the answer: {type(e).__name__}. "
                              "Text-only from now on; type 'unmute' to retry.)")

                t = result["timings"]
                total = round(t["total_seconds"] + stt_seconds + tts_seconds, 3)
                log_event(
                    logger, "Voice answer timing", question=question,
                    speech_to_text_seconds=stt_seconds, llm_seconds=t["llm_seconds"],
                    tools_seconds=t["tools_seconds"], text_to_speech_seconds=tts_seconds,
                    total_seconds=total, tools=result["tools_used"],
                )
                print(f"[total {total}s | speech-to-text {stt_seconds}s | LLM {t['llm_seconds']}s"
                      f" | tools {t['tools_seconds']}s | speaking {tts_seconds}s]\n")

                history = (history + result["history_entries"])[-2 * MAX_HISTORY_TURNS:]

            log_event(logger, "Voice chat session ended")
    except MCPUnavailable as e:
        print(f"\n{e}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles can choke on non-ASCII characters
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
