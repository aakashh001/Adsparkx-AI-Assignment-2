"""
Interactive command-line chatbot for the persona-adaptive support agent.

Usage:
    python cli.py
"""
import uuid
from app.agent import run_turn

BANNER = """
==================================================================
 Adsparkx AI - Persona-Adaptive Customer Support Agent (CLI demo)
==================================================================
Type your message and press Enter. Type 'exit' to quit, 'reset' to
start a new session.
"""


def main():
    print(BANNER)
    session_id = str(uuid.uuid4())
    print(f"[session_id: {session_id}]\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not message:
            continue
        if message.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if message.lower() == "reset":
            session_id = str(uuid.uuid4())
            print(f"\n[new session_id: {session_id}]\n")
            continue

        result = run_turn(session_id, message)

        print(f"\n  Detected persona : {result.persona}  (scores: {result.persona_scores})")
        if result.retrieved_sources:
            print("  Retrieved sources:")
            for s in result.retrieved_sources:
                print(f"    - {s['source']} | {s['section']} | score={s['score']}")
        else:
            print("  Retrieved sources: none")

        print(f"\nAgent: {result.response}\n")

        if result.escalated:
            print("  >>> ESCALATED TO HUMAN AGENT <<<")
            print("  Handoff summary:")
            import json
            print(json.dumps(result.handoff_summary, indent=2))
        print("-" * 66)


if __name__ == "__main__":
    main()
