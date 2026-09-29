"""Local Slack Socket Mode bot. Install requirements and set environment secrets."""
import os
import re
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from policy_engine import Gemini, PolicyError, load_policies, build_index, answer, DEFAULT_MODEL, DEFAULT_EMBEDDING


def main():
    records = load_policies()
    method = os.getenv("SLACK_POLICY_METHOD", "rules")
    if method not in ("rules", "full_context", "rag"):
        raise PolicyError("SLACK_POLICY_METHOD must be rules, full_context or rag.")
    client = Gemini(os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", DEFAULT_MODEL), os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING)) if method != "rules" else None
    index = build_index(client, records)[0] if method == "rag" else None
    if not os.getenv("SLACK_BOT_TOKEN") or not os.getenv("SLACK_APP_TOKEN"):
        raise PolicyError("Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN locally.")
    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    @app.event("app_mention")
    def handle_mention(event, say):
        question = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", "")).strip()
        thread = event.get("thread_ts", event["ts"])
        try:
            result = answer(question, method, records, client, index)
            sources = "\n".join(f"{r['id']} — {r['title']}\n{r['policy_text']}" for r in result["policies"])
            text = f"{result['answer']}\n\nSources:\n{sources or 'No valid policy cited.'}\n\nMethod: {method}"
            if result["invalid_policy_ids"]:
                text += "\nWarning: the model also returned invalid policy citations."
        except PolicyError as exc:
            text = str(exc)
        # Plain-text blocks prevent generated strings from notifying Slack users.
        say(text="Policy assistant response", blocks=[{"type": "section", "text": {"type": "plain_text", "text": text[:2900]}}], thread_ts=thread)

    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
