# Company Policy Assistant

Python assignment implementing three genuinely distinct approaches over the supplied 98-policy CSV, plus a Streamlit comparison website and a local Slack bot. Follows Lecture 5's retrieve–augment–generate workflow.

## Status

Implemented: three methods, website, measurements, manual review, result download, resumable embedding setup, benchmark runner and Slack bot code.

Not yet verified: live Gemini calls, the deployed website and a Slack workspace interaction. No AI measurements, preferred-method conclusion or Slack screenshot have been invented. These steps require account configuration and real testing.

## Start locally

Use Python 3.11 or newer. From this repository's folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows activate with `.venv\Scripts\activate` instead. Local Streamlit secrets go in `.streamlit/secrets.toml`, using `secrets.example.toml` as a template. Never commit API keys or Slack tokens.

## Deploy the website from GitHub

1. Sign in to https://share.streamlit.io/ using GitHub.
2. Create an app with repository `AeronTekle/company-policy-assistant`, branch `main`, entrypoint `app.py`.
3. Select Python 3.11 or newer if a version selector is offered.
4. In Advanced settings / Secrets paste the contents of `secrets.example.toml`, replacing only the key placeholder with your real Gemini key.
5. Deploy. Open the website and click **Prepare vector index** before comparing all three methods.

Model IDs are configurable: both AI methods use the same `GEMINI_MODEL`. Default is `gemini-3.5-flash-lite`; the embedding default is `gemini-embedding-001`. Check your key's actual access and quota: https://ai.google.dev/gemini-api/docs/models and https://aistudio.google.com/. No automatic paid fallback is used. Account billing determines whether usage is free, so this code cannot guarantee free access. A 404 means choose an available model; a 429 means quota has been exceeded. Embedding setup can resume after an interruption.

## How the methods differ

* **Rules:** a fixed stopword list, explicit synonyms and lexical overlap. Title matches are weighted more strongly. Returns up to three exact policy texts. This is a simple baseline; it may miss paraphrases or return irrelevant policies. It does not claim that every retrieved text answers the question.
* **LLM without vector index:** sends all CSV records and the question directly to Gemini. No keyword preselection and no vector retrieval.
* **LLM with vector index:** embeds policy records once, normalizes vectors and stores an exact cosine index. For each question, embeds the query with the same model, retrieves the five nearest records and gives their readable texts to the same Gemini answer model. This is a dense vector index implemented in Python; it is not a keyword or TF–IDF substitute. For 98 records, an external vector database is unnecessary.

The department field is shown as metadata; the CSV does not say that a policy exclusively applies to that department. Prompts use the CSV as the only evidence. A vector match is not proof that the policy answers the question.

## Genuine tests and comparison

Start with the four-question pilot in `test_questions.json`, then test all twelve questions. The expected answers in that file are for human evaluation only; they are never passed to either LLM or retrieval method.

The website allows manual accuracy and unsupported-claim review and computes the proportion of reviewed answers containing unsupported claims. An unanswered question can be inaccurate without containing a hallucination. Inspect factual claims, limits, approval requirements and citations against the full CSV. Invalid citation IDs are flagged, but valid citations alone do not prove correctness. No automatic factual accuracy score is claimed.

Download `policy_test_results.json` before closing the session: session results are not permanently stored by the website. Preserve error records and report them separately, rather than counting failed calls as successful zero-token answers. Repeat the same questions if you need a more reliable estimate; report sample size and repetition protocol. Keep model, prompt, database, top-k and generation settings fixed during the final comparison.

Timing includes retrieval and generation for each question, but excludes the initial policy index construction. Record the setup separately. LLM usage comes directly from API metadata (input, output, thinking and total); query embedding tokens are separate and shown as **Not reported** if absent. Total generation tokens can exceed input + output because of thinking tokens. Rules use zero LLM tokens. Lower LLM token use is not by itself proof of lower total cost: embeddings also consume quota/resources.

Optional command-line measurements:

```bash
python evaluate.py --rules-only --limit 12 --output results/rules.json
```

For AI tests set `GEMINI_API_KEY`, `GEMINI_MODEL` and `GEMINI_EMBEDDING_MODEL` in your local environment first (Streamlit Secrets are not automatically read by command-line programs), then run:

```bash
python evaluate.py --limit 4 --output results/pilot.json
python evaluate.py --limit 12 --output results/full.json
```

Method order is shuffled with seed 42, results are saved after every call, and the runner stops at the first error. Delay between successful AI answers defaults to six seconds but must be adapted to your actual quota. The initial embedding setup also requires quota. Identical code/input does not guarantee identical LLM answers even at temperature zero. Saved results document what actually happened.

Engine checks (no key or paid requests needed):

```bash
python -m unittest discover -s tests -v
```

Mocked API tests only verify implementation behavior; they do not demonstrate real AI accuracy or connectivity.

## Slack workspace and bot

1. Create a separate Slack workspace and test channel identifying the student, e.g. `Aeron Tekle Policy Assignment` and `aeron-policy-test`.
2. Create a Slack app at https://api.slack.com/apps. Use the configuration in `slack_manifest.json`, enable Socket Mode and generate an app token with `connections:write`.
3. Install the app into the workspace. Copy the bot token (`xoxb-…`) and app token (`xapp-…`) to local environment variables `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`. Do not put them in GitHub.
4. Invite the bot to the testing channel. Invite `raz@sdu.dk` to the workspace and grant access to that channel.
5. Run `python slack_bot.py`. Mention the bot in the channel, e.g. `@Policy Assistant How many vacation days do employees receive?`.
6. Expand the answer thread and take a genuine screenshot showing the channel, your question and the bot's answer with its policy source.

The bot defaults to `rules` so initial Slack testing does not require Gemini quota. After comparing the methods, set `SLACK_POLICY_METHOD` to `rules`, `full_context` or `rag` according to your chosen approach. AI modes also require the Gemini environment settings. The bot only listens for mentions and answers in a thread. Socket Mode allows local execution; leave the computer and script running while testing. Permanent hosting is not required.

## Final submission

* Deployed website URL (verify the lecturer can open it).
* Exactly two paragraphs comparing observed accuracy, speed, token use and unsupported answers, explaining the preferred method. Maximum two optional tables. Write after real tests.
* Source repository: https://github.com/AeronTekle/company-policy-assistant
* Actual Slack test screenshot.

The invitation and separate student-identifying workspace/channel are also required outcomes. This repository is code, not a completed four-item submission yet.

## Documentation used

* Gemini models: https://ai.google.dev/gemini-api/docs/models
* Gemini embeddings: https://ai.google.dev/api/embeddings
* Gemini generation: https://ai.google.dev/api/generate-content
* Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
* Slack Socket Mode: https://docs.slack.dev/tools/bolt-python/concepts/socket-mode/
