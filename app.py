import json
import os
import streamlit as st
from policy_engine import DEFAULT_MODEL, DEFAULT_EMBEDDING, EMBEDDING_ESTIMATE_METHOD, estimate_embedding_tokens, Gemini, PolicyError, load_policies, build_index, answer

st.set_page_config(page_title="Company Policy Assistant", page_icon="📖", layout="wide")


def setting(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except st.errors.StreamlitSecretNotFoundError:
        return os.getenv(name, default)


@st.cache_data
def policies():
    return load_policies()


@st.cache_resource
def get_index(_client, embedding_model, fingerprint):
    return build_index(_client, policies())


def token(value):
    return "Not reported" if value is None else str(value)


records = policies()
key = setting("GEMINI_API_KEY")
model = setting("GEMINI_MODEL", DEFAULT_MODEL)
embedding_model = setting("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING)
client = Gemini(key, model, embedding_model) if key else None
st.title("Company Policy Assistant")
st.caption(f"{len(records)} policies · Compare answers, evidence, response time and token use")
st.session_state.setdefault("history", [])

with st.sidebar:
    st.header("Methods")
    st.write("**1 · Rules**\n\nFixed keyword rules return policy text.")
    st.write("**2 · Full context**\n\nGemini receives every policy, without vector retrieval.")
    st.write("**3 · Vector retrieval (RAG)**\n\nEmbeddings retrieve five policies for Gemini.")
    st.caption(f"Answer model: {model}\n\nEmbedding model: {embedding_model}")
    if not key:
        st.warning("Add GEMINI_API_KEY in the app's Secrets settings to enable AI.")
    index = st.session_state.get("index")
    if st.button("Prepare vector index", disabled=not key):
        try:
            with st.spinner("Encoding policies. Completed embeddings are saved for reuse…"):
                import hashlib
                fingerprint = hashlib.sha256(json.dumps(records).encode()).hexdigest()
                st.session_state.index, st.session_state.index_setup = get_index(client, embedding_model, fingerprint)
            index = st.session_state.index
        except PolicyError as exc:
            st.error(str(exc))
    if index is not None and st.session_state.get("index_setup", {}).get("model") != embedding_model:
        index = None
    if index is not None:
        st.success("Vector index ready")
        with st.expander("Index setup measurements"):
            st.json(st.session_state.index_setup)
            st.caption("Index preparation is separate from per-question time. Missing token counts are not treated as zero.")
            st.caption(f"Estimated policy embedding tokens: {sum(estimate_embedding_tokens(json.dumps(r)) for r in records):,}. {EMBEDDING_ESTIMATE_METHOD}. This describes the entire index, not tokens spent loading cached vectors.")

with st.form("question"):
    question = st.text_input("Ask a company policy question", value="How many vacation days do employees receive?", max_chars=2000)
    submitted = st.form_submit_button("Compare the three methods", type="primary")
if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        results = []
        # Sequential calls are friendlier to a free-tier quota. No result cache.
        for method in ("rules", "full_context", "rag"):
            try:
                with st.spinner(f"Running {method}…"):
                    result = answer(question, method, records, client, index)
                results.append(result)
            except PolicyError as exc:
                results.append({"method": method, "error": str(exc), "question": question})
        st.session_state.current = results
        st.session_state.history.extend(results)

results = st.session_state.get("current", [])
if results:
    labels = ("1 · Rules-based search", "2 · LLM without vector index", "3 · LLM with vector index")
    for column, label, result in zip(st.columns(3), labels, results):
        with column:
            st.subheader(label)
            if "error" in result:
                st.error(result["error"])
                continue
            st.write(result["answer"])
            st.caption(f"Response time: {result['seconds']:.4f} seconds")
            st.write(f"**LLM tokens:** {token(result['llm_total_tokens'])}")
            st.caption(f"Input {token(result['input_tokens'])} · Output {token(result['output_tokens'])} · Thinking {token(result['thinking_tokens'])}")
            if result["method"] == "rag":
                if result["query_embedding_tokens"] is not None:
                    st.caption(f"Query embedding tokens: {result['query_embedding_tokens']} (API-reported)")
                else:
                    estimate = result.get("query_embedding_tokens_estimate", estimate_embedding_tokens(result["question"]))
                    st.caption(f"Query embedding tokens: approximately {estimate} (estimated)")
                    st.caption(f"API count not reported. {EMBEDDING_ESTIMATE_METHOD}.")
            st.write("**Relevant policies**")
            if not result["policies"]:
                st.caption("No valid policy cited.")
            for r in result["policies"]:
                with st.expander(f"{r['id']} · {r['title']}"):
                    st.write(r["policy_text"])
                    st.caption(f"Department: {r['department']} · Category: {r['category']}")
            if result["invalid_policy_ids"]:
                st.error("Invalid citations: " + ", ".join(result["invalid_policy_ids"]))
            if result["retrieved"]:
                with st.expander("Retrieved evidence and similarity"):
                    for hit in result["retrieved"]:
                        st.write(f"{hit['policy']['id']} · {hit['policy']['title']} · {hit['score']}")
            st.caption("A citation does not prove correctness. Review the answer against the policy database.")

st.subheader("Review test results")
st.caption("Check factual correctness and whether any claim lacks policy support. Unsupported answers include invented amounts, conditions or citations. Missing answers are accuracy errors, but are not automatically hallucinations.")
valid = [r for r in st.session_state.history if "error" not in r]
if valid:
    selected = st.selectbox("Result to review", range(len(valid)), format_func=lambda n: f"{n+1} · {valid[n]['method']} · {valid[n]['question']}")
    result = valid[selected]
    st.write(result["answer"])
    with st.form(f"review-{selected}"):
        accuracy = st.selectbox("Answer accuracy", ["unreviewed", "correct", "partly correct", "incorrect"], index=["unreviewed", "correct", "partly correct", "incorrect"].index(result["review"]["accuracy"]))
        unsupported = st.selectbox("Contains unsupported claims?", ["unreviewed", "no", "yes"], index=["unreviewed", "no", "yes"].index(result["review"]["unsupported"]))
        notes = st.text_area("Evidence / review notes", value=result["review"]["notes"])
        if st.form_submit_button("Save review"):
            result["review"] = {"accuracy": accuracy, "unsupported": unsupported, "notes": notes}
            st.rerun()
    summaries = []
    for method in ("rules", "full_context", "rag"):
        runs = [r for r in valid if r["method"] == method]
        reviewed = [r for r in runs if r["review"]["unsupported"] != "unreviewed"]
        unsupported_count = sum(r["review"]["unsupported"] == "yes" for r in reviewed)
        summaries.append({"Method": method, "Successful answers": len(runs), "Reviewed for unsupported claims": len(reviewed),
                          "Unsupported answer rate": f"{unsupported_count}/{len(reviewed)} ({100*unsupported_count/len(reviewed):.1f}%)" if reviewed else "Not evaluated",
                          "Mean seconds": round(sum(r["seconds"] for r in runs)/len(runs), 4) if runs else None})
    st.dataframe(summaries, hide_index=True, use_container_width=True)
    st.download_button("Download test results and reviews", json.dumps({"results": st.session_state.history, "index_setup": st.session_state.get("index_setup")}, indent=2), "policy_test_results.json", "application/json")
    st.caption("Results and reviews are kept only in this session. Download them before closing or restarting the app. Errors are retained in the download and excluded from averages.")
else:
    st.info("No measured results yet. Ask a question to begin.")

# Persisted evidence is separate from the current browser session's measurements.
from pathlib import Path
recorded_root = Path(__file__).resolve().parent
recorded_comparison = recorded_root / "comparison.md"
recorded_summary = recorded_root / "results" / "summary_2026-10-15.json"
recorded_evidence = recorded_root / "results" / "evaluation_2026-10-15.json"
if recorded_comparison.exists() and recorded_summary.exists() and recorded_evidence.exists():
    st.subheader("Recorded evaluation · 10 questions / 30 answers")
    st.caption("Measured on 15 October 2026 across two sessions. This saved evaluation is independent of the live-session history above. Reviews assess accuracy, relevance and support against the CSV.")
    st.markdown(recorded_comparison.read_text(encoding="utf-8"))
    st.dataframe(json.loads(recorded_summary.read_text(encoding="utf-8")), hide_index=True, use_container_width=True)
    st.caption("LLM token counts exclude embeddings. Index preparation: 45.1158 seconds. Embedding token usage was not reported. These selected tests do not establish a general hallucination rate.")
    saved_results = json.loads(recorded_evidence.read_text(encoding="utf-8"))["results"]
    saved_rag = [r for r in saved_results if r.get("method") == "rag" and "error" not in r]
    if saved_rag:
        estimated_query_tokens = sum(estimate_embedding_tokens(r["question"]) for r in saved_rag)
        estimated_index_tokens = sum(estimate_embedding_tokens(json.dumps(r)) for r in records)
        st.info(f"Embedding token estimates: approximately {estimated_query_tokens:,} across the {len(saved_rag)} recorded RAG queries (mean {estimated_query_tokens / len(saved_rag):.1f}); approximately {estimated_index_tokens:,} to embed the current {len(records)} policies once. These are rough estimates, not API-reported usage.")
        st.caption(f"Method: {EMBEDDING_ESTIMATE_METHOD}. Calculated from saved questions and serialized policy text; excludes task framing and any provider truncation. Original measurements and generation-token totals are preserved.")
    st.download_button("Download recorded evaluation and evidence notes", recorded_evidence.read_text(encoding="utf-8"), "policy_evaluation_2026-10-15.json", "application/json")

with st.expander("Policy database — source of truth"):
    st.dataframe(records, hide_index=True, use_container_width=True)

