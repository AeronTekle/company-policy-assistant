# Evaluation protocol and evidence

The final comparison covers ten distinct questions and 30 real answers. Each question was evaluated once per method; no measured answer, time or usage count was invented or changed. The raw exports preserve the original reviews. The consolidated file adds an explicitly attributed qualitative review and evidence notes.

See [comparison](../comparison.md), [consolidated evidence](evaluation_2026-10-15.json), [original pilot](raw_pilot_2026-10-15.json) and [eight additional questions](raw_extra_tests_2026-10-15.json).

## Scoring

Correct means supported, relevant and complete. Partly correct includes irrelevant extra policies or missing handling of part of a question. Incorrect means the response does not answer the question or recognise unavailable information. Unsupported claims are factual assertions absent from the database; irrelevant verbatim policy text is assessed separately. This rubric explains why Rules can have zero unsupported claims but poor answer quality.

## Sample and timing

The eight new questions extend the original vacation and fictional-benefit tests. They cover multiple subjects, a paraphrase, a multi-policy question, a realistic missing benefit and a question with a missing numeric detail. No minimum test count is claimed to be an assignment requirement. Two browser sessions were used; generation settings, database fingerprint, models and top-k stayed the same. Method order was fixed (rules, full_context, rag); latency was not controlled. Means and medians include exactly ten answers per method. Two earlier duplicate pilot answers and the initial RAG error are archived but excluded.

## Limits

No unsupported factual claim was observed; this does not establish a general zero hallucination rate. Embedding usage was not reported and must not be treated as zero. The initial index preparation was 45.1158 seconds and is separate from question latency. The 91.1% reduction concerns generation LLM tokens only, not verified total cost. Tests and source inspection cannot establish correctness for every possible question.
