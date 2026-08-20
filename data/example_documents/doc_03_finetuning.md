# Fine-Tuning a Retrieval Reranker

A general-purpose cross-encoder trained on public benchmarks such as
MS MARCO is a reasonable starting point, but it doesn't know anything about
a specific domain's vocabulary, document structure, or notion of relevance.
Fine-tuning on domain-specific query-document pairs closes that gap.

Training data for this is typically mined automatically: for each query in
a small labeled set (a "golden" set of queries with known correct
documents), run the existing retrieval pipeline to get candidates. Any
candidate matching the known correct document becomes a positive example;
any candidate that ranks highly but is not the correct document becomes a
"hard negative" — a genuinely useful training signal because it teaches the
model to distinguish superficially similar but wrong documents, which is a
harder and more valuable lesson than distinguishing obviously unrelated
text.

Evaluation should compare the base model and fine-tuned model on the same
held-out queries, using retrieval metrics such as Recall@1 (was the correct
document ranked first?) and Recall@5, not just a classification accuracy
score on the training objective, since the actual downstream task is
ranking, not binary classification.
