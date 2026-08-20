"""
llm_compression.py — LLM-generated compressed representations for retrieval
==============================================================================
Core idea: instead of embedding a raw chunk verbatim, have an LLM compress it
into a dense, retrieval-oriented representation first, then embed that. In
the source system (SecondBrain) this representation is called an "AKA"
(compressed knowledge encoding); this module reproduces the same prompt
under a generic name.

Two variants are exposed to make the ablation in evaluation/ meaningful:

- compress_for_retrieval(): the AKA-style prompt — dense shorthand notation
  (arrows for causation, pipes for alternatives, domain abbreviations),
  explicitly optimized to preserve entities/relationships/actions for search,
  not for human readability.
- summarize_plain(): a conventional summarization prompt. Included only to
  isolate one variable in the evaluation: does the retrieval improvement
  come from the text being SHORTER, or from the compression being
  RETRIEVAL-ORIENTED specifically? A generic summary controls for length
  while removing the retrieval-oriented structure.

Both take a `generate` callable — (prompt: str) -> str — so this module has
no hard dependency on any particular LLM backend. Point it at a local model
(Ollama, MLX, llama.cpp) or a hosted API.
"""

from typing import Callable

GenerateFn = Callable[[str], str]


def _compress_prompt(text: str) -> str:
    return f"""Compress this document into a single dense line using shorthand notation.

Rules:
- Use arrows (->) for causation/flow, pipes (|) for alternatives, parens for context
- Abbreviate common words: proj=project, cfg=config, impl=implementation, req=requirement
- Use domain abbreviations freely: ML, API, DB, UI, auth, infra
- Keep entity names intact (people, places, tools, technical terms)
- Target: 20-50 tokens, one line, no JSON
- Must preserve: key entities, relationships, actions, outcomes

Input:
{text}

Compressed (one line):"""


def _summary_prompt(text: str) -> str:
    return f"""Summarize this document in one or two plain sentences.

Input:
{text}

Summary:"""


def compress_for_retrieval(text: str, generate: GenerateFn, max_chars: int = 4000) -> str:
    """Retrieval-oriented compression (the "AKA" technique)."""
    text = text[:max_chars]
    response = generate(_compress_prompt(text))
    line = response.strip().split("\n")[0].strip().strip('"').strip("'")
    return line


def summarize_plain(text: str, generate: GenerateFn, max_chars: int = 4000) -> str:
    """Conventional summary — ablation control, isolates length from technique.

    Unlike compress_for_retrieval() (which explicitly asks for one line),
    the summary prompt asks for "one or two plain sentences" with no
    single-line constraint, so a second sentence may land on its own line.
    Join all lines rather than truncating to the first, or a two-sentence
    summary silently loses its second sentence."""
    text = text[:max_chars]
    response = generate(_summary_prompt(text))
    joined = " ".join(line.strip() for line in response.strip().splitlines() if line.strip())
    return joined.strip().strip('"').strip("'")
