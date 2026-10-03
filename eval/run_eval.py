"""Score the pipeline on a set of questions.

Usage:
    uv run python eval/run_eval.py eval/questions.example.json [--out results.json]

Metrics per question:
    retrieval_hit  The retrieved excerpts contain at least one of the expected phrases (no LLM involved).
    faithful       A judge model confirms every claim in the answer is supported by the retrieved excerpts.
    correct        A judge model confirms the answer agrees with the reference answer.
                   For unanswerable questions, it confirms the answer says the page does not cover it.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.pipeline import Pipeline

JUDGE_PROMPT = """You are a strict evaluator of a question-answering system that must answer only from page excerpts.
Reply with a JSON object: {"faithful": true|false, "correct": true|false, "reason": "<one sentence>"}.

- faithful: every factual claim in the answer is supported by the excerpts. An answer that only says the page
  does not cover the question is faithful.
- correct: if the question is answerable, the answer agrees with the reference answer in substance.
  If the question is NOT answerable from the page, the answer must clearly say the page does not cover it
  and must not invent an answer."""


@dataclass
class Result:
    """The scores for one evaluated question."""

    url: str
    question: str
    answerable: bool
    retrieval_hit: bool | None
    faithful: bool
    correct: bool
    reason: str
    answer: str


def judge(client: Groq, model: str, case: dict[str, Any], answer: str, excerpts: str) -> dict[str, Any]:
    """Ask the judge model whether an answer is faithful to the excerpts and correct.

    Args:
        client: Groq client.
        model: Judge model name.
        case: The question case from the dataset.
        answer: The answer produced by the pipeline.
        excerpts: The retrieved text the answer should be based on.

    Returns:
        The judge's verdict with ``faithful``, ``correct`` and ``reason`` fields.
    """
    reference = case.get("reference") if case.get("answerable", True) else "NOT ANSWERABLE FROM THE PAGE"
    user_message = (
        f"Question: {case['question']}\n\nReference answer: {reference}\n\n"
        f"Retrieved excerpts:\n{excerpts or '(none)'}\n\nSystem answer: {answer}"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user_message}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    verdict: dict[str, Any] = json.loads(completion.choices[0].message.content or "{}")
    return verdict


def main() -> None:
    """Run the evaluation and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, help="Write per-question results to this JSON file")
    parser.add_argument("--no-ingest", action="store_true", help="Skip loading the pages (already stored)")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = Pipeline(settings)
    judge_client = Groq(api_key=settings.groq_api_key)
    results: list[Result] = []

    try:
        for page in json.loads(args.dataset.read_text(encoding="utf-8")):
            url = page["url"]
            if not args.no_ingest:
                ingested = pipeline.ingest(url)
                print(f"Loaded {ingested.url} ({ingested.chunk_count} chunks)")

            for case in page["questions"]:
                answerable = case.get("answerable", True)
                outcome = pipeline.ask(url, case["question"])
                excerpts = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(outcome.retrieved, start=1))

                phrases = [phrase.casefold() for phrase in case.get("must_contain", [])]
                retrieval_hit = (
                    any(phrase in excerpts.casefold() for phrase in phrases) if answerable and phrases else None
                )
                verdict = judge(judge_client, settings.groq_model, case, outcome.answer, excerpts)
                result = Result(
                    url=url,
                    question=case["question"],
                    answerable=answerable,
                    retrieval_hit=retrieval_hit,
                    faithful=bool(verdict.get("faithful")),
                    correct=bool(verdict.get("correct")),
                    reason=str(verdict.get("reason", "")),
                    answer=outcome.answer,
                )
                results.append(result)
                flags = f"hit={result.retrieval_hit} faithful={result.faithful} correct={result.correct}"
                print(f"- {case['question']}\n    {flags}")
    finally:
        pipeline.close()

    if not results:
        sys.exit("The dataset has no questions.")

    hits = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    print("\n=== Summary ===")
    print(f"Questions:          {len(results)}")
    if hits:
        print(f"Retrieval hit rate: {sum(hits) / len(hits):.0%} ({len(hits)} scored)")
    print(f"Faithfulness:       {sum(r.faithful for r in results) / len(results):.0%}")
    print(f"Correctness:        {sum(r.correct for r in results) / len(results):.0%}")

    if args.out:
        args.out.write_text(json.dumps([r.__dict__ for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Per-question results written to {args.out}")


if __name__ == "__main__":
    main()
