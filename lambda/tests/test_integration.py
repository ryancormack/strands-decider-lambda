"""Runs the real model through the Lambda handler. Skipped unless DECIDER_MODEL_DIR points at a fetched checkpoint.

    python lambda/scripts/fetch_model.py .model
    DECIDER_MODEL_DIR=$PWD/.model/checkpoint python -m pytest lambda/tests/test_integration.py -s
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import pytest

MODEL_DIR = os.environ.get("DECIDER_MODEL_DIR", "")
pytestmark = pytest.mark.skipif(not Path(MODEL_DIR).is_dir(), reason="DECIDER_MODEL_DIR not set to a fetched checkpoint")

os.environ.setdefault("DECIDER_EAGER_LOAD", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import handler  # noqa: E402


def peak_rss_gib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**3) if sys.platform == "darwin" else peak / (1024**2)


def post(body: dict) -> tuple[dict, float]:
    event = {
        "rawPath": "/v1/systemone",
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps(body),
    }
    started = time.perf_counter()
    response = handler.handler(event)
    elapsed = time.perf_counter() - started
    assert response["statusCode"] == 200, response["body"]
    return json.loads(response["body"]), elapsed


URGENT = "Help! My payouts have been failing for 3 days! "
QUESTIONS = {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": None, "sales": None, "retail": None},
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the writer?",
        "criteria": ["calm", "frustrated", "depressed"],
    },
}


@pytest.fixture(scope="module")
def loaded() -> float:
    started = time.perf_counter()
    handler.get_engine()
    seconds = time.perf_counter() - started
    print(f"\nmodel load: {seconds:.1f}s, peak RSS {peak_rss_gib():.2f} GiB")
    return seconds


def test_health(loaded: float) -> None:
    response = handler.handler({"rawPath": "/health", "requestContext": {"http": {"method": "GET"}}})
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["status"] == "ok" and body["device"] == "cpu"
    assert body["model"] == "strands-decider-2B-hobson-v19"


def test_documented_example_matches_published_answers(loaded: float) -> None:
    body, elapsed = post({"state": URGENT, "model": "jev-latest", "questions": QUESTIONS})
    print(f"3 questions, {body['usage']['input_tokens']} tokens: {elapsed:.2f}s (first call)")
    assert set(body) == {"model", "answers", "usage"}
    answers = body["answers"]

    # Published v19 output for this input: noul 0.829, billing 0.769 confidence, score 1.10.
    assert 0.7 < answers["is_urgent"]["noul"] < 0.95
    assert answers["team"]["choice"] == "billing"
    assert set(answers["team"]["probabilities"]) == {"billing", "sales", "retail"}
    assert abs(sum(answers["team"]["probabilities"].values()) - 1) < 0.01
    assert 0.6 < answers["team"]["confidence"] < 0.95
    assert 0.8 < answers["frustration"]["score"] < 1.4
    assert answers["frustration"]["legend"] == {"0": "calm", "1": "frustrated", "2": "depressed"}
    assert set(answers["frustration"]["probabilities"]) == {"0", "1", "2"}


def test_single_question_latency_profile(loaded: float) -> None:
    timings = []
    for _ in range(5):
        _, elapsed = post({"state": URGENT, "model": "jev-latest", "questions": {"q": QUESTIONS["is_urgent"]}})
        timings.append(elapsed)
    print(f"1 question warm: min {min(timings):.2f}s median {sorted(timings)[2]:.2f}s max {max(timings):.2f}s")


def test_many_questions_share_one_state_pass(loaded: float) -> None:
    many = {f"q{i}": QUESTIONS["is_urgent"] for i in range(8)}
    _, elapsed = post({"state": URGENT * 20, "model": "jev-latest", "questions": many})
    print(f"8 questions over a long state: {elapsed:.2f}s, peak RSS {peak_rss_gib():.2f} GiB")


def test_structured_state_and_null_criteria(loaded: float) -> None:
    body, _ = post(
        {
            "state": {"ticket": {"subject": "Duplicate charge", "text": "I was charged twice. Refund me."}},
            "model": "jev-1.13",
            "questions": {
                "refund": {"type": "noul", "instructions": "Does `ticket.text` ask for money back?"},
                "dept": {
                    "type": "choice",
                    "instructions": "Which team should handle `ticket`?",
                    "criteria": {"billing": "Charges and refunds", "technical": None},
                },
            },
        }
    )
    assert body["answers"]["refund"]["noul"] > 0.5
    assert body["answers"]["dept"]["choice"] == "billing"
