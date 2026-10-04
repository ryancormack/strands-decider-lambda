from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ["DECIDER_EAGER_LOAD"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import handler  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)


class FakeEngine:
    def __init__(self) -> None:
        self.requests: list[SystemOneRequest] = []

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        self.requests.append(request)
        answers: dict[str, Any] = {}
        for name, question in request.questions.items():
            if question.type == "noul":
                answers[name] = NoulAnswer(noul=0.83)
            elif question.type == "choice":
                options = list(question.criteria)
                answers[name] = ChoiceAnswer(
                    choice=options[0],
                    probabilities={o: (0.8 if i == 0 else 0.2 / (len(options) - 1)) for i, o in enumerate(options)},
                    confidence=0.7,
                )
            else:
                levels = {str(i): str(c) for i, c in enumerate(question.criteria)}
                answers[name] = ScoreAnswer(
                    score=1.1,
                    legend=levels,
                    probabilities={k: 1 / len(levels) for k in levels},
                    confidence=0.5,
                )
        return SystemOneResponse(model="fake", answers=answers, usage=Usage(input_tokens=10, output_tokens=len(answers)))


@pytest.fixture()
def engine(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    fake = FakeEngine()
    monkeypatch.setattr(handler, "_engine", fake)
    return fake


def event(method: str, path: str, body: Any = None, *, base64_body: bool = False) -> dict[str, Any]:
    raw = body if isinstance(body, str) else json.dumps(body)
    if base64_body:
        raw = base64.b64encode(raw.encode()).decode()
    return {
        "rawPath": path,
        "requestContext": {"http": {"method": method, "path": path}},
        "body": raw if body is not None else None,
        "isBase64Encoded": base64_body,
    }


FULL_REQUEST = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments", "technical": "Bugs", "sales": "Pricing"},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


def test_systemone_returns_jev_shaped_response(engine: FakeEngine) -> None:
    res = handler.handler(event("POST", "/v1/systemone", FULL_REQUEST))
    assert res["statusCode"] == 200
    assert res["headers"]["content-type"] == "application/json"
    assert "x-decider-latency-ms" in res["headers"]
    body = json.loads(res["body"])
    assert set(body) == {"model", "answers", "usage"}
    assert body["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert body["answers"]["is_urgent"] == {"type": "noul", "noul": 0.83}
    assert set(body["answers"]["department"]) == {"type", "choice", "probabilities", "confidence"}
    assert set(body["answers"]["frustration"]) == {"type", "score", "legend", "probabilities", "confidence"}
    assert body["answers"]["frustration"]["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert list(engine.requests[0].questions) == ["is_urgent", "department", "frustration"]


def test_base64_body_and_trailing_slash(engine: FakeEngine) -> None:
    res = handler.handler(event("POST", "/v1/systemone/", FULL_REQUEST, base64_body=True))
    assert res["statusCode"] == 200


def test_structured_state_criteria_and_null_descriptions(engine: FakeEngine) -> None:
    request = {
        "state": {"ticket": {"subject": "Duplicate charge"}},
        "model": "jev-1.13",
        "questions": {
            "tone": {
                "type": "choice",
                "instructions": {"question": "Tone of `ticket.subject`?", "note": "x"},
                "criteria": {"calm": None, "angry": {"what": "shouting"}},
            }
        },
    }
    assert handler.handler(event("POST", "/v1/systemone", request))["statusCode"] == 200


def test_empty_state_is_allowed(engine: FakeEngine) -> None:
    request = {**FULL_REQUEST, "state": ""}
    assert handler.handler(event("POST", "/v1/systemone", request))["statusCode"] == 200


@pytest.mark.parametrize("payload", [{**FULL_REQUEST, "model": None}, {k: v for k, v in FULL_REQUEST.items() if k != "model"}])
def test_model_is_required(engine: FakeEngine, payload: dict[str, Any]) -> None:
    res = handler.handler(event("POST", "/v1/systemone", payload))
    assert res["statusCode"] == 422
    assert json.loads(res["body"])["detail"][0]["loc"] == ["body", "model"]


def test_missing_questions_is_422(engine: FakeEngine) -> None:
    res = handler.handler(event("POST", "/v1/systemone", {"state": "x", "model": "jev-latest"}))
    assert res["statusCode"] == 422
    assert json.loads(res["body"])["detail"][0]["loc"] == ["questions"]


def test_score_level_limits(engine: FakeEngine) -> None:
    question = {"type": "score", "instructions": "q", "criteria": [str(i) for i in range(11)]}
    res = handler.handler(event("POST", "/v1/systemone", {"state": "x", "model": "m", "questions": {"q": question}}))
    assert res["statusCode"] == 422


def test_choice_needs_two_options(engine: FakeEngine) -> None:
    question = {"type": "choice", "instructions": "q", "criteria": {"only": "one"}}
    res = handler.handler(event("POST", "/v1/systemone", {"state": "x", "model": "m", "questions": {"q": question}}))
    assert res["statusCode"] == 422


def test_invalid_json_and_non_object_are_422(engine: FakeEngine) -> None:
    assert handler.handler(event("POST", "/v1/systemone", "{nope"))["statusCode"] == 422
    assert handler.handler(event("POST", "/v1/systemone", "[1]"))["statusCode"] == 422
    bad_b64 = {**event("POST", "/v1/systemone", "x"), "isBase64Encoded": True, "body": "!!!"}
    assert handler.handler(bad_b64)["statusCode"] == 422


def test_engine_value_error_is_422(engine: FakeEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_: SystemOneRequest) -> SystemOneResponse:
        raise ValueError("prompt is too long")

    monkeypatch.setattr(engine, "evaluate", boom)
    res = handler.handler(event("POST", "/v1/systemone", FULL_REQUEST))
    assert res["statusCode"] == 422
    assert json.loads(res["body"]) == {"detail": "prompt is too long"}


def test_routing(engine: FakeEngine) -> None:
    assert handler.handler(event("GET", "/v1/systemone"))["statusCode"] == 405
    assert handler.handler(event("POST", "/health", {}))["statusCode"] == 405
    assert handler.handler(event("POST", "/nope", {}))["statusCode"] == 404
    assert handler.handler({})["statusCode"] == 404
