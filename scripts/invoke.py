"""Call the Function URL with SigV4 signing (the URL uses AWS_IAM auth).

    python scripts/invoke.py --profile ryan-demos --url https://<id>.lambda-url.eu-west-1.on.aws \
        --state "Help! My payouts have been failing for 3 days." \
        --noul "is_urgent=Does this convey urgency?" \
        --choice "team=Which team should handle this?=billing,technical,sales" \
        --score "frustration=How frustrated is the writer?=calm,frustrated,very angry"

Needs boto3 (pip install boto3). Use --raw to send a full Jev JSON body from stdin instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


def build_body(args: argparse.Namespace) -> dict:
    questions: dict = {}
    for spec in args.noul:
        qid, instructions = spec.split("=", 1)
        questions[qid] = {"type": "noul", "instructions": instructions}
    for spec in args.choice:
        qid, instructions, options = spec.split("=", 2)
        questions[qid] = {
            "type": "choice",
            "instructions": instructions,
            "criteria": {o.strip(): None for o in options.split(",")},
        }
    for spec in args.score:
        qid, instructions, levels = spec.split("=", 2)
        questions[qid] = {
            "type": "score",
            "instructions": instructions,
            "criteria": [level.strip() for level in levels.split(",")],
        }
    return {"state": args.state, "model": args.model, "questions": questions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--state", default="")
    parser.add_argument("--noul", action="append", default=[], metavar="ID=QUESTION")
    parser.add_argument("--choice", action="append", default=[], metavar="ID=QUESTION=A,B,C")
    parser.add_argument("--score", action="append", default=[], metavar="ID=QUESTION=LOW,MID,HIGH")
    parser.add_argument("--raw", action="store_true", help="read the JSON body from stdin")
    parser.add_argument("--health", action="store_true", help="GET /health instead")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    if args.health:
        method, url, data = "GET", f"{base}/health", None
    else:
        body = json.load(sys.stdin) if args.raw else build_body(args)
        method, url, data = "POST", f"{base}/v1/systemone", json.dumps(body).encode()

    credentials = boto3.Session(profile_name=args.profile, region_name=args.region).get_credentials()
    if credentials is None:
        sys.exit("no AWS credentials found")

    headers = {"content-type": "application/json", "host": urlparse(url).netloc}
    request = AWSRequest(method=method, url=url, data=data, headers=headers)
    SigV4Auth(credentials.get_frozen_credentials(), "lambda", args.region).add_auth(request)

    http_request = urllib.request.Request(url, data=data, method=method, headers=dict(request.headers))
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(http_request, timeout=args.timeout) as response:
            status, payload, response_headers = response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        status, payload, response_headers = error.code, error.read(), error.headers
    elapsed = time.perf_counter() - started

    print(f"HTTP {status} in {elapsed:.2f}s (server {response_headers.get('x-decider-latency-ms', '-')} ms)", file=sys.stderr)
    try:
        print(json.dumps(json.loads(payload), indent=2))
    except ValueError:
        print(payload.decode(errors="replace"))
    sys.exit(0 if status == 200 else 1)


if __name__ == "__main__":
    main()
