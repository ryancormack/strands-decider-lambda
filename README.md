# strands-decider-lambda

[Strands Decider](https://github.com/strands-labs/strands-decider) (`strands-decider-2B-hobson-v21`) behind an IAM-authenticated Lambda Function URL, speaking the Jev System One API.

This is written so the model can run as a remote service on CPU only, with no GPU and nothing to keep running. It is slow. A warm request takes 2 to 6 seconds, and a cold start takes about 30 seconds. It is also cheap: around $0.0002 to $0.0007 per request, and $0 while idle. The trade is latency for cost, which makes sense when a small model can take over classification and routing work that would otherwise go to a larger, more expensive model.

## How it works

- `lib/decider-stack.ts`: CDK (TypeScript). One `DockerImageFunction` (arm64, 10,240 MB, 6 vCPUs, 180 s timeout) and an `AWS_IAM` Function URL.
- `lambda/handler.py`: `POST /v1/systemone` takes a Jev request and returns a Jev response. `GET /health` reports the loaded model.
- `lambda/Dockerfile`: downloads the pinned checkpoint and base model at build time and bakes them into the image. The function makes no network calls at runtime.
- The model runs in fp32 on CPU. The Decider's own engine upcasts the bf16 weights because CPU bf16 kernels are slower. That takes about 7 GiB of memory, which is why the function needs the top memory setting. 10,240 MB is also the setting that gives 6 vCPUs.

## Deploy

Needs Node 22+, pnpm, Docker and an account that has been through `cdk bootstrap`.

```bash
pnpm install
pnpm test
pnpm exec cdk deploy --profile <profile>
```

The first deploy builds a ~5 GB image and pushes ~3.4 GB compressed to ECR, so it takes a while on a slow uplink. Later deploys reuse the layers, and only the layers that changed are pushed. The stack prints `FunctionUrl`.

## Call it

The URL uses IAM auth, so requests must be SigV4-signed for service `lambda`. `scripts/invoke.py` does this (needs `boto3`):

```bash
python scripts/invoke.py --profile <profile> --url <FunctionUrl> \
  --state "Help! My payouts have been failing for 3 days." \
  --noul "is_urgent=Does this convey urgency?" \
  --choice "team=Which team should handle this?=billing,technical,sales"
```

Request body (`--raw` sends your own JSON from stdin):

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": null, "technical": null, "sales": null}
    }
  }
}
```

Response (real output):

```json
{
  "model": "strands-decider-2B-hobson-v21",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.848},
    "team": {
      "type": "choice",
      "choice": "technical",
      "probabilities": {"billing": 0.2722, "technical": 0.7041, "sales": 0.0236},
      "confidence": 0.5562
    }
  },
  "usage": {"input_tokens": 133, "output_tokens": 2}
}
```

`score` questions take a list of levels as `criteria` and return a `score` plus per-level `probabilities`. The response carries an `x-decider-latency-ms` header. Validation failures return `422` with FastAPI-style `detail`. `model` is required by the Jev API, but any value is accepted and the function always serves the one baked-in model.

## Performance

Measured on the deployed function in eu-west-1. "Server" is the time the handler reported, not round trip.

| Request | Graviton3 sandbox | Graviton2 sandbox |
|---|---|---|
| 1 question, 86 input tokens | 1.3 s | 2.2 to 2.5 s |
| 3 questions, 196 input tokens | 3.3 s | 6.0 to 6.1 s |

Median warm server time over two bursts of five concurrent sandboxes. Lambda arm64 places a sandbox on either CPU generation: `/proc/cpuinfo` showed Neoverse N1 (Graviton2, no bf16 or i8mm) and Neoverse V1 (Graviton3) in a 12-sandbox probe, and you can't choose. Each response carries `x-decider-cpu-part` (`0xd0c` is Graviton2, `0xd40` is Graviton3) so a latency can be attributed. Expect roughly 1.8x between the two.

The LoRA adapter is merged into the fp32 weights at load (`DECIDER_MERGE_LORA=0` turns that off). Answers are identical to four decimals, and on the same image it was about 9% faster for 3 questions and 6% for 1 on Graviton3, and within 3% on Graviton2 (one to four sandboxes per figure, so treat it as indicative).

Time scales with questions and input length, since each is a forward pass of a 2B model on 6 vCPUs. The model has no GPU path here, so this is the floor for this configuration, and more memory doesn't help because 10,240 MB is Lambda's maximum.

### CPU scaling

Inference is parallel across cores, not bound by one. Same 3-question request on the live function, varying `DECIDER_NUM_THREADS` and `OMP_NUM_THREADS` together (server time, three warm runs each):

| Threads | Server time | Speed-up |
|---|---|---|
| 1 | 27.5 s | 1.0x |
| 3 | 10.7 s | 2.6x |
| 6 | 4.4 s | 6.3x |

At 10,240 MB Lambda allocates about 5.8 vCPUs, so 6 threads is the ceiling. The function can't go faster without a platform with more cores. Since billing is GB-seconds at a fixed memory size, the extra threads also cut cost per request: 1 thread would cost about 6x more per request at the same memory, and the fp32 weights mean memory can't drop much below 8 GiB anyway.

### Cold starts

A cold start loads about 4.5 GB of model weights, upcasts them to fp32 and imports torch and transformers. Measured:

- **About 18 to 30 s** when the image's chunks are already cached in the region.
- **Over 3 minutes** for the first request after a new image is pushed. Lambda fetches container images lazily in chunks, and nothing is cached yet. Expect to warm up after each deploy, and don't judge the function by that first request.

What caused this and what fixed it:

1. Lambda's init phase is capped at 10 s. Loading the model overruns it, so Lambda retries init inside the first invoke and the load counts against the function timeout.
2. `torch` upcasting mmap'd safetensors faulted pages in one chunk at a time from the lazy image. At roughly 13 MB/s that took minutes. `prefetch_model()` in `handler.py` now reads every model file in 32 parallel threads first (3.6 GiB in about 19 s once cached), and the upcast then runs against warm pages.
3. The image carried unused files (`.cache` download metadata, `MANIFEST.sha256`). These are removed in a `slim` build stage so the cached download layer is reused.

## Cost

Lambda arm64 in eu-west-1 is $0.0000133334 per GB-second plus $0.20 per million requests (AWS Price List API, effective 2026-09-01, first 7.5 billion GB-s tier). At 10 GB the compute rate is $0.000133 per second.

| Request | Billed | Per request | Per 1,000 |
|---|---|---|---|
| 1 question | 1.7 s | $0.00023 | $0.23 |
| 3 questions | 3.5 s | $0.00047 | $0.47 |
| 3 questions, longer state | 5.5 s | $0.00073 | $0.73 |
| Cold start (~30 s) | 30 s | $0.0040 | n/a |

Monthly compute at the 3-question rate ($0.47 per 1,000 requests), excluding cold starts:

| Requests per month | Cost |
|---|---|
| 10,000 | $4.70 |
| 100,000 | $47 |
| 1,000,000 | $467 |

Idle cost is zero. Other charges are small: the Function URL is free, and ECR storage is about $0.34 per retained 3.4 GB image at $0.10 per GB-month (price not re-checked). CDK keeps old image versions in ECR until you delete them. CloudWatch logs are kept for one month.

`reservedConcurrentExecutions` is 5. That caps concurrent sandboxes (and so spend and the ~10 GB of memory each holds), not cost per request. Raise it with the `reservedConcurrentExecutions` prop in `bin/app.ts` if you need more throughput.

## Limits and notes

- Throughput per sandbox is one request at a time. Concurrent requests each need their own sandbox and their own cold start.
- `AWS_IAM` auth means callers need `lambda:InvokeFunctionUrl` on the function.
- The model is pinned by revision in `lambda/scripts/fetch_model.py` and the `strands-decider` source by commit and sha256 in `lambda/requirements-decider.txt`. The build verifies the checkpoint against its own `MANIFEST.sha256` and the base weights against the published sha256.
- The base model is resharded into eight layer-sized files, one Docker layer each, so a dropped push to ECR repeats one shard instead of the whole model.

## Teardown

```bash
pnpm exec cdk destroy --profile <profile>
```

The ECR repository is CDK's shared asset repository, so old images stay there until deleted.
