FROM python:3.13-slim AS model
RUN pip install --no-cache-dir huggingface_hub==1.33.0
COPY scripts/fetch_model.py /fetch_model.py
RUN python /fetch_model.py /model && rm -rf /model/_hub

FROM model AS slim
RUN rm -rf /model/checkpoint/.cache /model/checkpoint/MANIFEST.sha256

FROM public.ecr.aws/lambda/python:3.13-arm64

COPY requirements.txt requirements-decider.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install --no-cache-dir --no-deps -r requirements-decider.txt

# One layer per shard (fetch_model.py EXPECTED_SHARDS) so a dropped push to ECR only repeats one.
COPY --from=slim /model/checkpoint /model/checkpoint
COPY --from=slim /model/base/model-00000-of-00008.safetensors /model/base/
COPY --from=slim /model/base/model-00001-of-00008.safetensors /model/base/
COPY --from=slim /model/base/model-00002-of-00008.safetensors /model/base/
COPY --from=slim /model/base/model-00003-of-00008.safetensors /model/base/
COPY --from=slim /model/base/model-00004-of-00008.safetensors /model/base/
COPY --from=slim /model/base/model-00005-of-00008.safetensors /model/base/
COPY --from=slim /model/base/model-00006-of-00008.safetensors /model/base/
COPY --from=slim /model/base/model-00007-of-00008.safetensors /model/base/
COPY --from=slim /model/base/*.json /model/base/*.txt /model/base/

ENV HF_HOME=/tmp/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    DECIDER_MODEL_DIR=/model/checkpoint

COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
