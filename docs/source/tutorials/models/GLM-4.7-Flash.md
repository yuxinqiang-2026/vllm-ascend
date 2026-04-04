# GLM-4.7-Flash

## Introduction

`GLM-4.7-Flash` is a 4.7B efficient model optimized for low-latency inference and real-time use cases.

The `GLM-4.7-Flash` model is verified in the vLLM Ascend E2E workflow through model config based evaluation.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- HuggingFace model: [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
- Local model path example: `/root/jupyter/workspace/GLM-4.7-Flash`

### Installation

You can use the official docker image of vLLM Ascend.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -v /root/jupyter/workspace:/root/jupyter/workspace \
    -it $IMAGE bash
```

## Deployment

### Single-node Deployment

Run online inference with local model weights:

```bash
export MODEL_PATH=/root/jupyter/workspace/GLM-4.7-Flash

vllm serve ${MODEL_PATH} \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name glm-4.7-flash \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8
```

## Functional Verification

After the service starts, run:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7-flash",
    "prompt": "Introduce Ascend NPU in one sentence.",
    "max_completion_tokens": 64,
    "temperature": 0
  }'
```

## Accuracy Evaluation (E2E)

Use the model config in `tests/e2e/models/configs/GLM-4.7-Flash.yaml`:

```bash
cd /workspace/vllm-ascend
pytest tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/GLM-4.7-Flash.yaml \
  --tp-size 1
```

If you run batch evaluation with list file, ensure `GLM-4.7-Flash.yaml` is included in `tests/e2e/models/configs/accuracy.txt`.
