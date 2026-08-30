# Running on RunPod

This project is a long-running GPU training job, so deploy it as a **Pod**, not
as a request-oriented Serverless worker.

## Recommended resources

- Use a 48 GB GPU such as an A40, RTX A6000, or L40S for the default model set.
- Attach a persistent volume at `/workspace` so model caches and checkpoints
  survive Pod replacement.
- Allocate at least 50 GB of storage; the Hugging Face cache, optimizer state,
  datasets, and checkpoints can grow beyond the raw model sizes.

## Build and deploy

```bash
docker build -t YOUR_DOCKERHUB_USER/nlp-project:latest .
docker push YOUR_DOCKERHUB_USER/nlp-project:latest
```

Create a RunPod Pod template using that image. The image starts `main.py`
automatically. Add these values through RunPod Secrets/environment variables:

- `OPENAI_API_KEY`: required for the external policy evaluator.
- `HF_TOKEN`: needed only for gated/private Hugging Face resources.
- `CHECKPOINT_DIR=/workspace/checkpoints`
- `HF_HOME=/workspace/cache/huggingface`

Do not copy `.env` into the image. It is excluded by `.dockerignore`.

## Before a paid training run

Run the fast local tests inside the container:

```bash
python -m unittest discover -s tests -v
python -c "from trl.experimental.ppo import PPOConfig, PPOTrainer; print('TRL PPO import OK')"
```

Then start with small `dataset_limit` and `static_dataset_length` values in
`config.py`. Baseline and post-PPO evaluation each make one OpenAI request per
static prompt, so a static set of 100 rows makes roughly 200 evaluator calls per
run.
