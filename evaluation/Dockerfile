FROM --platform=linux/amd64 docker.io/library/python:3.11-slim AS example-evaluation-amd64

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# Install CPU-only PyTorch first (much smaller than CUDA variant)
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --trusted-host download.pytorch.org \
    --trusted-host download-r2.pytorch.org \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Copy pre-downloaded models for offline use (no network at runtime)
# Run `python download_models_host.py` on the host BEFORE building.
COPY --chown=user:user model_cache/huggingface/ /home/user/.cache/huggingface/
COPY --chown=user:user model_cache/nltk_data/ /home/user/nltk_data/
ENV HF_HOME=/home/user/.cache/huggingface
ENV NLTK_DATA=/home/user/nltk_data
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

COPY --chown=user:user helpers.py /opt/app/
COPY --chown=user:user evaluate.py /opt/app/

# Setting this will limit the number of workers used by the evaluate.py
ENV GRAND_CHALLENGE_MAX_WORKERS=

ENTRYPOINT ["python", "evaluate.py"]
