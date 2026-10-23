# syntax=docker/dockerfile:1.7

# Runs the SECOM analysis on Linux. The default command fetches the data if it
# is missing and runs the full pipeline with PNG figures:
#
#   docker build -t secom-process-control .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" \
#       secom-process-control
#
# Any `secom` subcommand can follow the image name, e.g. `stages`, or
# `monitor --host 0.0.0.0` with `-p 8050:8050`.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Dependencies first, pinned to the versions that produced the committed
# outputs, so this layer is cached across source edits.
COPY docker/constraints.txt ./constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --constraint constraints.txt \
        numpy pandas scipy scikit-learn matplotlib statsmodels pyyaml

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --constraint constraints.txt --no-deps . \
 && pip check

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="secom-process-control" \
      org.opencontainers.image.description="Statistical process control analysis of the UCI SECOM fab dataset" \
      org.opencontainers.image.source="https://github.com/syedwajiulhassan715-rgb/semiconductor-process-control"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

RUN groupadd --system --gid 1000 secom \
 && useradd --system --uid 1000 --gid secom --home-dir /app --shell /sbin/nologin secom

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# The analysis config and the committed outputs. The outputs let `monitor`
# start without a pipeline run; mounting data/ and reports/ replaces them.
COPY --chown=secom:secom config ./config
COPY --chown=secom:secom data/powerbi ./data/powerbi
COPY --chown=secom:secom data/processed ./data/processed
COPY --chown=secom:secom reports ./reports
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/secom-entrypoint
RUN mkdir -p data/raw && chown secom:secom data/raw

USER secom
EXPOSE 8050

ENTRYPOINT ["secom-entrypoint"]
