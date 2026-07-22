#!/usr/bin/env bash

# Same as do_test_run.sh but SKIPS the Docker build step.
# Use this when the container (surgvu26-eval-cat2) is already built.

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="surgvu26-eval-cat2"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= Skipping build"

# Load the saved image if not already present in Docker
if ! docker image inspect "$DOCKER_IMAGE_TAG" &> /dev/null; then
    # Find the most recent saved .tar.gz
    SAVED_IMAGE=$(ls -t "${SCRIPT_DIR}"/${DOCKER_IMAGE_TAG}_*.tar.gz 2>/dev/null | head -1)
    if [ -z "$SAVED_IMAGE" ]; then
        echo "ERROR: Docker image '$DOCKER_IMAGE_TAG' not found and no saved .tar.gz in ${SCRIPT_DIR}"
        echo "Run ./do_build.sh first, or use ./do_test_run.sh instead."
        exit 1
    fi
    echo "=+= Loading saved image: $(basename "$SAVED_IMAGE") ..."
    docker load --input "$SAVED_IMAGE"
fi

echo "=+= Image ready: $DOCKER_IMAGE_TAG"

cleanup() {
    echo "=+= Cleaning permissions ..."
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"

    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/ground_truth"

if [ -d "$OUTPUT_DIR" ]; then
  chmod -f o+rwX "$OUTPUT_DIR"

  echo "=+= Cleaning up any earlier output"
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -m o+rwX "$OUTPUT_DIR"
fi

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

echo "=+= Doing a forward pass (no build)"
docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    --volume "${SCRIPT_DIR}/ground_truth":/opt/ml/input/data/ground_truth:ro \
    $DOCKER_IMAGE_TAG

echo "=+= Wrote results to ${OUTPUT_DIR}"

# Pretty-print the metrics if jq is available
if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo "=+= Metrics:"
    if command -v jq &> /dev/null; then
        jq . "${OUTPUT_DIR}/metrics.json"
    else
        cat "${OUTPUT_DIR}/metrics.json"
    fi
fi
