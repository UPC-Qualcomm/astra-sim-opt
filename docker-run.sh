#!/bin/bash

# Helper script to run ASTRA-SIM Docker container with proper volume mounting

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMAGE_NAME="mona_astra:latest"
CONTAINER_NAME="mona_astra"

# Build the Docker image if it doesn't exist
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo "Building Docker image ${IMAGE_NAME}..."
    docker build -t ${IMAGE_NAME} -f ${SCRIPT_DIR}/Dockerfile ${SCRIPT_DIR}
fi

# Run the container with ASTRA-SIM mounted
echo "Starting container ${CONTAINER_NAME} with ASTRA-SIM mounted from ${SCRIPT_DIR}"
docker run -it \
    --name ${CONTAINER_NAME} \
    -v ${SCRIPT_DIR}:/app/astra-sim \
    -w /app/astra-sim \
    ${IMAGE_NAME} \
    "$@"