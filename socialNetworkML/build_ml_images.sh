#!/bin/bash

DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DOCKER_DIR="${SCRIPT_DIR}/docker"
SRC_DIR="${SCRIPT_DIR}/src"

docker build -f ${DOCKER_DIR}/openresty-thrift/xenial/Dockerfile -t ${DOCKER_REGISTRY}/openresty-thrift-ml:xenial ${SCRIPT_DIR}
docker build -f ${SCRIPT_DIR}/Dockerfile -t ${DOCKER_REGISTRY}/social-network-ml-microservices:latest ${SCRIPT_DIR}

docker build -f ${SRC_DIR}/TextFilterService/Dockerfile -t ${DOCKER_REGISTRY}/social-network-text-filter-service:latest ${SCRIPT_DIR}
docker build -f ${SRC_DIR}/MediaFilterService/Dockerfile -t ${DOCKER_REGISTRY}/social-network-media-filter-service:latest ${SCRIPT_DIR}

# avx2
docker build -f ${SRC_DIR}/TextFilterService/Dockerfile.opt.avx2 -t ${DOCKER_REGISTRY}/social-network-text-filter-service-opt-avx2:latest ${SCRIPT_DIR}
docker build -f ${SRC_DIR}/MediaFilterService/Dockerfile.opt.avx2 -t ${DOCKER_REGISTRY}/social-network-media-filter-service-opt-avx2:latest ${SCRIPT_DIR}

# avx512
docker build -f ${SRC_DIR}/TextFilterService/Dockerfile.opt.avx512 -t ${DOCKER_REGISTRY}/social-network-text-filter-service-opt-avx512:latest ${SCRIPT_DIR}
docker build -f ${SRC_DIR}/MediaFilterService/Dockerfile.opt.avx512 -t ${DOCKER_REGISTRY}/social-network-media-filter-service-opt-avx512:latest ${SCRIPT_DIR}

docker build -f ${SCRIPT_DIR}/client/Dockerfile -t ${DOCKER_REGISTRY}/social-network-ml-client:latest ${SCRIPT_DIR}
docker build -f ${SCRIPT_DIR}/locust/Dockerfile -t ${DOCKER_REGISTRY}/locust-ml:latest ${SCRIPT_DIR}/locust

