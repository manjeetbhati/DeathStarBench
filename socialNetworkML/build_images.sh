#!/bin/bash

DOCKER_REGISTRY="${DOCKER_REGISTRY:-""}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DOCKER_DIR="${SCRIPT_DIR}/docker"
SRC_DIR="${SCRIPT_DIR}/src"

# build media-frontend, openresty-thrift, thrift-microservice-deps images
docker build -f ${DOCKER_DIR}/media-frontend/xenial/Dockerfile -t ${DOCKER_REGISTRY}/media-frontend:xenial ${SCRIPT_DIR}
docker build -f ${DOCKER_DIR}/openresty-thrift/xenial/Dockerfile -t ${DOCKER_REGISTRY}/openresty-thrift-ml:xenial ${SCRIPT_DIR}
docker build -f ${DOCKER_DIR}/thrift-microservice-deps/cpp/Dockerfile -t ${DOCKER_REGISTRY}/thrift-microservice-deps:xenial ${DOCKER_DIR}/thrift-microservice-deps/cpp

# pulling & tagging jaegertracing/all-in-one, memcached, mongo, redis images
docker build -f ${DOCKER_DIR}/jaegertracing/Dockerfile -t ${DOCKER_REGISTRY}/jaegertracing-all-in-one:latest ${SCRIPT_DIR}
docker build -f ${DOCKER_DIR}/memcached/Dockerfile -t ${DOCKER_REGISTRY}/memcached:1.6.7 ${SCRIPT_DIR}
docker build -f ${DOCKER_DIR}/mongo/Dockerfile -t ${DOCKER_REGISTRY}/mongo:4.4.6 ${SCRIPT_DIR}
docker build -f ${DOCKER_DIR}/redis/Dockerfile -t ${DOCKER_REGISTRY}/redis:6.2.4 ${SCRIPT_DIR}

# build social-network-ml-microservices image
docker build -f ${SCRIPT_DIR}/Dockerfile -t ${DOCKER_REGISTRY}/social-network-ml-microservices:latest ${SCRIPT_DIR}
docker build -f ${SRC_DIR}/TextFilterService/Dockerfile -t ${DOCKER_REGISTRY}/social-network-text-filter-service:latest ${SCRIPT_DIR}
docker build -f ${SRC_DIR}/MediaFilterService/Dockerfile -t ${DOCKER_REGISTRY}/social-network-media-filter-service:latest ${SCRIPT_DIR}

# avx2
docker build -f ${SRC_DIR}/TextFilterService/Dockerfile.opt.avx2 -t ${DOCKER_REGISTRY}/social-network-text-filter-service-opt-avx2:latest ${SCRIPT_DIR}
docker build -f ${SRC_DIR}/MediaFilterService/Dockerfile.opt.avx2 -t ${DOCKER_REGISTRY}/social-network-media-filter-service-opt-avx2:latest ${SCRIPT_DIR}

# avx512
docker build -f ${SRC_DIR}/TextFilterService/Dockerfile.opt.avx512 -t ${DOCKER_REGISTRY}/social-network-text-filter-service-opt-avx512:latest ${SCRIPT_DIR}
docker build -f ${SRC_DIR}/MediaFilterService/Dockerfile.opt.avx512 -t ${DOCKER_REGISTRY}/social-network-media-filter-service-opt-avx512:latest ${SCRIPT_DIR}

# build social-network-client image
docker build -f ${SCRIPT_DIR}/client/Dockerfile -t ${DOCKER_REGISTRY}/social-network-ml-client:latest ${SCRIPT_DIR}

# build locust-ml image
docker build -f ${SCRIPT_DIR}/locust/Dockerfile -t ${DOCKER_REGISTRY}/locust-ml:latest ${SCRIPT_DIR}/locust
