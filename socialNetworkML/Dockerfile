FROM localhost:5000/thrift-microservice-deps:xenial

ARG NUM_CPUS=40

COPY CMakeLists.txt /social-network-microservices/CMakeLists.txt
COPY cmake /social-network-microservices/cmake
COPY src /social-network-microservices/src
COPY third_party /social-network-microservices/third_party
COPY gen-cpp /social-network-microservices/gen-cpp
COPY gen-lua /social-network-microservices/gen-lua
COPY gen-py /social-network-microservices/gen-py
COPY config /social-network-microservices/config
COPY keys /keys

RUN cd /social-network-microservices \
    && mkdir -p build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Debug .. \
    && make -j${NUM_CPUS} \
    && make install

WORKDIR /social-network-microservices
