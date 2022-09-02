ARG BASE_IMAGE=centos:centos8.3.2011

FROM ${BASE_IMAGE} AS centos-intel-base
SHELL ["/bin/bash", "-c"]


# Intel Optimizations specific Envs
ENV KMP_AFFINITY='granularity=fine,verbose,compact,1,0' \
    KMP_BLOCKTIME=1 \
    KMP_SETTINGS=1 

# Set your proxies if under firewall
ENV http_proxy=""
ENV https_proxy=""
ENV no_proxy=""

ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV NO_PROXY=""
#ENV ONEDNN_MAX_CPU_ISA=avx512_core_vnni
#ENV ONEDNN_MAX_CPU_ISA=AVX2
#ENV DNNL_MAX_CPU_ISA=avx512_core_vnni

ENV LC_ALL=C

RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

ARG PYTHON=python3.8


RUN yum update -y && yum install -y \
    ${PYTHON} \
    python3-pip \
    which && \
    yum clean all


RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools


RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python


RUN yum update -y && \
    yum install -y \
        numactl \
        libXext \
        libSM \
        python3-tkinter && \
    pip install requests

RUN yum update -y && yum install -y numactl


COPY requirements.txt /social-network-microservices/requirements.txt
COPY src /social-network-microservices/src
COPY third_party /social-network-microservices/third_party
COPY gen-cpp /social-network-microservices/gen-cpp
COPY gen-lua /social-network-microservices/gen-lua
COPY gen-py /social-network-microservices/gen-py
COPY config /social-network-microservices/config
COPY keys /keys

#ENV DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# uncomment to set extra logs on
#ENV DNNL_VERBOSE=1

ENV OMP_NUM_THREADS=8 \
    TF_INTEROP_NUM_THREADS=2



RUN pip install --no-cache-dir -r /social-network-microservices/requirements.txt 
RUN pip install --no-cache-dir -r \
    /social-network-microservices/src/MediaFilterService/requirements.txt

WORKDIR /social-network-microservices/src/MediaFilterService
ENTRYPOINT ["python", "MediaFilterService.py"]
