FROM locustio/locust

ENV http_proxy=""
ENV https_proxy=""
ENV no_proxy=""

COPY docker-entrypoint.sh /docker-entrypoint.sh
COPY src /mnt/locust/src
COPY lib /mnt/locust/lib
COPY data /mnt/locust/data
COPY configs /mnt/locust/configs

USER root
RUN chown -R locust:locust /mnt/locust

USER locust
RUN mkdir -p /mnt/locust/logs
RUN pip install --user numpy
