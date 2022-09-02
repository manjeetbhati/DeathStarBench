#!/bin/bash
set -eu
# nohup /usr/local/bin/locust $@ &
# respond to docker stop (handle SIGINT) properly
trap : TERM INT
sleep infinity & wait
