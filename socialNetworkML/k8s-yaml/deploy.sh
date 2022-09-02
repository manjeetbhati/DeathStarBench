#!/bin/bash

NAMESPACE="deathstarbench-social-network"
# Create namespace
kubectl create namespace ${NAMESPACE}
# Deploy social network app under namespace
kubectl apply -f ./ -n ${NAMESPACE}
