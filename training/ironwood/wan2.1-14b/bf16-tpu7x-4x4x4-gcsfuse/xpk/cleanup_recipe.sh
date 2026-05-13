#!/bin/bash

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
# ---

source ~/.local/bin/venv/bin/activate

# --- Environment Variables ---
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE=""
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-wan")-$(date +%Y%m%d-%H%M)"

xpk storage detach ${WORKLOAD_NAME}-maxdiffusion-data --project=${PROJECT_ID} --cluster=${CLUSTER_NAME} --zone=${ZONE}
xpk storage detach ${WORKLOAD_NAME}-checkpoint --project=${PROJECT_ID} --cluster=${CLUSTER_NAME} --zone=${ZONE}
