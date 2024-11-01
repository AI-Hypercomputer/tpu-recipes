#!/bin/bash

source env.sh

python3 xpk.py workload create \
--cluster ${CLUSTER_NAME} \
--base-docker-image=${BASE_DOCKER_IMAGE} \
--workload=${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICE} \
--on-demand \
--zone=$ZONE \
--project=$PROJECT \
--enable-debug-logs \
--command="bash /app/train.sh"