python3 benchmarks/benchmark_runner.py xpk \
    project=$PROJECT \
    zone=$ZONE \
    device_type=v6e-256 \
    num_slices=1  \
    cluster_name=${CLUSTER} \
    base_output_directory=/tmp/ckpt \
    model_name="llama3_1_70b_8192_rd_ckpt_grain" \
    num_steps=100 \
    base_docker_image=maxtext_base_image \
    xpk_storage=$DATASET_STORAGE_NAME xpk_storage=$CHECKPOINT_STORAGE_NAME
