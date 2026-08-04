python3 benchmarks/benchmark_runner.py xpk \
    project=$PROJECT \
    zone=$ZONE \
    device_type=v6e-256 \
    num_slices=1  \
    cluster_name=$CLUSTER \
    base_output_directory=$OUTPUT_DIR \
    model_name="llama3_1_70b_8192_synthetic" \
    num_steps=100 \
    base_docker_image=maxtext_base_image
