# Run this command from the MaxText root directory using the setup described in the README.
python3 benchmarks/benchmark_runner.py xpk \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --device_type=v6e-256 \
    --num_slices=2  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="mixtral_8x7b_dropped" \
    --base_docker_image=maxtext_base_image
