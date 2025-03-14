# Run this command from the MaxText root directory using the setup described in the README.
python3 benchmarks/benchmark_runner.py xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="llama2_70b_4096_sc" \
    --base_docker_image=maxtext_base_image
