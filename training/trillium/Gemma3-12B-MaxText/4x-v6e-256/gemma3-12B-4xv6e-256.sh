# Run this command from the MaxText root directory using the setup described in the README.
python3 -m benchmarks.benchmark_runner xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=4  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="gemma3_12b_32768_4x_v6e256" \
    --base_docker_image=maxtext_base_image
