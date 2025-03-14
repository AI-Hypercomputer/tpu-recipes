# Run this command from the MaxText root directory using the setup described in the README.
python3 benchmarks/benchmark_runner.py xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-8 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME} \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="mistral_7b" \
    --base_docker_image=maxtext_base_image
