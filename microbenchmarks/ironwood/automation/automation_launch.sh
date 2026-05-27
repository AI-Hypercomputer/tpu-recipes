#!/usr/bin/env bash

######################################################################
#                            USER INPUT
######################################################################
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
export GCS_BUCKET_ROOT_DIR=""
export GCS_SA_NAME="gcs-writer"  # Service account with write access to GCS_BUCKET_ROOT_DIR
export PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

MAX_RETRIES=3
TIMEOUT_SECOND=7200

yaml_names=(
    "tpu7x-2x2x1-hbm.yaml"
    "tpu7x-2x2x1-host_device.yaml"
    "tpu7x-2x2x1-gemm_all_reduce.yaml"
    "tpu7x-2x2x1-gemm.yaml"
    "tpu7x-2x2x1-bmm.yaml"
    "tpu7x-2x2x1-attention.yaml"
    "tpu7x-2x2x1-collectives.yaml"
    "tpu7x-2x2x2-collectives.yaml"
    "tpu7x-2x2x4-collectives.yaml"
    "tpu7x-2x4x4-collectives.yaml"
    "tpu7x-4x4x4-collectives.yaml"
    "tpu7x-4x4x4-gemm_all_reduce.yaml"
)

######################################################################
#                        VALIDATION & SETUP
######################################################################

if [[ -z "${GCS_BUCKET_ROOT_DIR}" || "${GCS_BUCKET_ROOT_DIR}" != "gs://"* ]]; then
  echo "Error: GCS_BUCKET_ROOT_DIR must be set and start with gs://"
  exit 1
fi

echo "The intermediate result will be written to ${GCS_BUCKET_ROOT_DIR}"

required_topologies=($(printf "%s\n" "${yaml_names[@]}" | grep -oE '[0-9]+x[0-9]+x[0-9]+' | sort -u))

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if ! bash "${SCRIPT_DIR}/check_node_pool_setup.sh" "${required_topologies[@]}"; then
  exit 1
fi

for topology in "${required_topologies[@]}"; do
    export TOPOLOGY="${topology}"
    export TPUS=$(echo "${TOPOLOGY}" | sed 's/x/*/g' | bc)
    envsubst '${TOPOLOGY} ${TPUS}' < ${SCRIPT_DIR}/job-queue.yaml | kubectl apply -f -
done

######################################################################
#                  GCS PERMISSION CHECK
######################################################################

# Run the GCS permission check
export SA_NAME="${GCS_SA_NAME}"
export PROJECT_ID="${PROJECT_ID}"
if ! bash "${SCRIPT_DIR}/check_gcs_permissions.sh"; then
    echo "GCS Permission Check Failed. Exiting."
    exit 1
fi


######################################################################
#                 LAUNCH JOBS & WAIT FOR COMPLETION
######################################################################


# Function to wait for a job to complete or fail
wait_for_job_completion() {
    local job_name="$1"
    local timeout="$2"
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))

    while true; do
        current_time=$(date +%s)
        if [[ $current_time -gt $end_time ]]; then
            echo "Timeout waiting for job ${job_name}"
            return 2
        fi

        # Check for Complete condition
        if kubectl get job "${job_name}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null | grep -q "True"; then
            echo "Job ${job_name} completed successfully!"
            return 0
        fi

        # Check for Failed condition
        if kubectl get job "${job_name}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True"; then
            echo "Job ${job_name} FAILED!"
            return 1
        fi

        sleep 5
    done
}

# Function to apply jobs and wait for them to complete
# Returns a list of failed yaml files in the variable FAILED_JOBS
apply_and_wait() {
    local yaml_files=("$@")
    local job_names_in_batch=()
    FAILED_JOBS=()

    echo "Processing batch of ${#yaml_files[@]} jobs..."

    # Launch all jobs
    for yaml_file in "${yaml_files[@]}"; do
        local filepath="${SCRIPT_DIR}/${yaml_file}"
        # Derive job name: remove .yaml, lowercase, replace _ with -
        local job_name=$(basename "${yaml_file}" .yaml | tr '[:upper:]' '[:lower:]' | tr '_' '-')
        random_suffix=$(head /dev/urandom | tr -dc a-z0-9 | head -c 5)
        export JOB_NAME="${job_name}-${random_suffix}"
        export GCS_PATH="${GCS_BUCKET_ROOT_DIR}/${job_name}"
        
        echo "Launching job: ${filepath} (name: ${JOB_NAME})"
        envsubst '${JOB_NAME} ${GCS_PATH} ${GCS_SA_NAME}' < "${filepath}" | kubectl apply -f -
        job_names_in_batch+=("${JOB_NAME}")
    done

    # Monitor jobs
    local start_time=$(date +%s)
    local end_time=$((start_time + TIMEOUT_SECOND))
    local last_print_time=0
    
    while true; do
        local current_time=$(date +%s)
        if [[ $current_time -gt $end_time ]]; then
            echo "Timeout waiting for batch completion"
            break
        fi

        # Identify active jobs
        local active_jobs=()
        for job_name in "${job_names_in_batch[@]}"; do
            # Check for Complete
            if kubectl get job "${job_name}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null | grep -q "True"; then
                continue
            fi
            
            # Check for Failed
            if kubectl get job "${job_name}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True"; then
                continue
            fi
            
            # If neither, it's pending/running
            active_jobs+=("${job_name}")
        done

        if [[ ${#active_jobs[@]} -eq 0 ]]; then
            break
        fi

        # Dashboard View - Print every 60 seconds
        if [[ $((current_time - last_print_time)) -ge 60 ]]; then
            echo "======================================================================"
            date "+%Y-%m-%d %H:%M:%S"
            echo "----------------------------------------------------------------------"
            kubectl get jobs "${active_jobs[@]}"
            echo "======================================================================"
            last_print_time=$current_time
        fi
        
        sleep 10
    done

    # Collect results and cleanup
    FAILED_JOBS=()
    for i in "${!yaml_files[@]}"; do
        local yaml_file="${yaml_files[$i]}"
        local job_name="${job_names_in_batch[$i]}"
        local filepath="${SCRIPT_DIR}/${yaml_file}"
        
        # Check if failed or still running (timeout)
        if ! kubectl get job "${job_name}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null | grep -q "True"; then
             FAILED_JOBS+=("${yaml_files[$i]}")
        fi
        
        export JOB_NAME="${job_name}"
        export GCS_PATH="${GCS_BUCKET_ROOT_DIR}/${job_name}"
        envsubst '${JOB_NAME} ${GCS_PATH}' < "${filepath}" | kubectl delete -f - &> /dev/null
    done
}

# Retry loop
current_batch=("${yaml_names[@]}")

for (( retry=1; retry<=MAX_RETRIES; retry++ )); do
    apply_and_wait "${current_batch[@]}"

    if [[ ${#FAILED_JOBS[@]} -eq 0 ]]; then
        echo "All jobs completed successfully in Round ${retry}!"
        break
    fi

    echo "Round ${retry} finished. ${#FAILED_JOBS[@]} jobs failed."
    current_batch=("${FAILED_JOBS[@]}")

    if [[ ${retry} -lt ${MAX_RETRIES} ]]; then
        echo "Retrying failed jobs..."
        echo "========================================"
        echo "$((retry + 1)) / ${MAX_RETRIES}" max retries
        echo "========================================"
    else
        echo "Max retries reached. ¯\_(ツ)_/¯"
    fi
done

echo ""
echo "Jobs completed. Aggregating results..."
echo ""

# Ensure cleanup of any previous aggregator job to avoid immutable field errors
kubectl delete job aggregator --ignore-not-found=true

envsubst '${GCS_BUCKET_ROOT_DIR} ${GCS_SA_NAME}' < ${SCRIPT_DIR}/aggregator.yaml | kubectl apply -f -
wait_for_job_completion "aggregator" ${TIMEOUT_SECOND}
envsubst '${GCS_BUCKET_ROOT_DIR} ${GCS_SA_NAME}' < ${SCRIPT_DIR}/aggregator.yaml | kubectl delete -f -

# Print the failed jobs at the end for better visibility.

if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
    echo "The following jobs finally failed after ${MAX_RETRIES} rounds:"
    printf '%s\n' "${FAILED_JOBS[@]}"

    echo -e "\nTo retry manually, run:"
    for yaml_file in "${FAILED_JOBS[@]}"; do
        job_name=$(basename "${yaml_file}" .yaml | tr '[:upper:]' '[:lower:]' | tr '_' '-')
        GCS_PATH="${GCS_BUCKET_ROOT_DIR}/${job_name}"
        echo "JOB_NAME=\"${job_name}\" GCS_PATH=\"${GCS_PATH}\" GCS_SA_NAME=\"${GCS_SA_NAME}\" envsubst '\${JOB_NAME} \${GCS_PATH} \${GCS_SA_NAME}' < \"${SCRIPT_DIR}/${yaml_file}\" | kubectl apply -f -"
    done
else
    echo "Success! All jobs finished."
fi
