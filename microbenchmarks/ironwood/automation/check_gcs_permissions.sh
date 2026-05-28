#!/usr/bin/env bash

# This script checks if the configured Service Account has write permissions to the specified GCS bucket.
# If permissions are missing, it attempts to fix them by creating the SA and granting roles/storage.admin.
#
# Expected Environment Variables:
#   GCS_BUCKET_ROOT_DIR: The GCS path (must start with gs://)
#   SA_NAME: The Service Account name (default: gcs-writer)
#   PROJECT_ID: The GCP Project ID (optional, will try to detect if not set)

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SA_NAME="${SA_NAME:-gcs-writer}"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

if [[ -z "${GCS_BUCKET_ROOT_DIR}" || "${GCS_BUCKET_ROOT_DIR}" != "gs://"* ]]; then
  echo "Error: GCS_BUCKET_ROOT_DIR must be set and start with gs://"
  exit 1
fi

fix_gcs_permissions() {
    # See more context in https://docs.cloud.google.com/kubernetes-engine/docs/how-to/workload-identity#authenticating_to
    echo "Attempting to fix GCS permissions..."
    
    if [[ -z "${PROJECT_ID}" ]]; then
        echo "Error: PROJECT_ID is not set and could not be detected."
        echo "Please export PROJECT_ID=<your-project-id> and rerun."
        exit 1
    fi
    
    local bucket_name=$(echo "${GCS_BUCKET_ROOT_DIR}" | sed 's|^gs://||' | cut -d/ -f1)
    local ns_name="default"
    
    echo "Ensuring ServiceAccount ${SA_NAME} exists in namespace ${ns_name}..."
    kubectl create serviceaccount "${SA_NAME}" --namespace "${ns_name}" --dry-run=client -o yaml | kubectl apply -f -
    
    local project_number=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
    
    echo "Granting roles/storage.admin to ${SA_NAME} on gs://${bucket_name}..."
    gcloud storage buckets add-iam-policy-binding "gs://${bucket_name}" \
        --role=roles/storage.admin \
        --member="principal://iam.googleapis.com/projects/${project_number}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${ns_name}/sa/${SA_NAME}"
        
    echo "Permission fix command executed."
}

check_gcs_permission() {
    echo "Checking GCS write permissions..."
    export GCS_CHECK_PATH="${GCS_BUCKET_ROOT_DIR}/permission-check-$(date +%s).txt"
    export SA_NAME="${SA_NAME}"

    # Check if ServiceAccount exists first to fail fast
    if ! kubectl get serviceaccount "${SA_NAME}" &> /dev/null; then
        echo "ServiceAccount '${SA_NAME}' not found."
        return 1
    fi
    
    # Launch check pod
    # We capture the pod name from the output of kubectl create
    local apply_output=$(envsubst '${SA_NAME} ${GCS_CHECK_PATH}' < "${SCRIPT_DIR}/gcs-write.yaml" | kubectl create -f -)
    # output example: pod/gcs-writer-test-abcde created
    local pod_name=$(echo "${apply_output}" | awk -F'/' '{print $2}' | awk '{print $1}')
    
    echo "Launched GCS check pod: ${pod_name}"
    
    # Wait for completion
    local check_status="FAILED"
    for i in {1..20}; do
        sleep 5
        if kubectl get pod "${pod_name}" -o jsonpath='{.status.phase}' 2>/dev/null | grep -q "Succeeded"; then
            check_status="SUCCESS"
            break
        fi
        if kubectl get pod "${pod_name}" -o jsonpath='{.status.phase}' 2>/dev/null | grep -q "Failed"; then
            check_status="FAILED"
            break
        fi
    done

    # Check logs
    if kubectl logs "${pod_name}" 2>/dev/null | grep -q "GCS test complete!"; then
         echo "GCS permission check PASSED."
         check_status="SUCCESS"
    else
         echo "GCS permission check FAILED."
         check_status="FAILED"
         echo "Logs from ${pod_name}:"
         kubectl logs "${pod_name}" 2>/dev/null | tail -n 10
    fi
    
    # Cleanup
    kubectl delete pod "${pod_name}" --grace-period=0 --force &> /dev/null
    
    if [[ "${check_status}" != "SUCCESS" ]]; then
        return 1
    fi
    return 0
}

# Main Logic
echo "======================================================================"
echo "Starting GCS Permission Check (SA: ${SA_NAME}, Bucket: ${GCS_BUCKET_ROOT_DIR})"
echo "======================================================================"

if ! check_gcs_permission; then
    echo "GCS check failed. Attempting to fix..."
    fix_gcs_permissions
    
    echo "Retrying GCS check..."
    if ! check_gcs_permission; then
        echo "GCS permissions check failed even after attempted fix."
        echo "Please verify your Service Account '${SA_NAME}' has proper permissions on ${GCS_BUCKET_ROOT_DIR}"
        exit 1
    fi
fi

echo "GCS Check Verified Successfully."
