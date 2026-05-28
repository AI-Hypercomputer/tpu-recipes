#!/bin/bash

######################################################################
# create_ccc_templates.sh: Create Custom Compute Class templates
######################################################################
# This script creates the necessary Google Cloud Compute resource policies
# and Kubernetes Custom Compute Class (CCC) manifests for various TPU
# topologies.
#
# It iterates through a predefined list of TOPOLOGIES:
#   - For multi-host topologies, it creates a HIGH_THROUGHPUT
#     workload policy if it doesn't already exist.
#   - It then uses envsubst to populate a template YAML
#     (tpu-ccc-template.yaml) with the correct TPU_TOPOLOGY,
#     RESERVATION_NAME, PROJECT_ID, and POLICY_NAME.
#   - The resulting manifest is applied to the Kubernetes cluster using
#     kubectl apply.
#
# Required environment variables:
#   - RESERVATION_NAME: The name of the GCE reservation to use.
#   - PROJECT_ID: The Google Cloud Project ID.
#   - REGION: The Google Cloud Region.
#   - RESOURCE_NAME: A base name used for naming resources.
######################################################################

export RESERVATION_NAME="<RESERVATION_NAME>"


export TOPOLOGIES=(2x2x1 2x2x2 2x2x4 2x4x4 4x4x4 4x4x8)
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION=$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.topology\.kubernetes\.io/region}')
CLUSTER_NAME=$(kubectl config current-context | cut -d '_' -f 4)
export RESOURCE_NAME=${CLUSTER_NAME%-gke} # assumes cluster was created with setup script which creates cluster with ${RESOURCE_NAME}-gke as name
################################################################################
# COLOR OUTPUT
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_info "Creating CCC templates for all topoligies"
# Create workload policy
for TOPOLOGY in "${TOPOLOGIES[@]}"
do
    export TPU_TOPOLOGY=${TOPOLOGY}
    if [[ "${TOPOLOGY}" == "2x2x1" ]]; then
        print_warning "Skipping workload policy creation for ${TOPOLOGY} as it is not needed for single host topologies."
        export POLICY_NAME="" # No policy for single host
    else
        WORKLOAD_POLICY_NAME="${RESOURCE_NAME}-workload-policy${TOPOLOGY}"
        if gcloud compute resource-policies describe ${WORKLOAD_POLICY_NAME} --project=${PROJECT_ID} --region=${REGION} &> /dev/null; then
            print_info "Workload policy ${WORKLOAD_POLICY_NAME} already exists."
        else
            print_info "Creating workload policy ${WORKLOAD_POLICY_NAME}..."
            gcloud compute resource-policies create workload-policy ${WORKLOAD_POLICY_NAME} \
    --type HIGH_THROUGHPUT \
    --accelerator-topology ${TOPOLOGY} \
    --project ${PROJECT_ID} \
    --region ${REGION}
            print_success "Workload policy ${WORKLOAD_POLICY_NAME} created."
        fi
        export POLICY_NAME=${WORKLOAD_POLICY_NAME}
    fi

    echo "${TPU_TOPOLOGY} ${RESERVATION_NAME} ${PROJECT_ID} ${POLICY_NAME}"
    if [[ "${TOPOLOGY}" == "2x2x1" ]]; then
        envsubst '${TPU_TOPOLOGY} ${RESERVATION_NAME} ${PROJECT_ID}' < ${SCRIPT_DIR}/tpu-ccc-template.yaml | sed '/placement:/,/policyName:/d' | kubectl apply -f -
    else
        envsubst '${TPU_TOPOLOGY} ${RESERVATION_NAME} ${PROJECT_ID} ${POLICY_NAME}' < ${SCRIPT_DIR}/tpu-ccc-template.yaml | kubectl apply -f -
    fi
    print_success "Applied TPU Compute Class for ${TOPOLOGY}"
done
