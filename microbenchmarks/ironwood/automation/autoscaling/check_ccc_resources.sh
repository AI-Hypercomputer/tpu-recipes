#!/bin/bash

######################################################################
# check_ccc_resources.sh: Validate existence of CCC resources
######################################################################
# This script checks if the required Google Cloud Compute resource policies
# and Kubernetes Custom Compute Class (CCC) manifests exist for a given
# list of TPU topologies.
#
# It iterates through the provided TOPOLOGIES array:
#   - For multi-host topologies, it verifies the presence of the
#     expected workload policy using gcloud.
#   - It checks for the existence of the Custom Compute Class resource
#     in the Kubernetes cluster using kubectl.
#
# The script exits with status 1 if any required resource is missing,
# and status 0 if all resources are found.
######################################################################

export TOPOLOGIES=(2x2x1 2x2x2 2x2x4 2x4x4 4x4x4 4x4x8)
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION=$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.topology\.kubernetes\.io/region}')
CLUSTER_NAME=$(kubectl config current-context | cut -d '_' -f 4)
export RESOURCE_NAME=${CLUSTER_NAME%-gke}

################################################################################
# COLOR OUTPUT
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_info "Checking CCC resources for all topologies"
missing_resources=false

for TOPOLOGY in "${TOPOLOGIES[@]}"
do
    print_info "Checking resources for topology: ${TOPOLOGY}"
    # Check workload policy for multi-host topologies
    if [[ "${TOPOLOGY}" != "2x2x1" ]]; then
        WORKLOAD_POLICY_NAME="${RESOURCE_NAME}-workload-policy${TOPOLOGY}"
        if gcloud compute resource-policies describe ${WORKLOAD_POLICY_NAME} --project=${PROJECT_ID} --region=${REGION} &> /dev/null; then
            print_success "Workload policy ${WORKLOAD_POLICY_NAME} exists."
        else
            print_error "Workload policy ${WORKLOAD_POLICY_NAME} is MISSING."
            missing_resources=true
        fi
    else
        print_info "Skipping workload policy check for single-host topology ${TOPOLOGY}."
    fi

    # Check Custom Compute Class
    CCC_NAME="tpuv7-${TOPOLOGY}-class"
    if kubectl get computeclass ${CCC_NAME} &> /dev/null; then
        print_success "Custom Compute Class ${CCC_NAME} exists."
    else
        print_error "Custom Compute Class ${CCC_NAME} is MISSING."
        missing_resources=true
    fi
done

if [[ "${missing_resources}" == "true" ]]; then
    print_error "One or more required resources are missing. Please create them."
    exit 1
else
    print_success "All required CCC resources exist."
    exit 0
fi
