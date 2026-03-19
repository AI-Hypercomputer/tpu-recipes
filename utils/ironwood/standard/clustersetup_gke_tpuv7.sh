#!/bin/bash

################################################################################
# This script automates the setup of GKE clusters and TPU v7 node pools.
# It includes the following robust features:
# - Pre-flight checks for required tools (gcloud, kubectl).
# - Interactive section selection menu with validation.
#
# The script is divided into three main sections:
# 1. Infrastructure Deployment:
#    - Sets up GCP project and zone.
#    - Creates a custom VPC network, subnet, firewall rule, router, and NAT.
#    - Creates a GKE cluster with Workload Identity, multi-networking, and DPv2.
#    - Installs the JobSet CRD.
#    - Adds a maintenance exclusion to prevent unexpected upgrades.
# 2. NodePool Creation (TPU v7):
#    - Supports both DWS Flex Start and Specific Reservations.
#    - Prompts for TPU topology and validates selection.
#    - Automatically creates required High Throughput workload policies.
#    - Generates unique node pool names.
#    - Dynamic sizing for 2x2x1 topology with user prompts and contextual defaults.
# 3. NodePool Deletion:
#    - Lists existing node pools (excluding default-pool).
#    - Interactive selection for deletion with confirmation prompts.
################################################################################

set -e  # Exit on error

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

################################################################################
# CONFIGURATION - MODIFY THESE VALUES
################################################################################

# MUST CHANGE VALUES
export PROJECT_ID="<PROJECT_ID>"              # REQUIRED: Your GCP project ID
export REGION="<REGION>"                      # GCP region for resources
export ZONE="<ZONE>"                      # GCP zone for zonal resources
export GKE_VERSION="<GKE_VERSION>"
export RESOURCE_NAME="<RESOURCE_NAME>"
export MAINTENANCE_EXCLUSION_START="<MAINTENANCE_EXCLUSION_START>" # Example: 2026-03-01T00:00:00Z
export MAINTENANCE_EXCLUSION_END="<MAINTENANCE_EXCLUSION_END>" # Example: 2026-03-29T00:00:00Z
export RESERVATION_NAME="" # Optional: Leave empty for DWS Flex Start

export CPU_MACHINE_TYPE="n2-standard-8"
export JOBSET_VERSION=v0.10.1

export CLUSTER_NAME=${RESOURCE_NAME}-gke
export NETWORK_NAME=${RESOURCE_NAME}-privatenetwork
export SUBNET_NAME=${RESOURCE_NAME}-privatesubnet
export NETWORK_FW_NAME=${RESOURCE_NAME}-privatefirewall
export ROUTER_NAME=${RESOURCE_NAME}-network
export NAT_CONFIG=${RESOURCE_NAME}-natconfig

# Different TPUv7x accelerator topologies here: https://docs.cloud.google.com/tpu/docs/tpu7x#configurations
export TOPOLOGIES=(2x2x1 2x2x2 2x2x4 2x4x4 4x4x4 4x4x8 4x8x8 8x8x8 8x8x16 8x16x16)
# Check for required tools
for tool in gcloud kubectl; do
    if ! command -v "$tool" &> /dev/null; then
        print_error "$tool is not installed. Please install it and try again."
        exit 1
    fi
done

################################################################################
# Script Execution Selection
################################################################################

while true; do
    print_header "Select a section to execute:"
    echo "1. Cluster Creation (Section 1)"
    echo "2. NodePool Creation (Section 2)"
    echo "3. NodePool Deletion (Section 3)"
    echo "4. Exit"
    read -p "Enter selection (1-4): " SELECTION
    
    case $SELECTION in
        1) SECTION_NUMBER=1; break ;;
        2) SECTION_NUMBER=2; break ;;
        3) SECTION_NUMBER=3; break ;;
        4) exit 0 ;;
        *) print_error "Invalid selection. Please enter a number between 1 and 4." ;;
    esac
done

################################################################################
# Cluster Creation (Section 1)
################################################################################
if [[ "${SECTION_NUMBER}" -eq 1 ]]; then

print_header "Running : Infrastructure deployment"

# Set project context
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}


# Creating network setups

if gcloud compute networks describe ${NETWORK_NAME} --project=${PROJECT_ID} &> /dev/null; then
    print_info "Network ${NETWORK_NAME} already exists."
else
    print_info "Creating network ${NETWORK_NAME}..."
    gcloud compute networks create ${NETWORK_NAME} --mtu=8896 --project=${PROJECT_ID} --subnet-mode=custom --bgp-routing-mode=regional
    print_success "Network ${NETWORK_NAME} created."
fi
if gcloud compute networks subnets describe ${SUBNET_NAME} --region=${REGION} --project=${PROJECT_ID} &> /dev/null; then
    print_info "Subnet ${SUBNET_NAME} already exists."
else
    print_info "Creating subnet ${SUBNET_NAME}..."
    gcloud compute networks subnets create "${SUBNET_NAME}" --network="${NETWORK_NAME}" --range=10.10.0.0/18 --region="${REGION}" --project=$PROJECT_ID
    print_success "Subnet ${SUBNET_NAME} created."
fi

if gcloud compute firewall-rules describe ${NETWORK_FW_NAME} --project=${PROJECT_ID} &> /dev/null; then
    print_info "Firewall rule ${NETWORK_FW_NAME} already exists."
else
    print_info "Creating firewall rule ${NETWORK_FW_NAME}..."
    gcloud compute firewall-rules create ${NETWORK_FW_NAME} --network ${NETWORK_NAME} --allow tcp,icmp,udp --project=${PROJECT_ID}
    print_success "Firewall rule ${NETWORK_FW_NAME} created."
fi

if gcloud compute routers describe ${ROUTER_NAME} --region=${REGION} --project=${PROJECT_ID} &> /dev/null; then
    print_info "Router ${ROUTER_NAME} already exists."
else
    print_info "Creating router ${ROUTER_NAME}..."
    gcloud compute routers create "${ROUTER_NAME}" \
      --project="${PROJECT_ID}" \
      --network="${NETWORK_NAME}" \
      --region="${REGION}"
    print_success "Router ${ROUTER_NAME} created."
fi

if gcloud compute routers nats describe ${NAT_CONFIG} --router=${ROUTER_NAME} --region=${REGION} --project=${PROJECT_ID} &> /dev/null; then
    print_info "NAT config ${NAT_CONFIG} already exists."
else
    print_info "Creating NAT config ${NAT_CONFIG}..."
    gcloud compute routers nats create "${NAT_CONFIG}" \
      --router="${ROUTER_NAME}" \
      --region="${REGION}" \
      --auto-allocate-nat-external-ips \
      --nat-all-subnet-ip-ranges \
      --project="${PROJECT_ID}" \
      --enable-logging
    print_success "NAT config ${NAT_CONFIG} created."
fi

#Creating GKE cluster

if gcloud container clusters describe ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID} &> /dev/null; then
    print_info "GKE cluster ${CLUSTER_NAME} already exists."
    # Check if Workload Identity is enabled
    WORKLOAD_POOL=$(gcloud container clusters describe ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID} --format="value(workloadIdentityConfig.workloadPool)")
    if [[ -z "${WORKLOAD_POOL}" ]]; then
        print_info "Workload Identity not enabled on ${CLUSTER_NAME}. Enabling..."
        gcloud container clusters update ${CLUSTER_NAME} \
          --location=${REGION} \
          --project=${PROJECT_ID} \
          --workload-pool=${PROJECT_ID}.svc.id.goog
        print_success "Workload Identity enabled on ${CLUSTER_NAME}."
    else
        print_info "Workload Identity is already enabled on ${CLUSTER_NAME}."
    fi
else
    print_info "Creating GKE cluster ${CLUSTER_NAME}..."
    gcloud container clusters create ${CLUSTER_NAME} \
        --release-channel=regular \
        --cluster-version=${GKE_VERSION} \
        --machine-type=${CPU_MACHINE_TYPE} \
        --region=${REGION} \
        --node-locations=${ZONE} \
        --project=${PROJECT_ID} \
        --enable-dataplane-v2 \
        --enable-ip-alias \
        --enable-multi-networking \
        --network=${NETWORK_NAME} \
        --subnetwork=${SUBNET_NAME} \
        --workload-pool=${PROJECT_ID}.svc.id.goog

      
    print_success "GKE cluster ${CLUSTER_NAME} created."
fi

# Connect to cluster
gcloud container clusters get-credentials ${CLUSTER_NAME} --location ${REGION} --project ${PROJECT_ID}

if kubectl get crd jobsets.jobset.x-k8s.io &> /dev/null; then
    print_info "JobSet is already installed."
else
    print_info "Installing JobSet ${JOBSET_VERSION}..."
    kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/${JOBSET_VERSION}/manifests.yaml
    print_success "JobSet ${JOBSET_VERSION} installed."
fi

# Add maintianance exclution
EXCLUSION_NAME="stop-upgrade-in-a-month"
EXISTING_EXCLUSION=$(gcloud container clusters describe ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID} --format="value(maintenancePolicy.window.maintenanceExclusions[${EXCLUSION_NAME}])")

if [[ -z "${EXISTING_EXCLUSION}" ]]; then
    print_info "Adding maintenance exclusion '${EXCLUSION_NAME}' to ${CLUSTER_NAME}..."
    gcloud container clusters update ${CLUSTER_NAME} \
      --location=${REGION} \
      --add-maintenance-exclusion-name="${EXCLUSION_NAME}" \
      --add-maintenance-exclusion-start="${MAINTENANCE_EXCLUSION_START}" \
      --add-maintenance-exclusion-end="${MAINTENANCE_EXCLUSION_END}" \
      --add-maintenance-exclusion-scope="no_upgrades"
    print_success "Maintenance exclusion '${EXCLUSION_NAME}' added to ${CLUSTER_NAME}."
else
    print_info "Maintenance exclusion '${EXCLUSION_NAME}' already exists on ${CLUSTER_NAME}."
fi

fi
################################################################################
# NodePool Creation (Section 2)
################################################################################

if [[ $SECTION_NUMBER -eq 2 ]]; then

# Connect to cluster
gcloud container clusters get-credentials ${CLUSTER_NAME} --location ${REGION} --project ${PROJECT_ID}

# Get TPU Topology from user
print_info "Available TPU Topologies: ${TOPOLOGIES[*]}"
read -p "Enter TPU Topology for the new NodePool: " NEW_TOPOLOGY

# Validate Topology
if [[ ! " ${TOPOLOGIES[@]} " =~ " ${NEW_TOPOLOGY} " ]]; then
    print_error "Invalid TPU Topology: ${NEW_TOPOLOGY}. Please choose from: ${TOPOLOGIES[*]}"
    exit 1
fi

# Calculate required chips for the new nodepool
IFS='x' read -ra DIMS <<< "${NEW_TOPOLOGY}"
REQUIRED_CHIP_COUNT=$((${DIMS[0]} * ${DIMS[1]} * ${DIMS[2]}))

# Create Resource Policy
WORKLOAD_POLICY_NAME="${RESOURCE_NAME}-workload-policy${NEW_TOPOLOGY}"
if [[ "${NEW_TOPOLOGY}" == "2x2x1" ]]; then
    print_info "Skipping workload policy creation for topology 2x2x1."
    WORKLOAD_POLICY_NAME=""
else
    if gcloud compute resource-policies describe ${WORKLOAD_POLICY_NAME} --project=${PROJECT_ID} --region=${REGION} &> /dev/null; then
        print_info "Workload policy ${WORKLOAD_POLICY_NAME} already exists."
    else
        print_info "Creating workload policy ${WORKLOAD_POLICY_NAME}..."
        gcloud compute resource-policies create workload-policy ${WORKLOAD_POLICY_NAME} \
            --type HIGH_THROUGHPUT \
            --accelerator-topology ${NEW_TOPOLOGY} \
            --project ${PROJECT_ID} \
            --region ${REGION}
        print_success "Workload policy ${WORKLOAD_POLICY_NAME} created."
    fi
fi

# Create NodePool
NODE_POOL_SIZE=$((${REQUIRED_CHIP_COUNT} / 4))
if [[ "${NEW_TOPOLOGY}" == "2x2x1" ]]; then
    DEFAULT_SIZE=1000
    if [[ -n "${RESERVATION_NAME}" ]]; then
        DEFAULT_SIZE=2
    fi
    if [[ -z "${RESERVATION_NAME}" ]]; then
        print_info "Default is 1000 since it's autoscaled for DWS Flex Start mode."
    fi
    read -p "Enter number of nodes for 2x2x1 (default: ${DEFAULT_SIZE}): " USER_SIZE
    NODE_POOL_SIZE=${USER_SIZE:-${DEFAULT_SIZE}}
fi
print_info "Calculated Node Pool Size: ${NODE_POOL_SIZE} nodes"

# Find the next available index for the nodepool name
RAND_STR=$(head /dev/urandom | tr -dc a-z0-9 | head -c 5)
while true; do
    NODE_POOL_NAME=${RESOURCE_NAME}-np${NEW_TOPOLOGY}-${RAND_STR}
    if ! gcloud container node-pools describe ${NODE_POOL_NAME} --cluster=${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID} &> /dev/null; then
        break
    fi
    RAND_STR=$(head /dev/urandom | tr -dc a-z0-9 | head -c 5)
done

print_info "Creating node pool ${NODE_POOL_NAME}..."
if [[ -n "${RESERVATION_NAME}" ]]; then
    CMD=(
        "gcloud" "container" "node-pools" "create" "${NODE_POOL_NAME}"
        "--cluster=${CLUSTER_NAME}"
        "--machine-type=tpu7x-standard-4t"
        "--location=${REGION}"
        "--node-locations=${ZONE}"
        "--project=${PROJECT_ID}"
        "--num-nodes=${NODE_POOL_SIZE}"
        "--reservation=${RESERVATION_NAME}"
        "--reservation-affinity=specific"
        "--workload-metadata=GKE_METADATA"
    )
else
    CMD=(
        "gcloud" "container" "node-pools" "create" "${NODE_POOL_NAME}"
        "--cluster=${CLUSTER_NAME}"
        "--machine-type=tpu7x-standard-4t"
        "--location=${REGION}"
        "--project=${PROJECT_ID}"
        "--enable-autoscaling"
        "--min-nodes=0"
        "--max-nodes=${NODE_POOL_SIZE}"
        "--num-nodes=0"
        "--flex-start"
        "--workload-metadata=GKE_METADATA"
        "--reservation-affinity=none"
    )
fi
if [[ "${NEW_TOPOLOGY}" != "2x2x1" ]]; then
    CMD+=("--placement-policy=${WORKLOAD_POLICY_NAME}")
fi
"${CMD[@]}"
    print_success "Node pool ${NODE_POOL_NAME} created."

fi

################################################################################
# NodePool Deletion (Section 3)
################################################################################

if [[ $SECTION_NUMBER -eq 3 ]]; then

# Connect to cluster
gcloud container clusters get-credentials ${CLUSTER_NAME} --location ${REGION} --project ${PROJECT_ID}

print_info "Fetching node pools..."
NODE_POOL_LIST=$(gcloud container node-pools list --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID} --format="table(name)" --filter="name != default-pool")

if [[ -z "$(echo "${NODE_POOL_LIST}" | tail -n +2)" ]]; then
    print_warning "No node pools found to delete (excluding default-pool)."
    exit 0
fi

print_info "Available Node Pools:"

# Read the node pool names and counts into arrays for selection
mapfile -t NP_LINES < <(echo "${NODE_POOL_LIST}" | tail -n +2)
NODE_POOLS=()
for line in "${NP_LINES[@]}"; do
    NODE_POOLS+=($(echo $line | awk '{print $1}'))
done

print_info "Select a node pool to delete:"
i=1
for j in "${!NODE_POOLS[@]}"; do
    print_success "$i. ${NODE_POOLS[$j]}"
    i=$((i + 1))
done

read -p "Enter the number of the node pool to delete: " NP_INDEX

if [[ ! "$NP_INDEX" =~ ^[0-9]+$ ]] || [[ $NP_INDEX -lt 1 || $NP_INDEX -gt ${#NODE_POOLS[@]} ]]; then
    print_error "Invalid selection."
    exit 1
fi

NODE_POOL_TO_DELETE=${NODE_POOLS[$((NP_INDEX - 1))]}

read -p "Are you sure you want to delete node pool '${NODE_POOL_TO_DELETE}'? (y/N): " CONFIRMATION

if [[ "$CONFIRMATION" =~ ^[Yy]$ ]]; then
    print_info "Deleting node pool ${NODE_POOL_TO_DELETE}..."
    gcloud container node-pools delete ${NODE_POOL_TO_DELETE} \
        --cluster=${CLUSTER_NAME} \
        --region=${REGION} \
        --project=${PROJECT_ID} \
        --quiet
    print_success "Node pool ${NODE_POOL_TO_DELETE} deleted."
else
    print_info "Node pool deletion cancelled."
fi

fi
