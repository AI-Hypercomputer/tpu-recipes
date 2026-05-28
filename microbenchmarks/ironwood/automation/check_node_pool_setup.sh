#!/bin/bash

required_chip="tpu7x"
if [[ $# -gt 0 ]]; then
  required_topologies=("$@")
else
  required_topologies=("2x2x1" "2x2x2" "2x2x4" "2x4x4" "4x4x4")
fi

echo "Checking for required GKE TPU configurations..."
echo "Required TPU Type: ${required_chip}"
echo "-----------------------------------------------------------------"

all_found=true
missing_topologies=()


for topology in "${required_topologies[@]}"; do
  echo -n "Checking for TPU topology '${topology}' with type '${required_chip}': "

  matching_nodes=$(kubectl get nodes -l cloud.google.com/gke-tpu-topology=${topology},cloud.google.com/gke-tpu-accelerator=${required_chip} -o custom-columns=NAME:.metadata.name --no-headers 2>/dev/null)

  if [[ -n "${matching_nodes}" ]]; then
    echo "FOUND"
  else
    echo "MISSING"
    missing_topologies+=("${topology}")
    all_found=false
  fi
done

echo "-----------------------------------------------------------------"

if [[ "${all_found}" = true ]]; then
  echo "SUCCESS: All required TPU configurations (topology + type) are present in the cluster."
  exit 0
else
  echo "FAILURE: One or more required TPU configurations are missing."
  echo "Missing topologies: ${missing_topologies[@]}"
  exit 1
fi
