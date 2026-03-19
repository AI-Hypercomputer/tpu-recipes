# Ironwood GKE TPU v7 Utility Scripts

This directory contains utility scripts for automating the lifecycle of GKE clusters and TPU v7 node pools. It provides two distinct implementation patterns for managing TPU resources:

## Directory Structure

| Directory | Description |
| :--- | :--- |
| **[standard/](standard/README.md)** | Uses a **standard node pool based approach** for managing TPU v7 resources. This includes interactive scripts for creating and deleting node pools. |
| **[ccc/](ccc/README.md)** | Uses the **Cloud Compute Class (CCC)** / [ComputeClass](https://cloud.google.com/kubernetes-engine/docs/concepts/compute-class) based approach. This approach focuses on defining TPU topologies as reusable templates. |

## Highlights

- **Robust Automation**: Scripts include pre-flight checks and automated VPC/Subnet/NAT setup.
- **Support for TPU v7**: Tailored to the latest TPU v7 configurations and topologies.
- **Interactive Experience**: Menus for selecting infrastructure deployment, resource creation, and cleanup.
- **Workload Integration**: Documentation on how to target these resources using `nodeSelector`.

## Core Scripts

- `standard/clustersetup_gke_tpuv7.sh`: Main entry point for the standard approach.
- `ccc/clustersetup_gke_ccc_tpuv7.sh`: Main entry point for the Custom Compute Class (CCC) approach.
- `clustercleanup_gke_tpuv7.sh`: A shared cleanup script to tear down the entire infrastructure (Network, Cluster, etc.).

---

For detailed instructions, please refer to the READMEs in the respective subdirectories.
