# Ironwood GKE TPU v7 CCC Utility Scripts

This folder contains scripts and templates for automating the lifecycle of GKE clusters and Cloud Compute Class (CCC) templates for TPU v7.

## What is ComputeClass (CCC)?

[ComputeClass](https://cloud.google.com/kubernetes-engine/docs/concepts/compute-class) is a GKE resource that defines a set of requirements for a workload (e.g., TPU topology, reservation, flex-start). 

Unlike standard node pools, **ComputeClass does not create nodes immediately**. Instead, GKE uses the ComputeClass as a template to **automatically provision the required node pools only when a workload is submitted** that targets the class via a `nodeSelector`. This allows for highly efficient resource utilization and supports advanced features defining different options for capacity provisioning. In the template used in this directory we define two options for capacity provisioning options, reserved capacity and flex-start capacity. If the nodepool creation based on the reseravation fails, CCC will attempt to create the nodepool based on the flex-start option.

## Scripts Overview

### [clustersetup_gke_ccc_tpuv7.sh](clustersetup_gke_ccc_tpuv7.sh)
Automates the creation and management of infrastructure and ComputeClass (CCC) templates. It is divided into three sections:
- **Section 1: Infrastructure Deployment**: Sets up VPC Networking (Network, Subnet, Firewall, Router, NAT), GKE Cluster (with multi-networking and DPv2), and installs the JobSet CRD.
- **Section 2: CCC Template Creation**: Interactive creation of TPU v7 ComputeClass templates using either DWS Flex Start or specific reservations. Supports various topologies and automatically creates required High Throughput workload policies.
- **Section 3: CCC Template Deletion**: Interactive selection and deletion of existing ComputeClass resources.

## Templates

- **[tpu-ccc-template.yaml](tpu-ccc-template.yaml)**: The base template for rendering ComputeClass resources.

---

## Configuration

Before running the scripts, ensure the following variables are set at the top of the scripts or exported in your environment:

- `PROJECT_ID`: Your GCP Project ID.
- `REGION` / `ZONE`: The GCP region/zone for resources (e.g., `us-central1`, `us-central1-ai1a`).
- `RESOURCE_NAME`: A unique prefix for naming all created resources.
- `GKE_VERSION`: The GKE cluster version.
- `MAINTENANCE_EXCLUSION_START` / `END`: Time window to prevent cluster upgrades during experiments.

---

## Usage in Workloads

To run workloads on the TPU v7 ComputeClass (CCC) templates created by this script, use the following `nodeSelector` in your Kubernetes configurations:

```yaml
nodeSelector:
  cloud.google.com/compute-class: tpuv7-<TOPOLOGY>-class
```

> [!NOTE]
> Replace `<TOPOLOGY>` with the topology of the CCC template you created (e.g., `2x2x2`, `2x2x4`).

---

## Getting Started

1.  **Grant Permissions**:
    ```bash
    chmod +x clustersetup_gke_ccc_tpuv7.sh
    ```
2.  **Run Setup**:
    ```bash
    ./clustersetup_gke_ccc_tpuv7.sh
    ```
    Select a section from the interactive menu to begin.