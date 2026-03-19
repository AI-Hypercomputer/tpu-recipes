# Ironwood GKE TPU v7 Utility Scripts

This folder contains scripts for automating the lifecycle of GKE clusters and TPU v7 node pools. These scripts are designed for robustness, providing pre-flight checks and interactive menus for a seamless experience.

## Scripts Overview

### [clustersetup_gke_tpuv7.sh](clustersetup_gke_tpuv7.sh)
Automates the creation and management of infrastructure and node pools. It is divided into three sections:
- **Section 1: Infrastructure Deployment**: Sets up VPC Networking (Network, Subnet, Firewall, Router, NAT), GKE Cluster (with multi-networking and DPv2), and installs the JobSet CRD.
- **Section 2: NodePool Creation**: Interactive creation of TPU v7 node pools using either DWS Flex Start or specific reservations. Supports various topologies and automatically creates required workload policies.
- **Section 3: NodePool Deletion**: Interactive selection and deletion of existing TPU v7 node pools.

### [clustercleanup_gke_tpuv7.sh](clustercleanup_gke_tpuv7.sh)
Automates the teardown of the entire environment. It deletes:
- GKE Cluster
- Compute Resource Policies (Workload Policies)
- Cloud NAT & Router
- Firewall Rules
- Subnet & VPC Network

---

## Configuration

Before running the scripts, ensure the following variables are set at the top of the scripts or exported in your environment:

- `PROJECT_ID`: Your GCP Project ID.
- `REGION` / `ZONE`: The GCP region/zone for resources (e.g., `us-central1`, `us-central1-c`).
- `RESOURCE_NAME`: A unique prefix for naming all created resources.
- `GKE_VERSION`: The GKE cluster version.
- `MAINTENANCE_EXCLUSION_START` / `END`: Time window to prevent cluster upgrades during experiments.

---

## Getting Started

1.  **Grant Permissions**:
    ```bash
    chmod +x clustersetup_gke_tpuv7.sh clustercleanup_gke_tpuv7.sh
    ```
2.  **Run Setup**:
    ```bash
    ./clustersetup_gke_tpuv7.sh
    ```
    Select a section from the interactive menu to begin.
3.  **Run Cleanup**:
    ```bash
    ./clustercleanup_gke_tpuv7.sh
    ```
    Run this when you are finished to avoid unnecessary GCP costs.

---

## Usage in Workloads

To run workloads on the TPU v7 node pools created by this script, use the following `nodeSelector` in your Kubernetes configurations (e.g., Job, Deployment, JobSet):

```yaml
nodeSelector:
  cloud.google.com/gke-tpu-topology: <TOPOLOGY>
  cloud.google.com/gke-tpu-accelerator: tpu7x
```

> [!NOTE]
> Replace `<TOPOLOGY>` with the topology of the node pool you created (e.g., `2x2x2`, `2x2x4`).

---

