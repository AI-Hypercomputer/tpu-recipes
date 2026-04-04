# Serve GPT-OSS 120B with vLLM on Ironwood TPU with Google Cloud Storage

In this guide, we show how to serve
[GPT-OSS 120B](https://huggingface.co/openai/gpt-oss-120b) on Ironwood (TPU7x),
with the model stored in
[Google Cloud Storage](https://docs.cloud.google.com/storage/docs/introduction).

## Install `gcloud cli`

You can reproduce this experiment from your dev environment (e.g. your laptop).
You need to install `gcloud` locally to complete this tutorial.

To install `gcloud cli` please follow this guide:
[Install the gcloud CLI](https://cloud.google.com/sdk/docs/install#mac)

Once it is installed, you can login to GCP from your terminal with this command:
`gcloud auth login`.

## Cluster Prerequisites

Before deploying the vLLM workload, ensure your GKE cluster is configured with
the necessary networking and identity features.

### Define parameters

  ```bash
  # Set variables if not already set
  export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
  export PROJECT_ID=<YOUR_PROJECT_ID>
  export REGION=<YOUR_REGION>
  export ZONE=<YOUR_ZONE> # e.g., us-central1-a
  export NODEPOOL_NAME=<YOUR_NODEPOOL_NAME>
  export RESERVATION_NAME=<YOUR_RESERVATION_NAME> # Optional, if you have a reservation
  ```

### Create new cluster

Note: If a cluster already exists follows the steps in next section

The command below creates a cluster with the basic features required for this
recipe. For a more general guide on cluster creation please follow the GKE
documentation.

* [Creating Standard regional cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/creating-a-regional-cluster)
* [Creating Autopilot cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/creating-an-autopilot-cluster)
1. **Create GKE cluster with all the required features enabled**.

  ```bash
  gcloud container clusters create $CLUSTER_NAME \
    --project=$PROJECT_ID \
    --location=$REGION \
    --workload-pool=$PROJECT_ID.svc.id.goog \
    --release-channel=rapid \
    --num-nodes=1 \
    --gateway-api=standard \
    --addons GcsFuseCsiDriver,HttpLoadBalancing
  ```

### Updating existing cluster

Check if the following features are enabled in the cluster, if not use the
following steps to enable the required features.

1. **Enable Workload Identity:**. The cluster and the nodepool needs to have
    workload identity enabled.

    To enable in cluster:
    ```bash
    gcloud container clusters update ${CLUSTER_NAME} \
      --location=${REGION} \
      --workload-pool=PROJECT_ID.svc.id.goog
    ```

    To enable in existing nodepool.
    ```bash
    gcloud container node-pools update ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --workload-metadata=GKE_METADATA
    ```

1.  **Enable HTTP Load Balancing:** The cluster must have the
    `HttpLoadBalancing` add-on enabled. This is typically enabled by default,
    but you can confirm or add it:

    ```bash
    gcloud container clusters update ${CLUSTER_NAME} \
      --region ${REGION} \
      --project ${PROJECT_ID} \
      --update-addons=HttpLoadBalancing=ENABLED
    ```

2.  **Enable Cloud Storage FUSE CSI Driver:** Enable the Cloud Storage FUSE
    CSI driver to accss GCS bucket with GCSFuse:
    ```bash
    gcloud container clusters update ${CLUSTER_NAME} \
      --location ${REGION} \
      --project ${PROJECT_ID} \
      --update-addons GcsFuseCsiDriver=ENABLED
    ```

### Create nodepool

1. **Create TPU v7 (Ironwood) nodepool**. If a node pool does not already exist
create a node pool with a single TPU v7 node in 2x2x1 configuration.

    ```bash
    gcloud container node-pools create ${NODEPOOL_NAME} \
      --project=${PROJECT_ID} \
      --location=${REGION} \
      --node-locations=${ZONE} \
      --num-nodes=1 \
      --reservation=${RESERVATION_NAME} \
      --reservation-affinity=specific \
      --machine-type=tpu7x-standard-4t \
      --cluster=${CLUSTER_NAME}
    ```

## Storage Prerequisites

### Create GCS Bucket

Create a new GCS bucket following [instructions](https://docs.cloud.google.com/storage/docs/creating-buckets#create-bucket).

### Upload the Model Checkpoints

To download the model from HuggingFace, please follow the steps below:

1. [Mount the bucket](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/mount-bucket)
to your local system.

2. Access into the mount point and create the model folder.

3. Under the model folder,
[download](https://huggingface.co/docs/hub/en/models-downloading)
the model using the hf command:

```
hf download openai/gpt-oss-120b
```

### Grant Storage Permission to Kubernetes Service Account

For a cluster with
[Workload Identity Federation](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) enabled
, please follow
[these instructions](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity#kubernetes-resources-iam-policies)
to grant `roles/storage.objectAdmin` access to Kubernetes service
account.

## Deploy vLLM Workload on GKE

This recipe utilizes 50 nodes, totaling 200 TPUs. Please adjust the
`replicas` field in the GKE Deployment configuration below if you intend to
run the workload at a different scale
(Min: single node; Max: number of 2x2x1 nodepools in your cluster).

1.  Configure kubectl to communicate with your cluster

    ```bash
    gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION}
    ```

2.  Save this yaml file as `vllm-tpu.yaml`.

    Note: Please replace the `YOUR_GCS_BUCKET`, `YOUR_MODEL_FOLDER`
    and `YOUR_XLA_CACHE_FOLDER` values with your specific GCS bucket
    name and folder paths.

    ```
    apiVersion: v1
    kind: PersistentVolume
    metadata:
      name: vllm-pv
    spec:
      accessModes:
      - ReadWriteMany
      capacity:
        storage: 64Gi
      persistentVolumeReclaimPolicy: Retain
      storageClassName: gcsfuse-sc
      claimRef:
        namespace: default
        name: vllm-pvc
      mountOptions:
        - implicit-dirs
        - metadata-cache:negative-ttl-secs:0
        - metadata-cache:ttl-secs:-1
        - metadata-cache:stat-cache-max-size-mb:-1
        - metadata-cache:type-cache-max-size-mb:-1
        - file-system:kernel-list-cache-ttl-secs:-1
        - read:enable-buffered-read:true
        - read_ahead_kb=131072
      csi:
        driver: gcsfuse.csi.storage.gke.io
        volumeHandle: {YOUR_GCS_BUCKET}  # Replace the value with your actual GCS bucket.
        volumeAttributes:
          skipCSIBucketAccessCheck: "true"
          gcsfuseMetadataPrefetchOnMount: "true"
    ---
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: vllm-pvc
      namespace: default
    spec:
      accessModes:
      - ReadWriteMany
      resources:
        requests:
          storage: 64Gi
      volumeName: vllm-pv
      storageClassName: gcsfuse-sc
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-tpu
    spec:
      replicas: 50  # The recipe utilizes 50 nodes, totaling 200 TPUs. Please adjust this value if you intend to run the workload at a different scale.
      selector:
        matchLabels:
          app: vllm-tpu
      template:
        metadata:
          labels:
            app: vllm-tpu
          annotations:
            gke-gcsfuse/volumes: "true"
        spec:
          nodeSelector:
            cloud.google.com/gke-tpu-accelerator: tpu7x
            cloud.google.com/gke-tpu-topology: 2x2x1
          containers:
          - name: gke-gcsfuse-sidecar
            image: us.gcr.io/gke-release/gcr.io/gcs-fuse-csi-driver-sidecar-mounter:v1.22.4-gke.2
          - name: vllm-tpu
            image: vllm/vllm-tpu:nightly-20260316-e778980-57a314d
            command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
            args:
            - --host=0.0.0.0
            - --port=8000
            - --tensor-parallel-size=2
            - --data-parallel-size=4
            - --max-model-len=9216
            - --max-num-batched-tokens=16384
            - --max-num-seqs=2048
            - --no-enable-prefix-caching
            - --load-format=runai_streamer
            - --model=/model-vol-mount/$(MODEL_FOLDER_PATH)
            - --kv-cache-dtype=fp8
            - --async-scheduling
            - --gpu-memory-utilization=0.86
            env:
            - name: MODEL_FOLDER_PATH
              value: {YOUR_MODEL_FOLDER}  #  Please replace this with your actual GCS model folder path, omitting the gs://{YOUR_GCS_BUCKET}/ prefix.
            - name: TPU_BACKEND_TYPE
              value: jax
            - name: MODEL_IMPL_TYPE
              value: vllm
            - name: VLLM_XLA_CACHE_PATH
              value: /model-vol-mount/{YOUR_XLA_CACHE_FOLDER} # Please replace this with your actual XLA cache folder path. Provide a name for a new folder where the XLA cache will be saved if this is your first run.
            ports:
            - containerPort: 8000
            resources:
              limits:
                google.com/tpu: '4'
              requests:
                google.com/tpu: '4'
            readinessProbe:
              tcpSocket:
                port: 8000
              initialDelaySeconds: 15
              periodSeconds: 10
            volumeMounts:
            - mountPath: /model-vol-mount
              name: model-vol
            - mountPath: /dev/shm
              name: dshm
          volumes:
          - emptyDir:
              medium: Memory
            name: dshm
          - name: model-vol
            persistentVolumeClaim:
              claimName: vllm-pvc
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-service
    spec:
      selector:
        app: vllm-tpu
      type: LoadBalancer
      ports:
        - name: http
          protocol: TCP
          port: 8000
          targetPort: 8000
    ```

3.  Apply the vLLM manifest by running the following command

    ```bash
    kubectl apply -f vllm-tpu.yaml
    ```

    At the end of the server startup you’ll see logs such as:

    ```
    $ kubectl logs -f deployment/vllm-tpu
    …
    …
    (APIServer pid=1) INFO:     Started server process [1]
    (APIServer pid=1) INFO:     Waiting for application startup.
    (APIServer pid=1) INFO:     Application startup complete.
    ```

4.  Serve the model by port-forwarding the service

    ```bash
    kubectl port-forward service/vllm-service 8000:8000
    ```

5.  Interact with the model using curl (from your workstation/laptop)

    Note: Please replace `YOUR_MODEL_FOLDER` value
    with your specific model folder path.

    ```bash
    curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
        "model": "/model-vol-mount/{YOUR_MODEL_FOLDER}",  # Please replace this with your actual GCS model folder path, omitting the gs://{YOUR_GCS_BUCKET}/ prefix. Ensure this field matches the model flag specified in your server startup command.
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
    ```

### (Optional) Benchmark via Service

1.  Execute a short benchmark against the server using:

    Note: Please replace `YOUR_MODEL_FOLDER` value
    with your specific model folder path.

    ```
    apiVersion: v1
    kind: Pod
    metadata:
      name: vllm-bench
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      terminationGracePeriodSeconds: 60
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu7x
        cloud.google.com/gke-tpu-topology: 2x2x1
      containers:
      - name: gke-gcsfuse-sidecar
        image: us.gcr.io/gke-release/gcr.io/gcs-fuse-csi-driver-sidecar-mounter:v1.22.4-gke.2
      - name: vllm-bench
        image: vllm/vllm-tpu:latest
        command: ["vllm"]
        args:
        - bench
        - serve
        - --dataset-name=sonnet
        - --sonnet-input-len=1024
        - --sonnet-output-len=8192
        - --dataset-path=/workspace/vllm/benchmarks/sonnet.txt
        - --num-prompts=1000
        - --ignore-eos
        - --host=vllm-service
        - --port=8000
        - --model=/model-vol-mount/$(MODEL_FOLDER_PATH)
        env:
        - name: MODEL_FOLDER_PATH
          value: {YOUR_MODEL_FOLDER}  #  Please replace this with your actual GCS model folder path, omitting the gs://{YOUR_GCS_BUCKET}/ prefix.
        volumeMounts:
        - mountPath: /model-vol-mount
          name: model-vol
      volumes:
      - name: model-vol
        persistentVolumeClaim:
          claimName: vllm-pvc
    ```

Save this file as `vllm-benchmark.yaml`, then apply it using `kubectl apply -f
vllm-benchmark.yaml`.

1.  Check the progress of benchmark:

    ```
    $ kubectl logs -f vllm-bench
    …
    …
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Failed requests:                         0
    Benchmark duration (s):                  xx
    Total input tokens:                      xxx
    Total generated tokens:                  xxx
    Request throughput (req/s):              xx
    Output token throughput (tok/s):         xxx
    Peak output token throughput (tok/s):    xxx
    Peak concurrent requests:                1000.00
    Total Token throughput (tok/s):          xxx
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          xxx
    Median TTFT (ms):                        xxx
    P99 TTFT (ms):                           xxx
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          xxx
    Median TPOT (ms):                       xxx
    P99 TPOT (ms):                           xxx
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           xxx
    Median ITL (ms):                         xxx
    P99 ITL (ms):                            xxx
    ==================================================
    ```

2.  Clean up

    ```
    kubectl delete -f vllm-benchmark.yaml
    kubectl delete -f vllm-tpu.yaml
    ```
