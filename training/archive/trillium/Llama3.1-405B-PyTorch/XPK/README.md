# Instructions for training Llama 3.1 405B on Trillium TPU on multipod using XPK

The instructions and referenced docker image are optimized for training Llama 3.1
405B on two Trillium pods.

NOTE: the docker image contains a fork of `torch_xla`. We're working on
upstreaming the necessary dependencies. In the meantime, you may use this docker
image to study and reproduce the performance.

## Environment Setup
---
### 1. [Optional but suggested] Create virtual env
```bash
sudo apt-get update && sudo apt install python3.10-venv
python3.10 -m venv myenv
source myenv/bin/activate
```
---
### 2. Clone XPK repository and install XPK package
```bash
pushd ./
git clone https://github.com/google/xpk.git
cd xpk
pip install .
popd
```

---
### 3. Update and export environment variables
Modify environment variables in `env.sh` targetting your gcloud resource and the experiment model config. Source the script.
```
source env.sh
```

---
### 4. [Optional, skip if using existing XPK cluster] Create the XPK clusters
Please follow the corresponding XPK user guide to crea the XPK cluster first. If the cluster is already created, skip to Step 4.
```bash
NETWORK_NAME=${CLUSTER_NAME}-mtu9k
NETWORK_FW_NAME=${NETWORK_NAME}-fw

# Use a custom network for better performance as well as avoid the default network to be overloaded.
gcloud compute networks create ${NETWORK_NAME} --mtu=8896 --project=${PROJECT} --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create ${NETWORK_FW_NAME} --network ${NETWORK_NAME} --allow tcp,icmp,udp --project=${PROJECT}
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"

python3 xpk.py cluster create --cluster $CLUSTER_NAME --cluster-cpu-machine-type=n1-standard-8 --num-slices=$NUM_SLICES --tpu-type=$TPU_TYPE --zone=$ZONE  --project=$PROJECT --on-demand --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"  --create-vertex-tensorboard --gke-version=1.31.1-gke.1678000
```
Note that if the `gke-version` is not available anymore, pick one available from the error message from the terminal output.

---
### 5. Launch the Llama 3.1 training workload to XPK cluster.
```
bash benchmark.sh
```

Below is part of the sample output from

```
...
[XPK] Waiting for `Upload Docker Image`, for 7 seconds
sqpu-2024-11-01-01-15-40: digest: sha256:3fe8b828bc6f96b1c74220d90273147ee188601781330d3592bbffc4fa0897af size: 4951
[XPK] Task: `Upload Docker Image` terminated with code `0`
[XPK] Task: `Creating Workload` is implemented by `kubectl apply -f /tmp/tmpc65ikqh3`, streaming output live.
[XPK] Waiting for `Creating Workload`, for 0 seconds
jobset.jobset.x-k8s.io/piz-xpk-v6e-256 created
[XPK] Task: `Creating Workload` terminated with code `0`
[XPK] Task: `GKE Dashboard List` is implemented by `gcloud monitoring dashboards list --project=tpu-prod-env-automated --filter="displayName:'GKE - TPU Monitoring Dashboard'" --format="value(name)" --verbosity=error`, hiding output unless there is an error.
[XPK] No dashboard with displayName:'GKE - TPU Monitoring Dashboard' found in the project:tpu-prod-env-automated.
[XPK] Follow https://github.com/google/cloud-tpu-monitoring-debugging to deploy monitoring dashboard to view statistics and outlier mode of GKE metrics.
[XPK] Follow your workload here: https://console.cloud.google.com/kubernetes/service/us-east5/bodaborg-v6e-256/default/piz-xpk-v6e-256/details?project=tpu-prod-env-automated
[XPK] Exiting XPK cleanly
```

This will point you to a workload link `https://console.cloud.google.com/kubernetes/service/...`. Follow the workload link and check the log. If the training works correctly, we shall see below info from the log explorer:

```
...
INFO {'train_runtime': 3240.2466, 'train_samples_per_second': 1.58, 'train_steps_per_second': 0.003, 'train_loss': 117.80369873046875, 'epoch': 0.37}
INFO ***** train metrics *****
INFO epoch = 0.3704
INFO total_flos = 94629384960GF
INFO train_loss = 117.8037
INFO train_runtime = 0:54:00.24
INFO train_samples = 13983
INFO train_samples_per_second = 1.58
INFO train_steps_per_second = 0.003
...
EXIT_CODE=0
XPK End: Thu Oct 31 02:03:01 UTC 2024
```

---
### 6. [Optional] Metric processing

You can use the profile
```
# this is the place we place the profile processing script
export PROFILE_SCRIPT_PATH=../../../../utils/

# download the profile from gcp bucket to local
gcloud storage cp --recursive $PROFILE_LOG_DIR ./

# locate the profile output ending with ".pb".
# Name it xplane.pb file, and process it
PYTHONPATH==$PROFILE_SCRIPT_PATH:$PYTHONPATH python3 $PROFILE_SCRIPT_PATH/profile_convert.py xplane.pb
```

You will see output like that tells the average step time in second:
```
Parsing xplane.pb
Plane ID: 2, Name: /device:TPU:0
  Line ID: 2, Name: XLA Modules
    Event Metadata Name: SyncTensorsGraph.157979.161292(17070309993204983656), ID: 33756, Duration: 82.676126708172 s
    Event Metadata Name: SyncTensorsGraph.157979.161292(17070309993204983656), ID: 33756, Duration: 79.991382263094 s
    Event Metadata Name: SyncTensorsGraph.157979.161292(17070309993204983656), ID: 33756, Duration: 92.256847100156 s
    Event Metadata Name: SyncTensorsGraph.157979.161292(17070309993204983656), ID: 33756, Duration: 86.394679781422 s
    Event Metadata Name: SyncTensorsGraph.157979.161292(17070309993204983656), ID: 33756, Duration: 79.542469470578 s
    Event Metadata Name: SyncTensorsGraph.157979.161292(17070309993204983656), ID: 33756, Duration: 48.444764038344 s
Got 6 iterations
81.3338
```
