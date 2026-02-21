# Instructions for training Mixtral 8x7B on Trillium TPU on multipod using XPK

## Environment Steup
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
```bash
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
Note thatt if the `gke-version` is not available anymore, pick one available from the error message from the terminal output.

---
### 5. Launch the training workload to XPK cluster.
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
INFO 2024-10-31T11:23:30.060710856Z ***** train metrics *****
INFO 2024-10-31T11:23:30.060713436Z epoch = 3.125
INFO 2024-10-31T11:23:30.060715786Z total_flos = 109152470400GF
INFO 2024-10-31T11:23:30.060718096Z train_loss = 7.4942
INFO 2024-10-31T11:23:30.060720436Z train_runtime = 0:49:59.16
INFO 2024-10-31T11:23:30.060722736Z train_samples = 32816
INFO 2024-10-31T11:23:30.060725156Z train_samples_per_second = 34.143
INFO 2024-10-31T11:23:30.060727556Z train_steps_per_second = 0.033
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
# locate the xplane.pb file and process
PYTHONPATH==$PROFILE_SCRIPT_PATH:$PYTHONPATH python $PROFILE_SCRIPT_PATH/profile_convert.py xplane.pb
```

You will see output like that tells the average step time in second:
```
Parsing plugins/profile/2024_10_31_00_44_09/127.0.0.1_9012.xplane.pb
Plane ID: 2, Name: /device:TPU:0
  Line ID: 2, Name: XLA Modules
    Event Metadata Name: SyncTensorsGraph.65923(1604004898989247534), ID: 36337, Duration: 16.780938099922 s
    Event Metadata Name: SyncTensorsGraph.65924(16619407271639597682), ID: 72675, Duration: 1.846361047078 s
    Event Metadata Name: SyncTensorsGraph.65924(16619407271639597682), ID: 72675, Duration: 1.845788159422 s
    Event Metadata Name: SyncTensorsGraph.65924(16619407271639597682), ID: 72675, Duration: 1.84276413525 s
    Event Metadata Name: SyncTensorsGraph.65924(16619407271639597682), ID: 72675, Duration: 1.838797222828 s
    Event Metadata Name: SyncTensorsGraph.65924(16619407271639597682), ID: 72675, Duration: 1.850977674094 s
    Event Metadata Name: SyncTensorsGraph.65924(16619407271639597682), ID: 72675, Duration: 1.862297948406 s
    Event Metadata Name: SyncTensorsGraph.65924(16619407271639597682), ID: 72675, Duration: 1.838890659 s
    Event Metadata Name: SyncTensorsGraph.65924(16619407271639597682), ID: 72675, Duration: 1.837627439172 s
    Event Metadata Name: SyncTensorsGraph.65924(16619407271639597682), ID: 72675, Duration: 1.835626750328 s
Got 10 iterations
1.8454
```
