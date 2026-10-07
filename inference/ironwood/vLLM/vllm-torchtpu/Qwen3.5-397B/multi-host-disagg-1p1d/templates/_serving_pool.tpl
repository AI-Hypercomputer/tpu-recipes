{{/*
Disaggregated serving worker pool (framework-neutral: vLLM today, SGLang/TRT-LLM later) for tpu7x (used for both prefill and decode).

Renders a Deployment when the pool is single-node (num_nodes == 1) or a
LeaderWorkerSet when multi-node (num_nodes > 1). Topology is derived per-pool
from `.pool.num_nodes` (NOT the global `.Values.is_multi_host`), so prefill and
decode may independently be single- or multi-host.

Invoke: {{ include "tpu7x.serving.pool" (dict "ctx" $ "pool" .Values.prefill "role" "prefill") }}

This template is KV-agnostic: KV connector wiring (producer/consumer, transport)
is supplied entirely via `.pool.bash_command` and `.pool.server_env_vars`.
Assumes GKE serving (run_mode != 1); disaggregation is a cloud/GKE feature.
*/}}

{{- define "tpu7x.serving.pool.nodeSelector" -}}
{{- $ctx := .ctx -}}
{{- $pool := .pool -}}
cloud.google.com/gke-tpu-accelerator: tpu7x
cloud.google.com/gke-tpu-topology: {{ $pool.topology }}
{{- if $ctx.Values.gke_reservation }}
cloud.google.com/reservation-name: {{ $ctx.Values.gke_reservation | quote }}
{{- end }}
{{- if $ctx.Values.placement_policy }}
cloud.google.com/placement-policy-name: {{ $ctx.Values.placement_policy | quote }}
{{- end }}
{{- end -}}

{{/*
Node and process limits required by the TPU KV-transfer path (Raiden /
tpu_sync). Every P/D transfer pins host memory and registers DMA entries, so
with the kernel defaults (vm.max_map_count=65530, vfio dma_entry_limit=65535)
a prefill worker exhausts them after roughly a thousand long-context requests
and aborts with "mmap error: 12" / "Resource temporarily unavailable", which
restarts the engine mid-benchmark. The reference vLLM-TorchTPU recipes raise
both limits from a privileged init container and run the engine privileged
with IPC_LOCK; this mirrors that.
*/}}
{{- define "tpu7x.serving.pool.initContainers" -}}
{{- $ctx := .ctx -}}
- name: tpu-node-setup
  image: {{ $ctx.Values.tpu_node_setup_image | default "busybox" }}
  command: ["/bin/sh", "-c"]
  args:
  - |
    sysctl -w vm.max_map_count=8388608
    if [ -f /sys/module/vfio_iommu_type1/parameters/dma_entry_limit ]; then
      echo 2000000 > /sys/module/vfio_iommu_type1/parameters/dma_entry_limit
    fi
  securityContext:
    privileged: true
{{- end -}}

{{- define "tpu7x.serving.pool.securityContext" -}}
privileged: true
capabilities:
  add: ["IPC_LOCK"]
{{- end -}}

{{- define "tpu7x.serving.pool.env" -}}
{{- $ctx := .ctx -}}
{{- $pool := .pool -}}
- name: HF_HOME
  value: /data
- name: HUGGING_FACE_HUB_TOKEN
  valueFrom:
    secretKeyRef:
      name: "{{ $ctx.Release.Name }}-hf-secret"
      key: hf_api_token
# P/D pool: advertise each pod's own routable pod IP so the TPUConnector JAX
# transfer server + ZMQ side-channel bind/announce a routable address (decode
# pulls KV from prefill at this IP). Applies to single-host pools and to every
# node of a multi-host pool -- each node transfers its own KV-cache shard, so the
# leader and all workers must announce routable IPs. Without it, get_ip() may
# pick a non-routable interface -- the TPU analog of the GPU NIXL localhost bug.
- name: VLLM_HOST_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
{{- if and (gt (int $pool.num_nodes) 1) (ne $pool.engine_label "TORCH_TPU") }}
- name: TPU_MULTIHOST_BACKEND
  value: ray
- name: JAX_PLATFORMS
  value: ""
{{- end }}
{{- if $pool.model_impl_type }}
- name: MODEL_IMPL_TYPE
  value: {{ $pool.model_impl_type }}
{{- end }}
{{- if $pool.profile }}
- name: PHASED_PROFILING_DIR
  value: {{ $pool.phased_profiling_dir | default "/tmp/results/profile" | quote }}
{{- end }}
{{- range $k, $v := $pool.server_env_vars }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
# KV connectors that must advertise a routable host to cross-host peers (e.g.
# torchtpu Raiden's TPU_RAIDEN_ADVERTISE_HOST) get the pod's own routable pod
# IP here; a static server_env_vars string cannot express the downward API.
{{- range $pool.pod_ip_env_vars }}
- name: {{ . }}
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
{{- end }}
{{- end -}}

{{- define "tpu7x.serving.pool.readinessProbe" -}}
{{- $pool := .pool -}}
tcpSocket:
  port: 8000
{{- if $pool.readiness_probe }}
initialDelaySeconds: {{ $pool.readiness_probe.initial_delay_seconds | default 15 }}
periodSeconds: {{ $pool.readiness_probe.period_seconds | default 10 }}
failureThreshold: {{ $pool.readiness_probe.failure_threshold | default 3 }}
{{- else }}
initialDelaySeconds: {{ $pool.ready_check_timeout_sec | default 15 }}
periodSeconds: 10
{{- end }}
{{- end -}}

{{- define "tpu7x.serving.pool.ports" -}}
{{- $pool := .pool -}}
- containerPort: 8000
{{- if $pool.kv_port }}
- containerPort: {{ $pool.kv_port }}
{{- end }}
{{- end -}}

{{- define "tpu7x.serving.pool.volumeMounts" -}}
{{- $ctx := .ctx -}}
{{- $pool := .pool -}}
{{- range $ctx.Values.custom_volume_mounts }}
- name: {{ .name }}
  mountPath: {{ .mountPath }}
  {{- if .readOnly }}
  readOnly: {{ .readOnly }}
  {{- end }}
  {{- if .subPath }}
  subPath: {{ .subPath }}
  {{- end }}
{{- end }}
{{- range $pool.pvc_mounts }}
- mountPath: {{ .mount_path | quote }}
  name: {{ .volume_name | quote }}
  subPath: {{ tpl .sub_path $ctx | quote }}
  readOnly: {{ .read_only }}
{{- end }}
- mountPath: "/data"
  name: data-volume
- mountPath: /dev/shm
  name: dshm
{{- if $ctx.Values.gcsfuse.enabled }}
{{- range $ctx.Values.gcsfuse.volumes }}
- name: {{ .name }}
  mountPath: {{ .mountPath | quote }}
  readOnly: {{ .readOnly }}
{{- end }}
{{- end }}
{{- end -}}

{{/* Volumes shared by both controllers, excluding the pool's data-volume. */}}
{{- define "tpu7x.serving.pool.volumesCommon" -}}
{{- $ctx := .ctx -}}
{{- $pool := .pool -}}
{{- range $ctx.Values.custom_volume_mounts }}
- name: {{ .name }}
  persistentVolumeClaim:
    claimName: {{ $ctx.Release.Name }}-{{ .pvcName }}
{{- end }}
{{- range $pool.pvc_mounts }}
- name: {{ .volume_name | quote }}
  persistentVolumeClaim:
    claimName: {{ .pvc_name | quote }}
{{- end }}
- emptyDir:
    medium: Memory
  name: dshm
{{- if $ctx.Values.gcsfuse.enabled }}
{{- range $ctx.Values.gcsfuse.volumes }}
- name: {{ .name }}
  csi:
    driver: gcsfuse.csi.storage.gke.io
    readOnly: {{ .readOnly }}
    volumeAttributes:
      bucketName: {{ .bucketName }}
      mountOptions: {{ .mountOptions | quote }}
{{- end }}
{{- end }}
{{- end -}}

{{/* ------------------------- pool controller ------------------------- */}}
{{- define "tpu7x.serving.pool" -}}
{{- $ctx := .ctx -}}
{{- $pool := .pool -}}
{{- $role := .role -}}
{{- $name := printf "%s-vllm-%s" $ctx.Release.Name $role -}}
{{- $app := printf "%s-vllm-%s-pod" $ctx.Release.Name $role -}}
{{- $lwsname := printf "%s-vllm-%s-lws" $ctx.Release.Name $role -}}
{{- if gt (int $pool.num_nodes) 1 }}
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: {{ $lwsname | quote }}
  annotations:
    leaderworkerset.sigs.k8s.io/exclusive-topology: {{ $pool.exclusive_topology | default "cloud.google.com/gke-nodepool" | quote }}
  labels:
    {{- if $ctx.Values.kueue_local_queue }}
    kueue.x-k8s.io/queue-name: {{ $ctx.Values.kueue_local_queue | quote }}
    {{- end }}
spec:
  replicas: {{ $pool.replicas | default 1 }}
  rolloutStrategy:
    type: "RollingUpdate"
  leaderWorkerTemplate:
    size: {{ $pool.num_nodes }}
    restartPolicy: RecreateGroupOnPodRestart
    {{- if not $ctx.Values.kueue_local_queue }}
    leaderTemplate:
      metadata:
        {{- if $ctx.Values.gcsfuse.enabled }}
        annotations:
          gke-gcsfuse/volumes: "true"
        {{- end }}
        labels:
          role: leader
          app: {{ $app | quote }}
          app.kubernetes.io/instance: "{{ $ctx.Release.Name }}"
          app.kubernetes.io/component: server
      spec:
        {{- if $ctx.Values.kueue_priority_class }}
        priorityClassName: {{ $ctx.Values.kueue_priority_class | quote }}
        {{- end }}
        {{- if $ctx.Values.gcp_service_account }}
        serviceAccountName: {{ $ctx.Values.k8s_service_account | quote }}
        {{- end }}
        nodeSelector:
          {{- include "tpu7x.serving.pool.nodeSelector" (dict "ctx" $ctx "pool" $pool) | nindent 10 }}
        tolerations:
        - key: "google.com/tpu"
          operator: "Exists"
        initContainers:
          {{- include "tpu7x.serving.pool.initContainers" (dict "ctx" $ctx) | nindent 8 }}
        containers:
        - name: vllm-server
          image: {{ $pool.image }}
          imagePullPolicy: IfNotPresent
          securityContext:
            {{- include "tpu7x.serving.pool.securityContext" . | nindent 12 }}
          command: ["/bin/bash", "-c"]
          args:
          - {{ printf "if [ \"${LWS_WORKER_INDEX:-0}\" = \"0\" ]; then\n%s\nelse\n%s\nfi" $pool.bash_command $pool.worker_bash_command | quote }}
          env:
            {{- include "tpu7x.serving.pool.env" (dict "ctx" $ctx "pool" $pool) | nindent 10 }}
          resources:
            limits:
              google.com/tpu: "{{ $pool.num_chips_per_node }}"
            requests:
              google.com/tpu: "{{ $pool.num_chips_per_node }}"
          ports:
            {{- include "tpu7x.serving.pool.ports" (dict "pool" $pool) | nindent 10 }}
          readinessProbe:
            exec:
              command:
              - /bin/bash
              - -c
              - {{ "if [ \"${LWS_WORKER_INDEX:-0}\" = \"0\" ]; then (echo > /dev/tcp/localhost/8000) >/dev/null 2>&1; else exit 0; fi" | quote }}
            {{- if $pool.readiness_probe }}
            initialDelaySeconds: {{ $pool.readiness_probe.initial_delay_seconds | default 15 }}
            periodSeconds: {{ $pool.readiness_probe.period_seconds | default 10 }}
            failureThreshold: {{ $pool.readiness_probe.failure_threshold | default 3 }}
            {{- else }}
            initialDelaySeconds: {{ $pool.ready_check_timeout_sec | default 15 }}
            periodSeconds: 10
            {{- end }}
          volumeMounts:
            {{- include "tpu7x.serving.pool.volumeMounts" (dict "ctx" $ctx "pool" $pool) | nindent 10 }}
        volumes:
          {{- include "tpu7x.serving.pool.volumesCommon" (dict "ctx" $ctx "pool" $pool) | nindent 8 }}
        {{- if $pool.data_disk_size }}
        - name: data-volume
          ephemeral:
            volumeClaimTemplate:
              metadata:
                labels:
                  app.kubernetes.io/instance: "{{ $ctx.Release.Name }}"
              spec:
                accessModes: [ "ReadWriteOnce" ]
                storageClassName: "{{ $ctx.Release.Name }}-hyperdisk-balanced-tpu"
                resources:
                  requests:
                    storage: {{ $pool.data_disk_size }}
        {{- else }}
        - emptyDir:
            medium: Memory
          name: data-volume
        {{- end }}
    {{- end }}
    workerTemplate:
      metadata:
        {{- if $ctx.Values.gcsfuse.enabled }}
        annotations:
          gke-gcsfuse/volumes: "true"
        {{- end }}
        labels:
          role: worker
          app: {{ $app | quote }}
          app.kubernetes.io/instance: "{{ $ctx.Release.Name }}"
          app.kubernetes.io/component: server
      spec:
        {{- if $ctx.Values.kueue_priority_class }}
        priorityClassName: {{ $ctx.Values.kueue_priority_class | quote }}
        {{- end }}
        {{- if $ctx.Values.gcp_service_account }}
        serviceAccountName: {{ $ctx.Values.k8s_service_account | quote }}
        {{- end }}
        nodeSelector:
          {{- include "tpu7x.serving.pool.nodeSelector" (dict "ctx" $ctx "pool" $pool) | nindent 10 }}
        tolerations:
        - key: "google.com/tpu"
          operator: "Exists"
        initContainers:
          {{- include "tpu7x.serving.pool.initContainers" (dict "ctx" $ctx) | nindent 8 }}
        containers:
        - name: vllm-server
          image: {{ $pool.image }}
          imagePullPolicy: IfNotPresent
          securityContext:
            {{- include "tpu7x.serving.pool.securityContext" . | nindent 12 }}
          command: ["/bin/bash", "-c"]
          args:
          - {{ printf "if [ \"${LWS_WORKER_INDEX:-0}\" = \"0\" ]; then\n%s\nelse\n%s\nfi" $pool.bash_command $pool.worker_bash_command | quote }}
          env:
            {{- include "tpu7x.serving.pool.env" (dict "ctx" $ctx "pool" $pool) | nindent 10 }}
          resources:
            limits:
              google.com/tpu: "{{ $pool.num_chips_per_node }}"
            requests:
              google.com/tpu: "{{ $pool.num_chips_per_node }}"
          ports:
            {{- include "tpu7x.serving.pool.ports" (dict "pool" $pool) | nindent 10 }}
          readinessProbe:
            exec:
              command:
              - /bin/bash
              - -c
              - {{ "if [ \"${LWS_WORKER_INDEX:-0}\" = \"0\" ]; then (echo > /dev/tcp/localhost/8000) >/dev/null 2>&1; else exit 0; fi" | quote }}
            {{- if $pool.readiness_probe }}
            initialDelaySeconds: {{ $pool.readiness_probe.initial_delay_seconds | default 15 }}
            periodSeconds: {{ $pool.readiness_probe.period_seconds | default 10 }}
            failureThreshold: {{ $pool.readiness_probe.failure_threshold | default 3 }}
            {{- else }}
            initialDelaySeconds: {{ $pool.ready_check_timeout_sec | default 15 }}
            periodSeconds: 10
            {{- end }}
          volumeMounts:
            {{- include "tpu7x.serving.pool.volumeMounts" (dict "ctx" $ctx "pool" $pool) | nindent 10 }}
        volumes:
          {{- include "tpu7x.serving.pool.volumesCommon" (dict "ctx" $ctx "pool" $pool) | nindent 8 }}
        {{- if $pool.data_disk_size }}
        - name: data-volume
          ephemeral:
            volumeClaimTemplate:
              metadata:
                labels:
                  app.kubernetes.io/instance: "{{ $ctx.Release.Name }}"
              spec:
                accessModes: [ "ReadWriteOnce" ]
                storageClassName: "{{ $ctx.Release.Name }}-hyperdisk-balanced-tpu"
                resources:
                  requests:
                    storage: {{ $pool.data_disk_size }}
        {{- else }}
        - emptyDir:
            medium: Memory
          name: data-volume
        {{- end }}
{{- else }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ $name | quote }}
spec:
  replicas: {{ $pool.replicas | default 1 }}
  selector:
    matchLabels:
      app: {{ $app | quote }}
  template:
    metadata:
      {{- if $ctx.Values.gcsfuse.enabled }}
      annotations:
        gke-gcsfuse/volumes: "true"
      {{- end }}
      labels:
        app: {{ $app | quote }}
        app.kubernetes.io/instance: "{{ $ctx.Release.Name }}"
        app.kubernetes.io/component: server
        {{- if $ctx.Values.kueue_local_queue }}
        kueue.x-k8s.io/queue-name: {{ $ctx.Values.kueue_local_queue | quote }}
        {{- end }}
    spec:
      {{- if $ctx.Values.kueue_priority_class }}
      priorityClassName: {{ $ctx.Values.kueue_priority_class | quote }}
      {{- end }}
      {{- if $ctx.Values.gcp_service_account }}
      serviceAccountName: {{ $ctx.Values.k8s_service_account | quote }}
      {{- end }}
      nodeSelector:
        {{- include "tpu7x.serving.pool.nodeSelector" (dict "ctx" $ctx "pool" $pool) | nindent 8 }}
      tolerations:
      - key: "google.com/tpu"
        operator: "Exists"
      initContainers:
        {{- include "tpu7x.serving.pool.initContainers" (dict "ctx" $ctx) | nindent 6 }}
      containers:
      - name: {{ if eq $pool.engine_label "TORCH_TPU" }}vllm-torchtpu{{ else }}vllm-tpu{{ end }}
        image: {{ $pool.image }}
        imagePullPolicy: IfNotPresent
        securityContext:
          {{- include "tpu7x.serving.pool.securityContext" . | nindent 10 }}
        command: ["/bin/bash", "-c"]
        args:
        - {{ $pool.bash_command | quote }}
        env:
          {{- include "tpu7x.serving.pool.env" (dict "ctx" $ctx "pool" $pool) | nindent 8 }}
        ports:
          {{- include "tpu7x.serving.pool.ports" (dict "pool" $pool) | nindent 8 }}
        resources:
          limits:
            google.com/tpu: "{{ $pool.num_chips_per_node }}"
          requests:
            google.com/tpu: "{{ $pool.num_chips_per_node }}"
        readinessProbe:
          {{- include "tpu7x.serving.pool.readinessProbe" (dict "pool" $pool) | nindent 10 }}
        volumeMounts:
          {{- include "tpu7x.serving.pool.volumeMounts" (dict "ctx" $ctx "pool" $pool) | nindent 8 }}
      volumes:
        {{- include "tpu7x.serving.pool.volumesCommon" (dict "ctx" $ctx "pool" $pool) | nindent 6 }}
      - name: data-volume
        persistentVolumeClaim:
          claimName: "{{ $name }}-data-claim"
{{- end }}
{{- end -}}
