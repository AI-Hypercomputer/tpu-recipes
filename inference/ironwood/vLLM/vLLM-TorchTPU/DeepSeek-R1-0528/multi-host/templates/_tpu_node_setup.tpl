{{/*
Node and process limits required by the TPU KV-transfer path (Raiden /
tpu_sync). Every P/D transfer pins host memory and registers DMA entries, so
with the kernel defaults (vm.max_map_count=65530, vfio dma_entry_limit=65535)
a prefill worker exhausts them after roughly a thousand long-context requests
and aborts with "mmap error: 12" / "Resource temporarily unavailable", which
restarts the engine mid-benchmark. The reference vLLM-TorchTPU recipes raise
both limits from a privileged init container and run the engine privileged
with IPC_LOCK; this mirrors that.
Disaggregated tpu7x pools enable it by default; colocated tpu7x and v6e pods
opt in with tpu_node_setup.enabled=true.
*/}}
{{- define "tpu.nodeSetup.initContainers" -}}
- name: tpu-node-setup
  image: {{ .Values.tpu_node_setup_image | default "busybox" }}
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

{{- define "tpu.nodeSetup.securityContext" -}}
privileged: true
capabilities:
  add: ["IPC_LOCK"]
{{- end -}}
