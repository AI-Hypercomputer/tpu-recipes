{{/*
  Pod-template annotations for multi-host (LeaderWorkerSet) serving pods.

  Args (dict): ctx (the chart root context, i.e. `$` / `.`).

  Emits nothing when no annotation applies, so callers must guard the
  `annotations:` key on a non-empty result.

  kueue.x-k8s.io/podset-required-topology:
    The LWS `exclusive-topology` annotation pins every worker to its leader's
    node pool, but Kueue Topology-Aware Scheduling (TAS) independently stamps a
    `kubernetes.io/hostname` nodeSelector on each pod of the group. With an
    unconstrained TAS request, Kueue may pick hosts from two different TPU
    slices (node pools); the worker then ends up selecting a hostname outside
    its own pool and stays Pending forever (seen on bodaborg-tpu7x-nap whenever
    two 2x2x2 pools are provisioned together, e.g. P/D disaggregation).
    Requiring the node-pool topology level makes Kueue place the whole group
    inside one pool, consistent with `exclusive-topology`.

    Opt-in via `kueue_required_topology` (generator_specific_configs) because
    it is only valid when the ClusterQueue's ResourceFlavor is TAS-enabled and
    its Topology defines that level; on a non-TAS flavor Kueue would never
    admit the workload.
*/}}
{{- define "tpu7x.lws.podAnnotations" -}}
{{- $ctx := .ctx -}}
{{- if $ctx.Values.gcsfuse.enabled }}
gke-gcsfuse/volumes: "true"
{{- end }}
{{- if and $ctx.Values.kueue_local_queue $ctx.Values.kueue_required_topology }}
kueue.x-k8s.io/podset-required-topology: {{ $ctx.Values.kueue_required_topology | quote }}
{{- end }}
{{- end -}}
