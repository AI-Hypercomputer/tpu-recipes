{{/*
  Helpers for the serving pods' /data Hyperdisk volume.

  The claim MUST be declared at the LeaderWorkerSet controller level
  (`leaderWorkerTemplate.volumeClaimTemplates`) and NEVER as an inline pod
  Generic Ephemeral Volume (`ephemeral.volumeClaimTemplate`) inside
  `leaderTemplate.spec.volumes` / `workerTemplate.spec.volumes`.

  Why (b/565473329): Kueue copies the pod template into the
  ProvisioningRequest PodTemplate, and GKE Cluster Autoscaler evaluates the
  VolumeBinding predicate against *virtual* pods that never exist in the API
  server. ephemeral-volume-controller therefore never creates the
  `<virtual-pod>-data-volume` PVC those pods reference, VolumeBinding fails on
  every scale-up cycle, and the LeaderWorkerSet stays SchedulingGated until the
  run times out.

  Declaring the claim on the controller instead lets the pod templates carry a
  plain `emptyDir` placeholder, which Cluster Autoscaler can schedule, while
  the LWS/StatefulSet controller substitutes the real PVC on the pods it
  actually creates ("a claim in this list takes precedence over any volumes in
  the template, with the same name").
*/}}

{{/*
  Controller-level claim template for /data.
  Args (dict): releaseName (string), size (quantity, e.g. "100Gi").
*/}}
{{- define "tpu7x.dataVolumeClaimTemplates" -}}
volumeClaimTemplates:
- metadata:
    name: data-volume
  spec:
    accessModes: [ "ReadWriteOnce" ]
    storageClassName: "{{ .releaseName }}-hyperdisk-balanced-tpu"
    resources:
      requests:
        storage: {{ .size }}
# This is what actually reclaims the Hyperdisks (b/539580630).
#
# helm uninstall does not delete volumeClaimTemplate PVCs, and
# helm_recipe_runner._delete_release_pvcs cannot reap these either: the LWS
# controller strips metadata.labels when it copies volumeClaimTemplates into
# the underlying StatefulSets, so the app.kubernetes.io/instance selector the
# runner uses never matches. Verified on bodaborg-tpu7x-nap: the LWS requested
# {labels: {app.kubernetes.io/instance: ...}} and the StatefulSet received
# {name: data-volume, namespace: ...} only.
#
# The retention policy does not rely on labels. It makes the StatefulSet the
# PVC's owner (blockOwnerDeletion=true), so Kubernetes garbage-collects the
# PVC, and the Hyperdisk behind it, when the StatefulSet goes away.
persistentVolumeClaimRetentionPolicy:
  whenDeleted: Delete
  whenScaled: Delete
{{- end -}}

{{/*
  Pod-level placeholder for /data. Substituted by the real PVC whenever
  "tpu7x.dataVolumeClaimTemplates" is emitted; used as-is (tmpfs) otherwise.
  Takes no arguments.
*/}}
{{- define "tpu7x.dataVolumePlaceholder" -}}
# Placeholder only. Do NOT replace this with an inline
# `ephemeral.volumeClaimTemplate`: see "tpu7x.dataVolumeClaimTemplates" above
# and b/565473329.
- emptyDir:
    medium: Memory
  name: data-volume
{{- end -}}
