# Runbook: worker node lost

**Symptom:** `kubectl get nodes` shows `NotReady`. Pods on that node become `Terminating` or `Unknown` after about 5 minutes.

## Drill
```bash
kubectl get pods -n rag -o wide             # note which node the app is on
oci compute instance action --action STOP --instance-id <agent-ocid>
watch kubectl get nodes,pods -n rag -o wide
```

## Expected behavior
- After about 40s the node goes `NotReady` (node-monitor-grace-period).
- After about 5 minutes the default `tolerationSeconds: 300` for `node.kubernetes.io/unreachable` expires. The pod is evicted and rescheduled on a healthy node.
- The app is unavailable from that replica for those ~5 minutes unless there are 2+ replicas. With 2+ replicas, topology spread and the PDB keep it serving.

## Diagnose
```bash
kubectl describe node <node>                # Conditions, Taints
kubectl get events -A --sort-by=.lastTimestamp | tail -30
```
On the node (via the bastion): `sudo journalctl -u rke2-agent --since -10m`.

## Fix
1. Start the instance: `oci compute instance action --action START ...`. The agent rejoins automatically.
2. If the node is unrecoverable: `kubectl drain <node> --ignore-daemonsets --delete-emptydir-data`, then `kubectl delete node <node>`, then `terraform apply -replace='oci_core_instance.agent[0]'`.

## Prevent
Run `minReplicas: 2` in production. To fail over faster, set lower `tolerationSeconds` on the Deployment.

## Observed (fill in)
- Time to NotReady:
- Time to rescheduled + Ready:
