# Runbook: pods killed every few seconds (HPA vs Argo CD replica fight)

*This happened for real during the GitOps demo, when I raised the HPA `minReplicas` from 1 to 2.*

**Symptom:** New pods go `Pending`, then `ContainerCreating`, then `Terminating` / `ContainerStatusUnknown` within about 2 seconds. A new pod replaces each one, and it repeats. The existing pod is fine. Nothing is crashing.

```
rag-app-...-7ztdg   0/1   ContainerCreating        0   0s
rag-app-...-7ztdg   0/1   Terminating              0   0s
rag-app-...-tfslj   0/1   Running                  0   1s
rag-app-...-tfslj   0/1   Terminating              0   2s
```

## Diagnose
```bash
kubectl -n rag get events --sort-by=.lastTimestamp | grep -E "Scaled|SuccessfulRescale" | tail
#  HPA: "New size: 2; reason: Current number of replicas below Spec.MinReplicas"
#  Deployment: "Scaled down replica set ... to 1"   <- someone else scaling it back
kubectl -n argocd get application rag-app -o jsonpath='{.status.operationState.syncResult.resources}' | tr ',' '\n' | grep -i deploy
```
Pods dying in seconds with exit 0, no OOM, and no probe failures point to a *controller*. They don't point to the app.

## Root cause
Two controllers owned the same field, `Deployment.spec.replicas`:
- **The HPA** set it to 2, because `minReplicas: 2`.
- **Git** said `replicas: 1`. Argo CD's `selfHeal: true` treats any difference from git as drift and "fixed" it back to 1.

Each controller undid the other every few seconds.

## Fix
Remove `replicas:` from the Deployment manifest so the HPA is the only owner of the replica count. The commit has a comment explaining why. (Another fix: tell Argo CD to ignore that field with `ignoreDifferences` on `/spec/replicas`. Removing the field is simpler and makes the ownership explicit.)

## Verify
After the fix syncs, `kubectl -n rag get pods` shows 2 stable pods with growing AGE, and the HPA shows `REPLICAS 2`.

## Lesson
With GitOps, every field needs exactly one owner. If something else in the cluster (an HPA, an operator, a mutating webhook) manages a field, either leave it out of git or tell Argo CD to ignore it.
