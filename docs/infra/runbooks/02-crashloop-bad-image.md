# Runbook: bad image / CrashLoopBackOff

**Symptom:** The Argo app is `Degraded`. Pods are in `ImagePullBackOff` or `CrashLoopBackOff`.

## Drill A: tag that doesn't exist
In the manifests repo, set `newTag: does-not-exist`, then commit and push.

## Drill B: crash on start
In the manifests repo, add `command: ["python", "-c", "import sys; sys.exit(1)"]` to the container. Commit and push.

## Expected behavior
`maxUnavailable: 0` means the **old ReplicaSet keeps serving**. The new pods never pass readiness, so traffic never shifts to them. This is the main payoff of readiness probes plus a rolling update.

## Diagnose
```bash
kubectl -n rag get pods
kubectl -n rag describe pod <new-pod>        # Events: "Failed to pull image" / "Back-off restarting"
kubectl -n rag logs <new-pod> --previous     # crash output from the last attempt
kubectl -n rag rollout status deploy/rag-app
```
`ImagePullBackOff` with "unauthorized" means the `ocir-pull` secret is wrong or expired.
"manifest unknown" means the tag doesn't exist.
"no matching manifest for linux/arm64" means the image was built single-arch.

## Fix (GitOps way)
```bash
cd manifests && git revert HEAD && git push    # Argo syncs back to the last good tag
```
Don't `kubectl edit` the Deployment. Self-heal reverts it, and that's by design.

## Observed (fill in)
