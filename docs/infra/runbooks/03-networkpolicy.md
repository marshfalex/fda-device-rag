# Runbook: NetworkPolicy blocks traffic (502/504 from ingress)

**Symptom:** Pods are `Running` and `Ready`, but the browser gets **502/504**. ingress-nginx logs show `upstream timed out` or `connect() failed`.

## Drill
In `networkpolicy.yaml` in the manifests repo, change the allowed port from `8501` to `8080`, or change the ingress-nginx label selector to something wrong. Commit and push.

## Why it's confusing
Probes come from the **kubelet on the node**, and NetworkPolicy doesn't block those in Canal. So the pod still looks healthy while real traffic is dropped.

## Diagnose
```bash
kubectl -n rag get networkpolicy -o yaml
kubectl -n kube-system logs -l app.kubernetes.io/name=rke2-ingress-nginx --tail=50
# test from a throwaway pod inside the namespace vs outside
kubectl -n rag run t --rm -it --image=busybox -- wget -qO- -T3 rag-app/_stcore/health
kubectl -n default run t --rm -it --image=busybox -- wget -qO- -T3 rag-app.rag/_stcore/health   # should be blocked
kubectl -n kube-system get pods -l app.kubernetes.io/name=rke2-ingress-nginx --show-labels   # do labels match the policy?
```

## Fix
Revert the commit. Then confirm with `curl -I http://<app_hostname>`.

## Observed (fill in)
