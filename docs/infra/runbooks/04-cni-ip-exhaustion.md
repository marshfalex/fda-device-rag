# Runbook: pods stuck in ContainerCreating (CNI IP pool exhausted)

*This one wasn't staged. It happened during the first local RKE2 install, on Ubuntu 24.04 under WSL2.*

**Symptom:** New pods (all of Argo CD, plus `rke2-ingress-nginx`) sit in `ContainerCreating` or `Init:0/1`, and `kubectl rollout status` times out.

## Diagnose
```bash
kubectl -n argocd describe pod -l app.kubernetes.io/name=argocd-server | tail
#  FailedCreatePodSandBox ... plugin type="calico" failed (add): failed to allocate
#  for range 0: no IP addresses available in range set: 10.42.0.1-10.42.0.254
sudo ls /var/lib/cni/networks/k8s-pod-network | wc -l     # 255 reservations, but only ~20 pods exist
kubectl get pods -A | grep -v -E "Running|Completed"      # rke2-ingress-nginx stuck for 20+ min
kubectl -n kube-system describe pod -l app.kubernetes.io/name=rke2-ingress-nginx | tail
#  plugin type="portmap" failed (add): failed to open iptables:
#  exec: "iptables": executable file not found in $PATH
```

## Root cause
- ingress-nginx uses `hostPort` 80/443, so the CNI chain runs the `portmap` plugin, and `portmap` shells out to `iptables`.
- The WSL Ubuntu image ships without `iptables`.
- Every sandbox attempt went like this: host-local IPAM reserves an IP, then `portmap` fails, and the IP is never released.
- The kubelet retries every few seconds, so the /24 node pod range ran out within minutes. That starved every other new pod.

This is why clearing the IPAM state alone only *treated the symptom*. The count climbed straight back to 89 within 2 minutes.

## Fix
```bash
sudo apt-get install -y iptables                 # the real fix
sudo systemctl stop rke2-server
sudo /usr/local/bin/rke2-killall.sh
sudo rm -rf /var/lib/cni/networks/k8s-pod-network   # drop leaked reservations (rebuilt on start)
sudo systemctl start rke2-server
```

## Verify
- `kubectl get pods -A | grep -v -E "Running|Completed"` prints only the header.
- The IPAM reservation count is ~15 and **stays flat** across two checks a couple of minutes apart.

## Prevent
- Add `iptables` to the node bootstrap (cloud-init or the setup script). It's a host dependency of RKE2's CNI chain.
- Alert on `FailedCreatePodSandBox` events, and on IPAM reservations far exceeding the pod count.

## Lesson
Don't stop at the first error. "No IPs available" was a symptom. The cause was one level down, in a different plugin's error on a different pod.
