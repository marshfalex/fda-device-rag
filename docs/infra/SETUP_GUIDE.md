# Step-by-step setup guide

Each step is marked with who does it:
- **[YOU]**: needs your accounts, browser, or laptop. Nobody else can do it.
- **[AUTO]**: a script or workflow in this repo does it once you run one command.

Expected time: about 1 day for steps 0–5, half a day for 6–7, and 1–2 days for step 8.

---

## 0. Tools on your laptop [YOU]

### Windows: use WSL2 (recommended)
Every command in this guide is bash, so run them inside Ubuntu on WSL2, not in PowerShell.
In **PowerShell as Administrator**:
```powershell
wsl --install -d Ubuntu-24.04
```
Reboot, open "Ubuntu" from the Start menu, and create a Linux username and password. Everything after this runs in that Ubuntu terminal.
Clone the repo *inside* WSL (`~/fda-device-rag`), not under `/mnt/c/`. That keeps file permissions and SSH keys working.

### Install the tools (Ubuntu / WSL2 / Debian)
```bash
sudo apt-get update && sudo apt-get install -y git curl unzip python3-pip pipx gnupg software-properties-common
```
```bash
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt-get update && sudo apt-get install -y terraform
```
```bash
pipx install oci-cli && pipx ensurepath
```
```bash
curl -LO "https://dl.k8s.io/release/v1.31.4/bin/linux/amd64/kubectl" && sudo install kubectl /usr/local/bin/ && rm kubectl
```
macOS: `brew install terraform oci-cli kubectl`.

Then:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519   # skip if you already have one
```
`kubectl kustomize` is built in, so you don't need a separate kustomize install.

## 1. OCI account and API key [YOU]

1. Sign up at cloud.oracle.com (Always Free). **Choose your home region carefully.** A1 ARM capacity only exists there. `us-ashburn-1` and `us-phoenix-1` often run out of capacity; less busy regions work better.
2. Optional but recommended: upgrade to **Pay As You Go**. Always Free resources stay free, and A1 "out of capacity" errors mostly go away. Set a budget alert at $1.
3. Run `oci setup config`. It asks for your user OCID and tenancy OCID (Profile menu → *My profile* / *Tenancy*), generates an API key, and tells you to upload the public key under *My profile → API keys*.
4. Check that it works: `oci iam region list` should print regions.

## 2. Build the network and VMs with Terraform [AUTO after config]

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# edit: region, compartment_ocid (= tenancy OCID), admin_cidrs (= ["<your-ip>/32"], one per location you work from)
terraform init
terraform plan      # review: 1 VCN, 2 subnets, IGW, NAT, 2 route tables, 3 NSGs, 4 VMs, 1 LB
terraform apply
```
Save the outputs: `bastion_public_ip`, `lb_public_ip`, `app_hostname`, and `ssh_tunnel_command`.

If you get **"Out of host capacity"**, retry later, try another AD, or upgrade to PAYG. If you have to, set `agent_count = 1` and give the agent 2 OCPU and 12 GB.

Cloud-init installs RKE2 automatically. Allow about 5 minutes after apply.

## 3. Get kubectl access [YOU, 2 commands]

```bash
# copy the kubeconfig from the server via the bastion
ssh -J ubuntu@<bastion_public_ip> ubuntu@<server_private_ip> 'sudo cat /etc/rancher/rke2/rke2.yaml' > kubeconfig-oci.yaml
```
```bash
# in a separate terminal, keep this running (forwards the API to localhost:6443)
ssh -N -L 6443:<server_private_ip>:6443 ubuntu@<bastion_public_ip>
```
```bash
export KUBECONFIG=$PWD/kubeconfig-oci.yaml
kubectl get nodes -o wide     # expect 3 Ready nodes, arch arm64
```
The kubeconfig already points at `https://127.0.0.1:6443`, so it works through the tunnel unchanged.

If a node is missing, SSH to it and run `sudo tail -100 /var/log/cloud-init-output.log` and `sudo journalctl -u rke2-agent -f`.

## 4. Container registry (OCIR) [YOU]

1. Find your **tenancy namespace**: `oci os ns get`.
2. Create an **auth token**: *My profile → Auth tokens → Generate*. Copy it; it is shown only once.
3. Create a **private** repository named `fda-device-rag`: *Developer Services → Container Registry*.
4. Registry host = `<region-key>.ocir.io` (for example `iad.ocir.io` for Ashburn or `phx.ocir.io` for Phoenix).
5. Create the cluster pull secret and the app secret:
```bash
kubectl create namespace rag
```
```bash
kubectl -n rag create secret docker-registry ocir-pull \
  --docker-server=<region-key>.ocir.io \
  --docker-username='<namespace>/<your-oci-username-or-email>' \
  --docker-password='<auth-token>'
```
```bash
printf 'GROQ_API_KEY = "%s"\n' "$GROQ_API_KEY" > /tmp/secrets.toml
kubectl -n rag create secret generic rag-app-secrets --from-file=secrets.toml=/tmp/secrets.toml
rm /tmp/secrets.toml
```

## 5. GitHub secrets and branch protection [YOU, in the GitHub UI]

Go to repo **Settings → Secrets and variables → Actions**.

| Name | Type | Value |
|---|---|---|
| `OCIR_REGISTRY` | secret | `iad.ocir.io` (your region key) |
| `OCIR_NAMESPACE` | secret | output of `oci os ns get` |
| `OCIR_USERNAME` | secret | `<namespace>/<oci-username>` |
| `OCIR_AUTH_TOKEN` | secret | the auth token from step 4 |
| `ANTHROPIC_API_KEY` | secret | from console.anthropic.com (for the AI reviewer) |
| `MANIFESTS_REPO_TOKEN` | secret | fine-grained PAT, *Contents: read & write* on the manifests repo only |
| `MANIFESTS_REPO` | **variable** | `marshfalex/fda-device-rag-manifests` |

Then go to **Settings → Branches → Add rule for `master`**:
- Require a pull request, with **1 approval**. This is the human-in-the-loop gate.
- Require status checks: `test`, `secrets-scan`, `build-scan-push`.
- Do **not** make `AI review` a required check. It's advisory.

## 6. Manifests repo and Argo CD [YOU create repo, AUTO the rest]

1. Create an empty GitHub repo `fda-device-rag-manifests`, then copy this repo's `deploy/` folder into it:
```bash
git clone https://github.com/marshfalex/fda-device-rag-manifests.git ../manifests
cp -r deploy ../manifests/ && cd ../manifests
```
2. In `deploy/k8s/overlays/oci/kustomization.yaml`, replace `REGION_KEY`, `TENANCY_NAMESPACE`, and `REPLACE_LB_IP`. Commit and push.
3. Install Argo CD and register the app:
```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/v2.13.2/manifests/install.yaml
kubectl -n argocd rollout status deploy/argocd-server
kubectl apply -f deploy/argocd/application.yaml
```
If the manifests repo is private, add it with `argocd repo add` using a read-only token.

4. Open the Argo CD UI:
```bash
kubectl -n argocd port-forward svc/argocd-server 8080:443
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d
```
Then browse to https://localhost:8080 and log in as `admin`.

## 7. First deploy [AUTO]

Merge any PR to `master`. Then:
`container.yml` builds and pushes `sha-xxxxxxx` → commits the tag to the manifests repo → Argo syncs → the app comes up at `http://<app_hostname>`.

Check it:
```bash
kubectl -n rag get pods,svc,ingress,hpa
```
Metrics-server ships with RKE2, so the HPA shows real CPU numbers.

## 8. AI reviewer evaluation [AUTO, costs a few dollars of API usage]

```bash
pip install -e ".[dev,review]"
export ANTHROPIC_API_KEY=...
python eval/ai_review/run_eval.py --runs 3
```
This writes `eval/ai_review/results.json` and prints the table. Copy the numbers into `docs/ai-review/EVALUATION.md`.

To get live PR cases for the demo, create a branch, copy one case's `after.py` into `app_demo/`, and open a PR. The bot comments within about a minute.

## 9. Break it on purpose [YOU, following the runbooks]

See `docs/infra/runbooks/`. Do each drill once, take screenshots, and fill in the "observed" sections.

## 10. Tear down (optional)

```bash
cd infra/terraform && terraform destroy
```
