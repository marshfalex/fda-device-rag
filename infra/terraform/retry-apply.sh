#!/usr/bin/env bash
# Retries `terraform apply` until Always Free ARM capacity is available.
# Cycles the node availability domain 1 -> 2 -> 3 each round (the bastion AD
# stays whatever terraform.tfvars says). Stops on success or after MAX_ROUNDS.
#   ./retry-apply.sh            # default: 5 min between rounds, 48 rounds (~12 h)
#   SLEEP=600 MAX_ROUNDS=20 ./retry-apply.sh
set -u
SLEEP="${SLEEP:-300}"
MAX_ROUNDS="${MAX_ROUNDS:-48}"
AD_COUNT=$(oci iam availability-domain list --query 'length(data)' --raw-output 2>/dev/null || echo 3)

for round in $(seq 1 "$MAX_ROUNDS"); do
  # Once any node exists, pin to its AD: changing AD would destroy and recreate it.
  PINNED=$(terraform state list 2>/dev/null | grep -E 'oci_core_instance\.(server|agent)' | head -1)
  if [ -n "$PINNED" ]; then
    AD_NAME=$(terraform state show "$PINNED" | grep -m1 'availability_domain' | grep -oE 'AD-[0-9]+')
    ADS="${AD_NAME#AD-}"
    echo "Node already exists in $AD_NAME; staying there."
  else
    ADS=$(seq 1 "$AD_COUNT")
  fi
  for ad in $ADS; do
    echo "=== $(date '+%H:%M:%S') round $round/$MAX_ROUNDS, nodes in AD-$ad ==="
    if terraform apply -auto-approve -input=false -var "nodes_ad_number=$ad" 2>&1 | tee last-apply.log | tail -5; then
      if grep -q "Apply complete" last-apply.log; then
        echo "SUCCESS: nodes landed in AD-$ad. Set nodes_ad_number = $ad in terraform.tfvars so future applies don't move them."
        terraform output
        exit 0
      fi
    fi
    grep -E "Error:" last-apply.log | sort -u
  done
  echo "No capacity yet; sleeping ${SLEEP}s..."
  sleep "$SLEEP"
done
echo "Gave up after $MAX_ROUNDS rounds. Try again later or upgrade to Pay As You Go."
exit 1
