#!/usr/bin/env bash
# Delete old vesper droplet snapshots, keeping the newest KEEP (default 1).
# DO bills snapshot storage per GB-month whether or not a droplet exists, and
# these images are large (Isaac's docker layers alone are ~36 GB), so stale
# snapshots are a standing charge. Restore time also scales with image size.
#
#   infra/do/prune_snapshots.sh          # keep newest 1
#   KEEP=3 infra/do/prune_snapshots.sh   # keep newest 3
#   DRY_RUN=1 infra/do/prune_snapshots.sh
source "$(dirname "$0")/env.sh"
KEEP="${KEEP:-1}"

SNAPS=$(api GET "/snapshots?resource_type=droplet&per_page=200" \
  | jq -c '[.snapshots[] | select(.name|startswith("vesper"))] | sort_by(.created_at) | reverse')
N=$(echo "$SNAPS" | jq 'length')
TOTAL=$(echo "$SNAPS" | jq '[.[].size_gigabytes] | add // 0')
printf 'found %s vesper snapshots, %.1f GB total (~$%.2f/mo at $0.06/GB)\n' \
  "$N" "$TOTAL" "$(echo "$TOTAL * 0.06" | bc -l)"

if [ "$N" -le "$KEEP" ]; then echo "nothing to prune (keeping $KEEP)"; exit 0; fi

echo "$SNAPS" | jq -r ".[:$KEEP][] | \"  KEEP  \(.name)  \(.size_gigabytes) GB\""
FREED=$(echo "$SNAPS" | jq "[.[$KEEP:][].size_gigabytes] | add // 0")
for row in $(echo "$SNAPS" | jq -r ".[$KEEP:][] | \"\(.id)|\(.name)|\(.size_gigabytes)\""); do
  ID="${row%%|*}"; REST="${row#*|}"; NAME="${REST%%|*}"; GB="${REST##*|}"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "  would delete  $NAME  $GB GB"
  else
    api DELETE "/snapshots/$ID" >/dev/null && echo "  deleted  $NAME  $GB GB"
  fi
done
printf 'freed %.1f GB (~$%.2f/mo)\n' "$FREED" "$(echo "$FREED * 0.06" | bc -l)"
