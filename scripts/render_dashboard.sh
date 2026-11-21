#!/usr/bin/env bash
# Render the Grafana dashboard (or one panel) to PNG for a past time range.
#   scripts/render_dashboard.sh FROM_UNIX TO_UNIX OUT.png [PANEL_ID] [WIDTH] [HEIGHT]
# Needs the stack running with deploy/compose.render.yaml.
set -euo pipefail
from=$1 to=$2 out=$3 panel=${4:-} width=${5:-1600} height=${6:-1900}
base=${GRAFANA_URL:-http://127.0.0.1:28300}
auth=${GRAFANA_AUTH:-admin:${GRAFANA_ADMIN_PASSWORD:-admin}}
q="orgId=1&from=$((from * 1000))&to=$((to * 1000))&width=$width&height=$height&theme=dark&tz=UTC&kiosk"
if [ -n "$panel" ]; then
  url="$base/render/d-solo/inferobs-llm-serving/x?$q&panelId=$panel"
else
  url="$base/render/d/inferobs-llm-serving/x?$q"
fi
mkdir -p "$(dirname "$out")"
curl -fsS -u "$auth" --max-time 180 "$url" -o "$out"
echo "wrote $out"
