#!/bin/bash
# Dual-gate provisioner for the EMA run. Needs 48GB: the EMA variant holds a
# second model copy AND cannot use the chunked rollout path (compute_rollout_features
# exists only on OnlineEBPModel), so it does the unchunked 32-seq backward (24.0GB
# measured) plus 1.2GB of EMA weights -> will not fit 24GB.
set -u
export PATH="$HOME/.local/bin:$PATH"
D="/home/natekimball/Projects/EBP/.runpod"
IMG="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"   # cu128: A40 has no 13.x
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519"
MIN_BPS=3000000
# gpu|price|cloud
SKUS=("NVIDIA A40|0.35|SECURE" "NVIDIA L40|0.69|COMMUNITY" "NVIDIA L40|0.82|SECURE" "NVIDIA L40S|0.79|COMMUNITY")
rm -f "$D/pod.txt"; A=0
for E in "${SKUS[@]}"; do
 IFS='|' read -r GPU PRICE CLOUD <<< "$E"
 for TRY in 1 2; do
  A=$((A+1)); echo "=== attempt $A: $GPU \$$PRICE/hr $CLOUD $(date -u +%T) ==="
  EXTRA=""; [ "$CLOUD" = "COMMUNITY" ] && EXTRA="--public-ip"
  OUT=$(runpodctl pod create --name "ebp-ema" --image "$IMG" --gpu-id "$GPU" \
        --cloud-type "$CLOUD" --min-cuda-version 12.8 --container-disk-in-gb 30 \
        --volume-in-gb 60 --volume-mount-path /workspace --ports "22/tcp" --ssh $EXTRA 2>&1)
  PID=$(echo "$OUT" | python3 -c "import sys,json
try: print(json.load(sys.stdin).get('id',''))
except: print('')" 2>/dev/null)
  [ -z "$PID" ] && { echo "  no stock (free)"; continue; }
  echo "  pod=$PID waiting for ssh..."
  IP=""; PORT=""
  for i in $(seq 1 45); do
    S=$(runpodctl pod get "$PID" 2>/dev/null | python3 -c "import sys,json
try:
 d=json.load(sys.stdin); s=d.get('ssh') or {}
 h=s.get('host') or s.get('ip'); p=s.get('port'); print(f'{h} {p}' if h and p else '')
except: print('')" 2>/dev/null)
    [ -n "$S" ] && { IP=${S% *}; PORT=${S#* }; break; }
    sleep 20
  done
  [ -z "$IP" ] && { echo "  no ssh -> terminate"; runpodctl pod delete "$PID" >/dev/null 2>&1; sleep 5; continue; }
  echo "  ssh root@$IP -p $PORT"
  OK=""; for w in $(seq 1 12); do scp $SSHO -P "$PORT" "$D/gate.py" root@"$IP":/tmp/gate.py >/dev/null 2>&1 && { OK=1; break; }; sleep 12; done
  [ -z "$OK" ] && { echo "  scp failed -> terminate"; runpodctl pod delete "$PID" >/dev/null 2>&1; sleep 5; continue; }
  RES=$(timeout 300 ssh $SSHO -p "$PORT" root@"$IP" 'python /tmp/gate.py >/dev/null 2>&1; cat /tmp/gate_result.txt 2>/dev/null; echo; timeout 60 curl -sL -o /dev/null -w "NETBPS %{speed_download}\n" -r 0-33554432 https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/model.safetensors' 2>&1)
  echo "$RES" | grep -E "GATE_PASS|GATE_FAIL|NETBPS|Error" | sed 's/^/    /'
  BPS=$(echo "$RES" | grep -oE "NETBPS [0-9.]+" | awk '{print int($2)}'); BPS=${BPS:-0}
  VR=$(echo "$RES" | grep -oE "vram=[0-9.]+" | head -1 | cut -d= -f2 | cut -d. -f1); VR=${VR:-0}
  if echo "$RES" | grep -q GATE_PASS && [ "$BPS" -ge "$MIN_BPS" ] && [ "$VR" -ge 40 ]; then
    echo "  ALL GATES PASSED (net $((BPS/1000000)) MB/s, vram ${VR}GB)"
    printf '%s\n' "$PID" "$IP" "$PORT" "$GPU" "$PRICE" > "$D/pod.txt"; exit 0
  fi
  echo "  rejected (cuda=$(echo "$RES"|grep -c GATE_PASS) net=$((BPS/1000))kB/s vram=${VR}GB)"
  runpodctl pod delete "$PID" >/dev/null 2>&1; sleep 5
 done
done
echo "ALL SKUS EXHAUSTED"; exit 1
