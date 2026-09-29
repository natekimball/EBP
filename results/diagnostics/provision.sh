#!/bin/bash
# Gated retry provisioner.
# Gate 1: CUDA init + real forward/backward. A pod reaches RUNNING with SSH up
#         and still fails cuda_init on ~3 of 4 community hosts.
# Gate 2: network throughput. One host passed CUDA but pulled 9 kB/s from HF,
#         so it could not download the model or stream data. Equally useless,
#         bills the same, and invisible to gate 1.
set -u
export PATH="$HOME/.local/bin:$PATH"
D="/home/natekimball/Projects/EBP/.runpod"
IMG="runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster"
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519"
MIN_BPS=3000000   # 3 MB/s floor
SKUS=("NVIDIA RTX A5000|0.16" "NVIDIA GeForce RTX 3090|0.22" "NVIDIA GeForce RTX 3090 Ti|0.27" \
      "NVIDIA RTX A6000|0.33" "NVIDIA GeForce RTX 4090|0.34")
rm -f "$D/pod.txt"
A=0
for ENTRY in "${SKUS[@]}"; do
 GPU="${ENTRY%|*}"; PRICE="${ENTRY#*|}"
 for TRY in 1 2; do
  A=$((A+1)); echo "=== attempt $A: $GPU \$$PRICE/hr (try $TRY) $(date -u +%T) ==="
  OUT=$(runpodctl pod create --name "ebp-run" --image "$IMG" --gpu-id "$GPU" \
        --cloud-type COMMUNITY --min-cuda-version 13.0 --container-disk-in-gb 30 \
        --volume-in-gb 60 --volume-mount-path /workspace --ports "22/tcp" --public-ip --ssh 2>&1)
  PID=$(echo "$OUT" | python3 -c "import sys,json
try: print(json.load(sys.stdin).get('id',''))
except: print('')" 2>/dev/null)
  [ -z "$PID" ] && { echo "  no stock (free)"; continue; }
  echo "  pod=$PID  waiting for ssh..."
  IP=""; PORT=""
  for i in $(seq 1 40); do
    S=$(runpodctl pod get "$PID" 2>/dev/null | python3 -c "import sys,json
try: d=json.load(sys.stdin); s=d.get('ssh') or {}; h=s.get('host') or s.get('ip'); p=s.get('port'); print(f'{h} {p}' if h and p else '')
except: print('')" 2>/dev/null)
    [ -n "$S" ] && { IP=${S% *}; PORT=${S#* }; break; }
    sleep 20
  done
  if [ -z "$IP" ]; then echo "  ssh never appeared -> terminate"; runpodctl pod delete "$PID" >/dev/null 2>&1; sleep 5; continue; fi
  echo "  ssh root@$IP -p $PORT"
  OK=""; for w in $(seq 1 10); do scp $SSHO -P "$PORT" "$D/gate.py" root@"$IP":/tmp/gate.py >/dev/null 2>&1 && { OK=1; break; }; sleep 12; done
  if [ -z "$OK" ]; then echo "  scp failed -> terminate"; runpodctl pod delete "$PID" >/dev/null 2>&1; sleep 5; continue; fi
  RES=$(timeout 300 ssh $SSHO -p "$PORT" root@"$IP" 'python /tmp/gate.py >/dev/null 2>&1; cat /tmp/gate_result.txt 2>/dev/null; echo; timeout 60 curl -sL -o /dev/null -w "NETBPS %{speed_download}\n" -r 0-33554432 https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/model.safetensors' 2>&1)
  echo "$RES" | grep -E "GATE_PASS|GATE_FAIL|NETBPS|RuntimeError|AssertionError" | sed 's/^/    /'
  BPS=$(echo "$RES" | grep -oE "NETBPS [0-9.]+" | awk '{print int($2)}')
  BPS=${BPS:-0}
  if echo "$RES" | grep -q GATE_PASS && [ "$BPS" -ge "$MIN_BPS" ]; then
    echo "  BOTH GATES PASSED (net $((BPS/1000000)) MB/s)"
    printf '%s\n' "$PID" "$IP" "$PORT" "$GPU" "$PRICE" > "$D/pod.txt"; exit 0
  fi
  if echo "$RES" | grep -q GATE_PASS; then echo "  CUDA ok but network only $((BPS/1000)) kB/s -> reject"; else echo "  CUDA gate failed -> reject"; fi
  runpodctl pod delete "$PID" >/dev/null 2>&1; sleep 5
 done
done
echo "ALL SKUS EXHAUSTED"; exit 1
