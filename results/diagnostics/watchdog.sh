#!/bin/bash
# LOCAL watchdog, detached to ppid 1 via setsid so it outlives the agent session.
# A harness-scoped Monitor/cron dies with the session and the pod bills on.
# Gates on PROGRESS (step counter advancing), mirrors results out, then
# terminates and VERIFIES with a pod list - never trusts the delete response.
set -u
export PATH="$HOME/.local/bin:$PATH"
D=/home/natekimball/Projects/EBP/.runpod
DEST=/home/natekimball/Projects/EBP/runpod_results
read PID IP PORT GPU PRICE < <(tr '\n' ' ' < $D/pod.txt)
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519"
STALL=2400      # 40 min with no step advance -> assume dead
CAP=14400       # 4h ceiling: run is ~2.5h at 4.5s/step; caps cost near $2
START=$(date +%s); LASTP=$START; LASTSIG=""; LOWU=0
mkdir -p "$DEST"
log(){ echo "[$(date -u +%FT%TZ)] $*" >> $D/watchdog.log; }
log "START pod=$PID $IP:$PORT $GPU \$$PRICE/hr stall=${STALL}s cap=${CAP}s"

mirror(){
  log "mirroring -> $DEST"
  rsync -az -e "ssh $SSHO -p $PORT" root@$IP:/workspace/run/ "$DEST"/ >>$D/watchdog.log 2>&1 && log "mirror ok" || log "mirror FAILED"
}
finish(){
  log "TERMINATING: $1"
  mirror
  for i in 1 2 3 4 5 6; do
    runpodctl pod delete "$PID" >>$D/watchdog.log 2>&1
    sleep 10
    if ! runpodctl pod list 2>/dev/null | grep -q "$PID"; then
      log "VERIFIED TERMINATED $PID"; echo "terminated: $1" > $D/watchdog.done; exit 0
    fi
    log "still listed; retry $i"
  done
  log "!!! COULD NOT VERIFY TERMINATION of $PID - MANUAL CHECK REQUIRED"
  echo "UNVERIFIED: $1" > $D/watchdog.done; exit 1
}
trap 'log "signal received; terminating pod before exit"; finish "watchdog signalled"' TERM INT

while :; do
  NOW=$(date +%s)
  (( NOW-START >= CAP )) && finish "hard cap ${CAP}s reached"
  OUT=$(timeout 90 ssh $SSHO -p $PORT root@$IP '
    test -f /workspace/run/DONE && echo DONE
    pgrep -f train.py >/dev/null && echo ALIVE
    grep -hoE "^Step +[0-9]+" /workspace/run/*.log 2>/dev/null | tail -1
    echo "UTIL $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)"
    tail -2 /workspace/run/progress 2>/dev/null' 2>/dev/null)
  # Utilisation guard: a live trainer pinned near 0% is paid idle time just as
  # surely as a dead one, and the progress gate alone would not catch a run that
  # crawls rather than stops (e.g. a starved streaming dataloader).
  U=$(echo "$OUT" | grep -oE "UTIL [0-9]+" | awk "{print \$2}")
  U=${U:-0}
  if [ "$U" -lt 15 ]; then
    LOWU=$((LOWU+1))
    log "LOW GPU UTIL ${U}% (${LOWU} consecutive checks)"
    (( LOWU >= 10 )) && finish "GPU idle <15% for ~20 min - paying for nothing"
  else
    [ "$LOWU" -gt 0 ] && log "util recovered ${U}%"
    LOWU=0
  fi
  echo "$OUT" | grep -q DONE && finish "experiments complete"
  SIG=$(echo "$OUT" | grep -oE "Step +[0-9]+" | tail -1)
  if [ -n "$SIG" ] && [ "$SIG" != "$LASTSIG" ]; then LASTSIG="$SIG"; LASTP=$NOW; log "progress $SIG"; fi
  if ! echo "$OUT" | grep -q ALIVE; then
    sleep 90
    O2=$(timeout 90 ssh $SSHO -p $PORT root@$IP 'test -f /workspace/run/DONE && echo DONE; pgrep -f train.py >/dev/null && echo ALIVE' 2>/dev/null)
    echo "$O2" | grep -q DONE && finish "experiments complete"
    echo "$O2" | grep -q ALIVE || finish "trainer gone with no DONE marker (crash)"
  fi
  (( NOW-LASTP >= STALL )) && finish "no progress for ${STALL}s"
  sleep 120
done
