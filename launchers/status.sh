#!/bin/bash
# Panoramica di ciò che sta girando sul server:
# sessioni screen attive, processi di training (run/run_*.py), core assegnati, uptime.

echo "=== Sessioni screen ==="
if screen -ls 2>/dev/null | grep -qE '^\s+[0-9]+\.'; then
    screen -ls | sed -n 's/^\s\+\([0-9]\+\.\S\+\)\s\+.*(\(.*\))$/  \1  [\2]/p'
else
    echo "  nessuna sessione screen attiva"
fi

echo
echo "=== Processi di training (run/run_*.py) ==="
PIDS=$(pgrep -u "$USER" -f 'run/run_.*\.py' || true)
if [ -z "$PIDS" ]; then
    echo "  nessun processo di training attivo"
    exit 0
fi

printf "  %-8s %-12s %-16s %s\n" "PID" "UPTIME" "CORES" "COMANDO"
for PID in $PIDS; do
    UPTIME=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
    [ -z "$UPTIME" ] && continue   # processo terminato nel frattempo
    CORES=$(taskset -cp "$PID" 2>/dev/null | awk -F': ' '{print $2}')
    CMD=$(ps -o args= -p "$PID" | sed 's/^.*\.venv\/bin\///' | cut -c1-110)
    printf "  %-8s %-12s %-16s %s\n" "$PID" "$UPTIME" "$CORES" "$CMD"
done
