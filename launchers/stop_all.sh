#!/bin/bash
# Ferma le run sul server: chiude le sessioni screen degli esperimenti e termina
# i processi di training (run/run_*.py). Mostra cosa sta per fermare e chiede conferma.
#
# Uso:
#   stop_all.sh              ferma tutto
#   stop_all.sh --exp exp003 ferma solo le sessioni screen il cui nome contiene "exp003"
#                            (e i processi di training che girano al loro interno)

FILTER=""
if [ "$1" = "--exp" ]; then
    [ -z "$2" ] && { echo "Uso: stop_all.sh [--exp expNNN]"; exit 1; }
    FILTER="$2"
fi

# tutti i discendenti (ricorsivo) di un PID
descendants() {
    local pid=$1 child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        echo "$child"
        descendants "$child"
    done
}

# sessioni screen: "PID.nome", eventualmente filtrate per nome esperimento
SESSIONS=$(screen -ls 2>/dev/null | sed -n 's/^\s\+\([0-9]\+\.\S\+\)\s\+.*$/\1/p')
if [ -n "$FILTER" ]; then
    SESSIONS=$(echo "$SESSIONS" | grep -- "$FILTER" || true)
fi

# processi di training da terminare
if [ -n "$FILTER" ]; then
    # solo quelli dentro le sessioni selezionate
    PIDS=""
    for S in $SESSIONS; do
        SPID=${S%%.*}
        for P in $(descendants "$SPID"); do
            if ps -o args= -p "$P" 2>/dev/null | grep -q 'run/run_.*\.py'; then
                PIDS="$PIDS $P"
            fi
        done
    done
else
    PIDS=$(pgrep -u "$USER" -f 'run/run_.*\.py' || true)
fi

if [ -z "$SESSIONS" ] && [ -z "${PIDS// /}" ]; then
    echo "Niente da fermare${FILTER:+ per '$FILTER'}."
    exit 0
fi

echo "Sto per fermare:"
for S in $SESSIONS; do
    echo "  screen  $S"
done
for P in $PIDS; do
    CMD=$(ps -o args= -p "$P" 2>/dev/null | cut -c1-100)
    [ -n "$CMD" ] && echo "  proc    $P  $CMD"
done

read -r -p "Confermi? [y/N] " ANSWER
[ "$ANSWER" = "y" ] || [ "$ANSWER" = "Y" ] || { echo "Annullato."; exit 0; }

# prima i processi di training (SIGTERM), poi le sessioni screen
for P in $PIDS; do
    kill "$P" 2>/dev/null && echo "terminato processo $P"
done
sleep 2
for P in $PIDS; do
    kill -0 "$P" 2>/dev/null && kill -9 "$P" 2>/dev/null && echo "kill -9 su processo $P"
done
for S in $SESSIONS; do
    screen -S "$S" -X quit 2>/dev/null && echo "chiusa sessione $S"
done

echo "Fatto."
