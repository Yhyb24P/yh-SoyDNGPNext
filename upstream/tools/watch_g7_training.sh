#!/usr/bin/env bash
# Watchdog for the G7 paper-baseline training (hang/suspend resilience).
#
# The machine has intermittent GPU-CPU sync deadlocks that kill all processes
# (including setsid'd ones) and Qwen Code. After a recovery, this cron job
# relaunches the G7 run if the process is dead and it is not yet complete.
# The G7 script checkpoints every finished fold to g7_progress.jsonl, so a
# relaunch resumes from the last completed fold (no wasted epochs).
# Idempotent: flock guard, early exit when complete or the process is alive.
#
# Install (user, agent cannot write crontab):
#   */5 * * * * /home/yhshy/git_files/SoyDNGPNext/upstream/tools/watch_g7_training.sh
set +u
LOCK=/tmp/g7_training_watchdog.lock
exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0

ROOT=/home/yhshy/git_files/SoyDNGPNext/upstream
PY=/home/yhshy/miniconda3/envs/soydngp312/bin/python
G7DIR="$ROOT/results/trainer_validation/g7"
LOG="$G7DIR/g7_run.log"

# Complete when both per-trait aggregate JSONs exist.
[ -f "$G7DIR/regression_full.json" ] && [ -f "$G7DIR/classification_full.json" ] && exit 0
# User-requested pause: a PAUSE marker in the results dir stops auto-relaunch.
# (Create it to hold G7 paused; remove it to let the watchdog resume.)
[ -f "$G7DIR/PAUSE" ] && exit 0
# Already running.
pgrep -f "g7_paper_baseline.py full" >/dev/null && exit 0

touch "$LOG"
setsid nohup "$PY" -u "$ROOT/scripts/g7_paper_baseline.py" full all >> "$LOG" 2>&1 < /dev/null &
echo "$(date -Is) [watchdog] relaunched G7 (setsid, new session; resumes from g7_progress.jsonl)" >> "$LOG"
