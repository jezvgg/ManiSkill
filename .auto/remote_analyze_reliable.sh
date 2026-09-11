#!/usr/bin/env bash
# Run a ManiSkill planner across N seeds on a remote host (gangway) with W
# parallel workers, pull the run logs back, and list the failed runs.
#
# The run happens in an isolated git worktree on the remote so it never
# disturbs the remote's main checkout; everyone else's work (and the venv at
# $REMOTE_REPO/.venv) is left untouched.
#
# Usage: remote_analyze.sh <planner_module> <seed_count> <workers> [remote_host] [remote_repo]
#   planner_module  Python module under planners/, e.g. myrobocasa_fridge_veggies_planner
#   seed_count      number of seeds to run (1..N)
#   workers         how many seeds to run in parallel (1..N)
#   remote_host     ssh alias (default: gangway)
#   remote_repo     remote clone that holds the working .venv (default: ~/ManiSkill)
#
# Env: NO_VIDEO=1 appends --no-video to every remote run (skips render, ~5-10x
# faster). Use it for any batch with workers >= 10.
set -uo pipefail

PLANNER_NAME=${1:?usage: remote_analyze.sh <planner> <seed_count> <workers> [remote_host] [remote_repo]}
SEED_COUNT=${2:?usage: remote_analyze.sh <planner> <seed_count> <workers> [remote_host] [remote_repo]}
WORKERS=${3:?usage: remote_analyze.sh <planner> <seed_count> <workers> [remote_host] [remote_repo]}
REMOTE_HOST=${4:-gangway}
# Quoted default keeps the literal ~ so the REMOTE shell expands it; unquoted it
# would expand to the LOCAL home and the remote git/venv paths would not exist.
REMOTE_REPO=${5:-'~/ManiSkill'}

# Be forgiving if a caller passes <seed_count> <planner> <workers>.
if [[ ${1} =~ ^[0-9]+$ ]]; then
  SEED_COUNT=$1
  PLANNER_NAME=$2
fi

JOB="gw-$(date +%Y%m%d_%H%M%S)"
RUNDIR="~/gw-runs/${JOB}"
# Persistent cache mirror: rsync is delta-incremental into it, then each run
# dir is instantiated as hardlinks (cp -al) — no full ~800 MB transfer per run.
SRC_DIR="~/gw-runs/src"
VENV_PY="${REMOTE_REPO}/.venv/bin/python"
# NO_VIDEO=1 appends --no-video to every remote planner run (no render, much faster).
PLANNER_ARGS=""
if [[ ${NO_VIDEO:-0} == 1 ]]; then
  PLANNER_ARGS="--no-video"
  echo "==> video recording disabled (--no-video)"
fi

remote_ssh() {
  for _ in $(seq 1 6); do
    ssh -o BatchMode=yes -o ConnectTimeout=20 "${REMOTE_HOST}" "$@" && return 0
    sleep 10
  done
  return 1
}

echo "==> sync working tree -> ${REMOTE_HOST}:${RUNDIR} (workers=${WORKERS})"
THRESHOLD_START=$(date +%s)

# Isolated remote run dir (no git worktree needed: content is fully seeded
# from the cache mirror, so a plain dir is enough and much cheaper).
remote_ssh "mkdir -p ${SRC_DIR} ${RUNDIR} && rm -f ${SRC_DIR}/mani_skill/assets" ||
  {
    echo "!! could not create remote run dir"
    exit 1
  }

# Push only what the planners need: code + configs. Everything else (archives,
# videos, docs) is local-only weight -- the remote keeps its own assets in
# ~/.maniskill AND in its own clone at $REMOTE_REPO/mani_skill/assets, so the
# data dir is symlinked below instead of being transferred.
# - -z: the bastion tunnel is ~0.7 MB/s, compression pays off on text code
rsync -rltz --no-perms --no-owner --no-group \
  --exclude '/.venv' --exclude '/.git' --exclude '/.pi' --exclude '/logs' --exclude '/__pycache__' \
  --exclude '/logs_archive_old' --exclude '/videos' --exclude '/figures' --exclude '/docs' \
  --exclude '/mshab' --exclude '/examples' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '*.ipynb' --exclude 'mani_skill/assets' \
  ./ "${REMOTE_HOST}:${SRC_DIR}/" >/dev/null 2>&1 ||
  {
    echo "!! rsync to remote failed"
    exit 1
  }

# Instantiate the run dir as hardlinks from the cache: no per-run data transfer.
remote_ssh "cp -al ${SRC_DIR}/. ${RUNDIR}/ >/dev/null 2>&1 || cp -a ${SRC_DIR}/. ${RUNDIR}/ >/dev/null 2>&1" ||
  {
    echo "!! could not seed run dir from cache"
    exit 1
  }

# Scenes read assets from <repo>/mani_skill/assets; the remote clone already
# has the full data dir, so symlink it into this run dir (NFS) -- a hardlink on
# the symlink itself is rejected as cross-device, hence it is done per run dir.
remote_ssh "ln -sfn ${REMOTE_REPO}/mani_skill/assets ${RUNDIR}/mani_skill/assets" >/dev/null 2>&1 ||
  {
    echo "!! could not symlink assets into run dir"
    exit 1
  }

# Launch the workers DETACHED on the remote (short ssh): a long-lived worker
# ssh idles while seeds run (all output goes to /dev/null) and the bastion
# tunnel drops it, killing the seeds with SIGHUP. A short launch ssh plus
# short polling ssh sessions survives the flaky tunnel.
launch_cmd="cd ${RUNDIR} && setsid bash -c 'seq 1 ${SEED_COUNT} | xargs -P ${WORKERS} -I{} env PYTHONPATH=${RUNDIR} MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 ${VENV_PY} -m planners.${PLANNER_NAME} --seed {} ${PLANNER_ARGS} >/dev/null 2>&1' >/dev/null 2>&1 & echo started"
if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "${REMOTE_HOST}" "${launch_cmd}"; then
  # The detached command can start successfully even when the tunnel drops
  # before its tiny acknowledgement arrives. Check; never launch it twice.
  echo "==> launch acknowledgement lost; checking detached batch"
  launched=0
  for _ in $(seq 1 12); do
    n=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE_HOST}" \
      "pgrep -fc '^/.*/python -m planners\\.${PLANNER_NAME} ' || true" 2>/dev/null) || {
        sleep 10
        continue
      }
    if [ "${n}" -gt 0 ]; then
      launched=1
      break
    fi
    sleep 10
  done
  if [ "${launched}" -ne 1 ]; then
    echo "!! could not verify remote worker launch"
    exit 1
  fi
fi

# Poll with short SSH sessions. A failed poll is UNKNOWN, never zero; require
# three successful zero-worker observations before cleanup.
echo "==> waiting for ${SEED_COUNT} seeds (workers=${WORKERS})"
deadline=$(( $(date +%s) + 5400 ))
zero_polls=0
n=1
while [ "$(date +%s)" -lt "$deadline" ]; do
  if n=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE_HOST}" \
    "pgrep -fc '^/.*/python -m planners\\.${PLANNER_NAME} ' || true" 2>/dev/null); then
    if [ "${n}" -eq 0 ]; then
      zero_polls=$((zero_polls + 1))
      [ "${zero_polls}" -ge 3 ] && break
    else
      zero_polls=0
    fi
  else
    zero_polls=0
  fi
  sleep 15
done
if [ "${zero_polls}" -lt 3 ]; then
  echo "!! remote workers did not finish before deadline"
  exit 1
fi
echo "==> workers done"

# Alternative (only if one-ssh-per-worker is ever needed again): ssh ControlMaster
# multiplexing in ~/.ssh/config for Host gangway, so parallel sessions share one
# bastion tunnel instead of each opening its own.

echo "==> pull logs back -> ./logs/"
pulled=0
for _ in $(seq 1 12); do
  if rsync -rlt --no-perms --no-owner --no-group \
    --exclude='*.h5' --exclude='*.mp4' \
    "${REMOTE_HOST}:${RUNDIR}/logs/" ./logs/ >/dev/null 2>&1; then
    pulled=1
    break
  fi
  sleep 10
done
if [ "${pulled}" -ne 1 ]; then
  echo "!! could not pull remote logs; preserving ${RUNDIR}"
  exit 1
fi

echo "==> failed runs (result event success=false) created since start:"
# rsync -t preserves the remote mtimes, so the same newermt filter as the
# local analyze.sh selects exactly the runs this invocation produced.
find ./logs -maxdepth 2 -type f -name '*_events.jsonl' -newermt "@${THRESHOLD_START}" 2>/dev/null |
  while IFS= read -r f; do
    if grep -q '"success": *false' "$f"; then
      echo "$(dirname "$f")"
    fi
  done | sort -u

# Clean up the remote run dir (logs already pulled); the cache mirror stays
# so the next run's rsync is delta-only.
ssh -o BatchMode=yes -o ConnectTimeout=20 "${REMOTE_HOST}" "rm -rf ${RUNDIR}" >/dev/null 2>&1
echo "==> done"
