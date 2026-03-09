#!/bin/bash
# SafeVLA constrained-GRPO monitor — checks every hour, calls Claude on crash or queue-empty

SAFEVLA_DIR=/network/scratch/d/deschaer/SafeVLA
LOG_FILE=${SAFEVLA_DIR}/slurm_logs/monitor_constrained.log
JOB_NAME=safevla-constrained
TRAIN_SBATCH=scripts/train_constrained.sbatch
CLAUDE_SESSION_ID="1eef80e2-c905-4e64-af0d-6aa758270c88"
WANDB_PROJECT="rdesc1-milaquebec/safety_chores"
SUMMARY_FILE=${SAFEVLA_DIR}/slurm_logs/SUMMARY_constrained.md

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_section() {
    {
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  $*"
        echo "  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "════════════════════════════════════════════════════════════════"
    } | tee -a "$LOG_FILE"
}

log_block() {
    local title="$1"; shift
    {
        echo "── ${title} ──────────────────────────────────────"
        echo "$*"
        echo "──────────────────────────────────────────────────"
    } | tee -a "$LOG_FILE"
}

has_active_job() {
    squeue -u "$USER" -h -o "%j %T" 2>/dev/null \
        | grep "^${JOB_NAME} " \
        | grep -qE "(RUNNING|PENDING|COMPLETING)"
}

latest_err_log() {
    ls -t "${SAFEVLA_DIR}/slurm_logs/"slurm-constrained-*.err 2>/dev/null | head -1
}

# Returns: "crash", "timelimit", "signal", or "clean"
classify_log() {
    local f="$1"
    [[ -z "$f" ]] && echo "clean" && return

    if grep -qE "abnormally terminated|Traceback \(most recent call last\)|CUDA out of memory|Killed" "$f"; then
        echo "crash"
    elif grep -qE "DUE TO TIME LIMIT" "$f"; then
        echo "timelimit"
    elif grep -qE "DUE to SIGNAL|CANCELLED AT" "$f"; then
        echo "signal"
    else
        echo "clean"
    fi
}

log_squeue() {
    local sq
    sq=$(squeue -u "$USER" -o "%.10i %.20j %.8T %.12M %.6D %R" 2>/dev/null)
    log_block "SQUEUE (user=$USER)" "$sq"
}

call_claude() {
    local prompt="$1"
    log "Calling Claude (session ${CLAUDE_SESSION_ID})..."
    echo "" | tee -a "$LOG_FILE"
    echo "┌─ CLAUDE OUTPUT ─────────────────────────────────────────────" | tee -a "$LOG_FILE"
    unset CLAUDECODE
    claude --dangerously-skip-permissions \
        --resume "${CLAUDE_SESSION_ID}" \
        -p "$prompt" 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    echo "└─ END CLAUDE OUTPUT (exit code: ${rc}) ──────────────────────" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

handle_crash() {
    local err_log="$1"
    local err_tail
    err_tail=$(tail -200 "$err_log")
    log_block "ERROR LOG TAIL (last 200 lines of ${err_log})" "$err_tail"

    call_claude "[MONITOR-CONSTRAINED] '${JOB_NAME}' crashed. Error log: ${err_log}

Last 200 lines:
---
${err_tail}
---

Please:
1. Diagnose the root cause.
2. If it is a code bug (Python exception, assertion error, OOM, etc.), find and fix the relevant source file(s) in ${SAFEVLA_DIR}.
3. If it is infrastructure-related (node failure, etc.), just relaunch.
4. Relaunch: cd ${SAFEVLA_DIR} && sbatch ${TRAIN_SBATCH}
5. Report the new job ID.
6. Update ${SUMMARY_FILE} by appending a new entry with: timestamp, event type (crash), root cause diagnosis, fix applied (or 'infra issue'), new job ID, and current ${TRAIN_SBATCH} config summary."
}

handle_new_experiment() {
    local reason="$1"
    local err_log="$2"
    local last_lines=""
    [[ -n "$err_log" ]] && last_lines=$(tail -20 "$err_log")

    call_claude "[MONITOR-CONSTRAINED] '${JOB_NAME}' queue is empty (reason: ${reason}).

Last lines of most recent log (${err_log}):
---
${last_lines}
---

This is the constrained GRPO testing monitor. The goal is to systematically test the constrained GRPO setup with small house sets to verify correctness before scaling up. Key params in ${TRAIN_SBATCH}:
  --use_constraints True  (already set)
  --max_houses (controls house subset size: 77=~4 houses, 299=~16 houses)
  --cost_limit 2.31964
  --grpo_num_generations 3
  --lr 1e-5
  --grpo_entropy_coef 0.01

Please:
1. Use the wandb Python API (conda run -n safevla python3) with project '${WANDB_PROJECT}' to fetch recent metrics for runs matching 'ObjNav_constrained_GRPO' (success, spl, cost/constraint violation rate if available).
2. Based on results, decide the next experiment: e.g. increase max_houses (4 → 16 houses), tune cost_limit, or adjust lr if training is unstable.
3. Update ${TRAIN_SBATCH} with the new values and a new --tag reflecting the changes.
4. Launch: cd ${SAFEVLA_DIR} && sbatch ${TRAIN_SBATCH}
5. Report the new job ID and what you changed.
6. Update ${SUMMARY_FILE} by appending a new entry with: timestamp, event type (new experiment), wandb metrics that informed the decision, hyperparams changed and why, new job ID."
}

# After launching a job, wait for it to actually start, then verify it started cleanly.
post_launch_check() {
    local new_job_id="$1"
    log "Post-launch check: waiting for job ${new_job_id} to start (up to 30 min)..."

    # Poll until job transitions from PENDING to RUNNING (or disappears), max 30 min.
    local waited=0
    while [[ $waited -lt 1800 ]]; do
        local state
        state=$(squeue -u "$USER" -h -o "%i %T" 2>/dev/null | awk -v jid="$new_job_id" '$1==jid{print $2}')
        if [[ "$state" == "RUNNING" ]]; then
            log "Job ${new_job_id} is now RUNNING. Waiting 3 min for log to populate..."
            sleep 180
            break
        elif [[ -z "$state" ]]; then
            log "Job ${new_job_id} is no longer in queue (may have crashed immediately)."
            break
        fi
        sleep 60
        (( waited += 60 ))
    done
    if [[ $waited -ge 1800 ]]; then
        log "Job ${new_job_id} never left PENDING after 30 min -- skipping post-launch check."
        return
    fi

    log_section "POST-LAUNCH CHECK (job ${new_job_id})"
    log_squeue

    local new_err_log="${SAFEVLA_DIR}/slurm_logs/slurm-constrained-${new_job_id}.err"

    if [[ -z "$new_err_log" || ! -f "$new_err_log" ]]; then
        log "No err log found for job ${new_job_id} -- skipping post-launch check."
        return
    fi

    local early_tail
    early_tail=$(tail -100 "$new_err_log")
    log_block "EARLY LOG TAIL (${new_err_log})" "$early_tail"

    local kind
    kind=$(classify_log "$new_err_log")
    log "Early classification: ${kind}"

    if [[ "$kind" == "crash" ]]; then
        log "ACTION: Early crash detected -- invoking Claude to diagnose, fix, and relaunch."
        local err_tail
        err_tail=$(tail -200 "$new_err_log")
        call_claude "[MONITOR-CONSTRAINED] Job ${new_job_id} crashed within the first 5 minutes. Error log: ${new_err_log}

Last 200 lines:
---
${err_tail}
---

Please:
1. Diagnose the root cause.
2. If it is a code bug (Python exception, assertion error, OOM, etc.), find and fix the relevant source file(s) in ${SAFEVLA_DIR}.
3. If it is infrastructure-related (node failure, etc.), just relaunch.
4. Relaunch: cd ${SAFEVLA_DIR} && sbatch ${TRAIN_SBATCH}
5. Report the new job ID.
6. Update ${SUMMARY_FILE} by appending a new entry with: timestamp, event type (early crash), root cause diagnosis, fix applied (or 'infra issue'), new job ID, and current ${TRAIN_SBATCH} config summary."
    else
        log "Post-launch check passed -- job ${new_job_id} looks healthy (classification: ${kind})."
    fi
}

# Initialize summary file if missing
if [[ ! -f "$SUMMARY_FILE" ]]; then
    cat > "$SUMMARY_FILE" << 'EOF'
# SafeVLA Constrained GRPO Testing — Summary

Auto-updated by Claude monitor (`scripts/monitor_constrained.sh`). Each entry is appended after an action is taken.

---

## Goal
Systematically test the constrained GRPO setup starting with small house subsets, fix any bugs, then scale up.

## Progression plan
1. `--max_houses 77`  (~4 houses)  — verify no crashes, constraints engaged
2. `--max_houses 299` (~16 houses) — check constraint violation rate and stability
3. Full house set — if small-set runs look good

---

## Event Log

<!-- Claude appends entries below -->
EOF
fi

log_section "Constrained monitor started (PID $$, checking every 60 min) -- sleeping 60s before first check"
sleep 60

while true; do
    log_section "CHECK CYCLE"
    log_squeue

    if has_active_job; then
        log "Job '${JOB_NAME}' is RUNNING or PENDING -- checking log for silent crashes..."
        err_log=$(latest_err_log)
        if [[ -n "$err_log" && -f "$err_log" ]]; then
            kind=$(classify_log "$err_log")
            log "Running job log classification: ${kind} (${err_log})"
            if [[ "$kind" == "crash" ]]; then
                log "ACTION: Crash detected in log of running job -- invoking Claude to diagnose, fix, and relaunch."
                handle_crash "$err_log"
                new_job_id=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | grep " ${JOB_NAME}$" | awk '{print $1}' | head -1)
                if [[ -n "$new_job_id" ]]; then
                    post_launch_check "$new_job_id"
                else
                    log "No new job found in queue after Claude action -- skipping post-launch check."
                fi
            fi
        else
            log "No err log found yet for running job -- nothing to do."
        fi
    else
        log "No active '${JOB_NAME}' job found."
        err_log=$(latest_err_log)
        kind=$(classify_log "$err_log")
        log "Latest log  : ${err_log}"
        log "Classification: ${kind}"

        case "$kind" in
            crash)
                log "ACTION: Crash detected -- invoking Claude to diagnose, fix, and relaunch."
                handle_crash "$err_log"
                new_job_id=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | grep " ${JOB_NAME}$" | awk '{print $1}' | head -1)
                if [[ -n "$new_job_id" ]]; then
                    post_launch_check "$new_job_id"
                else
                    log "No new job found in queue after Claude action -- skipping post-launch check."
                fi
                ;;
            timelimit|clean)
                log "ACTION: Run finished normally (${kind}) -- invoking Claude to pick next experiment."
                handle_new_experiment "$kind" "$err_log"
                new_job_id=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | grep " ${JOB_NAME}$" | awk '{print $1}' | head -1)
                if [[ -n "$new_job_id" ]]; then
                    post_launch_check "$new_job_id"
                else
                    log "No new job found in queue after Claude action -- skipping post-launch check."
                fi
                ;;
            signal)
                log "ACTION: Job was preempted/signalled -- invoking Claude to relaunch same config."
                handle_crash "$err_log"
                new_job_id=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | grep " ${JOB_NAME}$" | awk '{print $1}' | head -1)
                if [[ -n "$new_job_id" ]]; then
                    post_launch_check "$new_job_id"
                else
                    log "No new job found in queue after Claude action -- skipping post-launch check."
                fi
                ;;
        esac
    fi

    log "Sleeping 60 min until next check..."
    sleep 3600
done
