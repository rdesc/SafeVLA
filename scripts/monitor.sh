#!/bin/bash
# SafeVLA training monitor — checks every hour, calls Claude on crash or queue-empty

SAFEVLA_DIR=/network/scratch/d/deschaer/SafeVLA
LOG_FILE=${SAFEVLA_DIR}/slurm_logs/monitor.log
JOB_NAME=safevla-train
CLAUDE_SESSION_ID="1eef80e2-c905-4e64-af0d-6aa758270c88"
WANDB_PROJECT="rdesc1-milaquebec/safety_chores"

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
    ls -t "${SAFEVLA_DIR}/slurm_logs/"slurm-*.err 2>/dev/null | head -1
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

SUMMARY_FILE=${SAFEVLA_DIR}/slurm_logs/SUMMARY.md

handle_crash() {
    local err_log="$1"
    local err_tail
    err_tail=$(tail -200 "$err_log")
    log_block "ERROR LOG TAIL (last 200 lines of ${err_log})" "$err_tail"

    call_claude "[MONITOR] '${JOB_NAME}' crashed. Error log: ${err_log}

Last 200 lines:
---
${err_tail}
---

Please:
1. Diagnose the root cause.
2. If it is a code bug (Python exception, assertion error, OOM, etc.), find and fix the relevant source file(s) in ${SAFEVLA_DIR}.
3. If it is infrastructure-related (node failure, etc.), just relaunch.
4. Relaunch: cd ${SAFEVLA_DIR} && sbatch scripts/train.sbatch
5. Report the new job ID.
6. Update ${SUMMARY_FILE} by appending a new entry with: timestamp, event type (crash), root cause diagnosis, fix applied (or 'infra issue'), new job ID, and current train.sbatch config summary."
}

handle_new_experiment() {
    local reason="$1"
    local err_log="$2"
    local last_lines=""
    [[ -n "$err_log" ]] && last_lines=$(tail -20 "$err_log")

    call_claude "[MONITOR] '${JOB_NAME}' queue is empty (reason: ${reason}).

Last lines of most recent log (${err_log}):
---
${last_lines}
---

Please:
1. Use the wandb Python API (conda run -n safevla python3) with project '${WANDB_PROJECT}' to fetch recent run metrics (success, spl, reward) for the most recent runs matching 'ObjNav_debug_GRPO_lambda_4_gpu_all_house'.
2. Based on those results and your knowledge of GRPO hyperparameter sensitivity, pick the most promising next hyperparameter config to try. Key tunable params in scripts/train.sbatch:
   --lr (current: 1e-5)
   --grpo_num_generations (current: 3)
   --grpo_lambda (current: 0.98)
   --grpo_lambda_gamma (current: 0.98)
   --grpo_advantage_clamp_min (current: -0.1)
   --gamma (current: 0.98)
3. Update scripts/train.sbatch with the new values and a new --tag reflecting the changes.
4. Launch: cd ${SAFEVLA_DIR} && sbatch scripts/train.sbatch
5. Report the new job ID and what you changed.
6. Update ${SUMMARY_FILE} by appending a new entry with: timestamp, event type (new experiment), wandb metrics that informed the decision, hyperparams changed and why, new job ID."
}

# After launching a job, wait 5 min and verify it started cleanly.
# Calls Claude again if a crash signature appears in the early log.
post_launch_check() {
    local new_job_id="$1"
    log "Post-launch check: waiting 5 min for job ${new_job_id} to start..."
    sleep 300

    log_section "POST-LAUNCH CHECK (job ${new_job_id})"
    log_squeue

    local new_err_log="${SAFEVLA_DIR}/slurm_logs/slurm-${new_job_id}.err"
    if [[ ! -f "$new_err_log" ]]; then
        new_err_log=$(latest_err_log)
    fi

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
        call_claude "[MONITOR] Job ${new_job_id} crashed within the first 5 minutes. Error log: ${new_err_log}

Last 200 lines:
---
${err_tail}
---

Please:
1. Diagnose the root cause.
2. If it is a code bug (Python exception, assertion error, OOM, etc.), find and fix the relevant source file(s) in ${SAFEVLA_DIR}.
3. If it is infrastructure-related (node failure, etc.), just relaunch.
4. Relaunch: cd ${SAFEVLA_DIR} && sbatch scripts/train.sbatch
5. Report the new job ID.
6. Update ${SUMMARY_FILE} by appending a new entry with: timestamp, event type (early crash), root cause diagnosis, fix applied (or 'infra issue'), new job ID, and current train.sbatch config summary."
    else
        log "Post-launch check passed -- job ${new_job_id} looks healthy (classification: ${kind})."
    fi
}

# Extract job ID from sbatch output (e.g. "Submitted batch job 12345")
launch_job() {
    local output
    output=$(cd "${SAFEVLA_DIR}" && sbatch scripts/train.sbatch 2>&1)
    log "sbatch output: ${output}"
    local job_id
    job_id=$(echo "$output" | grep -oE '[0-9]+$')
    if [[ -n "$job_id" ]]; then
        log "New job ID: ${job_id}"
        post_launch_check "$job_id"
    else
        log "WARNING: Could not parse job ID from sbatch output."
    fi
}

log_section "Monitor started (PID $$, checking every 60 min) -- sleeping 60s before first check"
sleep 60

while true; do
    log_section "CHECK CYCLE"
    log_squeue

    if has_active_job; then
        log "Job '${JOB_NAME}' is RUNNING or PENDING -- nothing to do."
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
                log "ACTION: Run finished normally (${kind}) -- invoking Claude to pick next hyperparams."
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
