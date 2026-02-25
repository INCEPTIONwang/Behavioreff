#!/usr/bin/env bash
set -euo pipefail

# ===================== 配置区 =====================
GPU_INDEX=0
THRESHOLD_MIB=2000
INTERVAL_SEC=300
RUN_ONCE=1
COOLDOWN_SEC=10
LOG_FILE="/tmp/gpu_watch_${GPU_INDEX}.log"

# 你要执行的命令
TARGET_CMD=(bash "/mnt/43t/wxh/RLinf/examples/embodiment/run_embodiment.sh" "libero_spatial_grpo_gr00t")
# ==================================================

log() {
    local msg="[$(date '+%F %T')] $*"
    echo "$msg"
    if [[ -n "${LOG_FILE:-}" ]]; then
        echo "$msg" >> "$LOG_FILE"
    fi
}

SCRIPT_FILE="${TARGET_CMD[1]}"
if [[ ! -f "$SCRIPT_FILE" ]]; then
    log "ERROR: 脚本不存在: $SCRIPT_FILE"
    exit 1
fi
if [[ ! -x "$SCRIPT_FILE" ]]; then
    log "ERROR: 脚本无权限: chmod +x $SCRIPT_FILE"
    exit 1
fi
if ! command -v nvidia-smi &>/dev/null; then
    log "ERROR: nvidia-smi 不存在"
    exit 1
fi

LOCK="/tmp/gpu_watch_${GPU_INDEX}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    log "已有监控在运行，退出"
    exit 0
fi

get_used_mib() {
    timeout 5s nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

last_run_ts=0
log "启动监控: GPU=$GPU_INDEX 阈值=$THRESHOLD_MIB MiB 间隔=$INTERVAL_SEC s 命令=${TARGET_CMD[*]}"

while true; do
    used=$(get_used_mib || true)
    if [[ -z "$used" || ! "$used" =~ ^[0-9]+$ ]]; then
        log "WARN: 获取显存失败，等待重试"
        sleep "$INTERVAL_SEC"
        continue
    fi

    log "轮询: used=${used}MiB 阈值=${THRESHOLD_MIB}MiB"

    if (( used < THRESHOLD_MIB )); then
        if (( RUN_ONCE == 1 )); then
            log "触发执行，运行一次后退出"
            "${TARGET_CMD[@]}" >> "${LOG_FILE:-/dev/null}" 2>&1
            exit 0
        else
            now=$(date +%s)
            if (( now - last_run_ts >= COOLDOWN_SEC )); then
                log "触发执行"
                "${TARGET_CMD[@]}" >> "${LOG_FILE:-/dev/null}" 2>&1
                last_run_ts=$now
            else
                log "冷却中，跳过"
            fi
        fi
    fi

    sleep "$INTERVAL_SEC"
done
