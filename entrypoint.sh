#!/bin/bash
# NInfer-4090 服务入口 / service entrypoint.
#
# 所有可调项都有环境变量默认值，也可在 docker run 末尾追加原生 ninfer-serve 参数覆盖。
# Every knob has an environment default; extra args appended to `docker run` are passed through
# verbatim to ninfer-serve, so anything not covered here still works.
set -euo pipefail

MODEL=${NINFER_MODEL:-/models/qwen3_6_35b_a3b.ninfer}
PORT=${NINFER_PORT:-8000}
HOST=${NINFER_HOST:-0.0.0.0}
CONC=${NINFER_CONC:-8}
MAXCTX=${NINFER_MAX_CONTEXT:-262144}
KVD=${NINFER_KV_DTYPE:-rk4v4-e8}
VMT=${NINFER_VISION_MAX_TOKENS:-65536}
MMI=${NINFER_MAX_MEDIA_ITEMS:-64}
MVP=${NINFER_MAX_VIDEO_PIXELS:-4294967296}
VMP=${NINFER_VIDEO_MAX_PIXELS:-67108864}
DRAFT=${NINFER_DRAFT_TOKENS:-3}
CHUNK=${NINFER_PREFILL_CHUNK:-1024}
MID=${NINFER_MODEL_ID:-qwen3.6-35b-a3b}
APIKEY=${NINFER_API_KEY:-}
THINKING=${NINFER_THINKING:-off}

THINK_ARG=()
case "$THINKING" in
  off|false|0) THINK_ARG=(--no-thinking) ;;
  on|true|1)   THINK_ARG=() ;;
  *) echo "FATAL: NINFER_THINKING must be off or on (got '$THINKING')" >&2; exit 1 ;;
esac

AUTH_ARG=()
if [ -n "$APIKEY" ]; then AUTH_ARG=(--api-key "$APIKEY"); fi

# 视频 resize 上限: 0 表示沿用模型 artifact 内的 video_preprocessor_config.json。
VMP_ARG=()
if [ "$VMP" != "0" ]; then VMP_ARG=(--video-max-pixels "$VMP"); fi

if [ ! -f "$MODEL" ]; then
  echo "FATAL: model artifact not found: $MODEL" >&2
  echo "Mount the directory holding the .ninfer file, e.g." >&2
  echo "  -v /path/to/models:/models:ro" >&2
  echo "and point NINFER_MODEL at the file inside it if the name differs." >&2
  exit 1
fi

echo "[entry] model=$MODEL host=$HOST port=$PORT conc=$CONC ctx=$MAXCTX kv=$KVD"
echo "[entry] vision_max_tokens=$VMT media_items=$MMI max_video_pixels=$MVP video_max_pixels=$VMP thinking=$THINKING"

exec /usr/local/bin/ninfer-serve "$MODEL" \
  --device 0 \
  --host "$HOST" --port "$PORT" \
  --kv-dtype "$KVD" \
  --max-context "$MAXCTX" --kv-capacity auto \
  --vision --vision-max-tokens "$VMT" \
  --max-media-items "$MMI" \
  --max-video-pixels "$MVP" \
  "${VMP_ARG[@]}" \
  --spec mtp --draft-tokens "$DRAFT" --lm-head-draft \
  --max-concurrency "$CONC" \
  --max-pending-requests 64 --pending-timeout-ms 900000 \
  --prefill-chunk "$CHUNK" \
  --model-id "$MID" \
  "${THINK_ARG[@]}" \
  "${AUTH_ARG[@]}" \
  "$@"
