# NInfer-4090 部署与使用指南

本文是 NInfer-4090 的唯一用户手册，覆盖：GHCR 镜像部署、模型获取与校验、环境变量与服务参数全表、服务验证、HTTP API、命令行工具 ninfer、性能参考、从源码构建、常见问题与版本历史。

- 仓库：https://github.com/yorkane/ninfer-4090
- 镜像：ghcr.io/yorkane/ninfer-4090:latest，是 **public** 包，可直接 docker pull，无需登录；只有推送新镜像才需要 docker login ghcr.io
- 服务：OpenAI 兼容（/v1/chat/completions、/v1/responses）+ Anthropic 兼容（/v1/messages）
- 模型：Qwen3.6-35B-A3B 视觉版（约 21 GiB 的 .ninfer 文件），**不打进镜像**，必须挂载到容器 /models
- 运行底座：nvidia/cuda:13.1.2-runtime-ubuntu24.04；入口 /usr/local/bin/ninfer-entrypoint；二进制 /usr/local/bin/ninfer-serve（服务）与 /usr/local/bin/ninfer（CLI）
- entrypoint 内置已调优默认参数（视觉、MTP 投机解码、KV 自动容量等），全部可用环境变量覆盖；docker run 末尾追加的原生 ninfer-serve 参数原样透传，同名参数后者生效

## 1. 前置条件

| 项 | 要求 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090（sm_89）。24 GB 或 48 GB 显存均可；显存越小，需要把上下文 / KV 量化调得越激进（见第 7 节） |
| 驱动 | 需支持 CUDA 13.1（镜像运行阶段基于 nvidia/cuda:13.1.2-runtime-ubuntu24.04）。两台已实测可用的 4090 机器驱动分别为 595.58.03（CUDA 13.2）与 610.43.02，均正常工作；最低驱动版本以 NVIDIA 官方 CUDA 13.1 兼容性矩阵为准 |
| Docker | 任意较新版本 Docker Engine |
| nvidia-container-toolkit | 必需。安装后 docker info 的 Runtimes 里应能看到 nvidia |
| 磁盘 | 模型约 21 GiB（约 22.8 GB）+ 镜像（CUDA runtime 底座，几个 GiB，以 docker image ls 实际输出为准） |
| 网络 | 能访问 ghcr.io 拉镜像、能访问 huggingface.co 下载模型 |

### 安装 nvidia-container-toolkit（Ubuntu/Debian 示例）

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证 GPU 能被容器看到：

```bash
docker run --rm --gpus all nvidia/cuda:13.1.2-runtime-ubuntu24.04 nvidia-smi
```

## 2. 快速开始

假设模型在宿主机 /opt/models/qwen3_6_35b_a3b.ninfer（换成你的实际路径；下载与校验见第 3 节）：

```bash
docker pull ghcr.io/yorkane/ninfer-4090:latest

docker run -d --name ninfer \
  --gpus "device=0" \
  -v /opt/models:/models:ro \
  -p 8000:8000 \
  ghcr.io/yorkane/ninfer-4090:latest
```

说明：

- -v /opt/models:/models:ro：模型目录只读挂载到容器 /models（NINFER_MODEL 默认值就是 /models/qwen3_6_35b_a3b.ninfer）
- -p 8000:8000：容器内服务默认监听 8000
- --gpus "device=0"：entrypoint 固定使用容器内 --device 0，想跑在物理 N 号卡上就把 device=N 填进去；单卡机也可以用 --gpus all
- 本服务是单进程 C++ 服务，不依赖 NCCL，通常**不需要** --ipc=host 或 --shm-size
- 首次启动需加载约 21 GiB 权重并构建 CUDA Graph，耗时以实际为准；docker logs -f ninfer 观察，看到服务就绪 / 开始监听即完成

### 模型目录挂载

映射关系：**宿主机目录 → 容器 /models**。entrypoint 要求 NINFER_MODEL 指向容器内一个已存在的文件，找不到会直接退出并打印：

```
FATAL: model artifact not found: /models/qwen3_6_35b_a3b.ninfer
Mount the directory holding the .ninfer file, e.g.
  -v /path/to/models:/models:ro
```

三种常见情况：

1. **文件名就是 qwen3_6_35b_a3b.ninfer**：-v <宿主机目录>:/models:ro 即可，其他什么都不用做。
2. **文件名不同**（如 qwen3_6_35b_a3b_v2.ninfer）：加 -e NINFER_MODEL=/models/qwen3_6_35b_a3b_v2.ninfer。
3. **想挂到容器内其他路径**：-v /opt/models:/srv/models:ro -e NINFER_MODEL=/srv/models/qwen3_6_35b_a3b.ninfer。

挂载加 ro（只读）是推荐做法，服务不会写模型文件。

## 3. 获取并校验模型

官方 HF 仓库：https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer，文件名 qwen3_6_35b_a3b.ninfer。

> **重要：该仓库存在多个不同 revision 的模型文件，大小和 hash 都不同。**
> 撰写本文档时已知至少三个版本（以下为快照，可能随仓库更新变化，请勿当作唯一标准）：
>
> | 来源 | 字节数 | SHA256 |
> |---|---|---|
> | main 分支（artifact-manifest.json 与 SHA256SUMS 一致） | 22,790,484,480 | 3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84 |
> | 仓库自带下载脚本钉住的 revision c8b8c1c0 | 22,373,184,256 | 9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163 |
> | 一份更早 revision 的历史下载 | 22,783,246,080 | 1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2 |
>
> 正确做法：**先认准你要用的 revision，再在同一 revision 上取 SHA256SUMS 做比对**，避免"文件 hash 对不上 SUMS"的困惑。

### 下载方式一：huggingface-cli（指定 revision，推荐）

```bash
pip install -U "huggingface_hub[cli]"
# 不指定 --revision 默认拉 main；也可 --revision <commit-hash> 精确锁定
huggingface-cli download neroued/Qwen3.6-35B-A3B-NInfer qwen3_6_35b_a3b.ninfer \
  --local-dir /opt/models
# 锁 revision 示例：
# huggingface-cli download neroued/Qwen3.6-35B-A3B-NInfer qwen3_6_35b_a3b.ninfer \
#   --revision c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c --local-dir /opt/models
```

### 下载方式二：仓库自带脚本（curl 断点续传）

scripts/download-qwen36-35b-vision.sh 使用 curl -L -C - 断点续传，中断后重新运行即可继续。注意该脚本钉住了 revision c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c（校验时请对同一 revision 的 SHA256SUMS）：

```bash
# 克隆仓库后执行，可用 NINFER_MODEL_DIR 指定落盘目录
NINFER_MODEL_DIR=/opt/models bash scripts/download-qwen36-35b-vision.sh
```

### 校验：与同一 revision 的 SHA256SUMS 比对

```bash
# 1) 取下载所用同一 revision 的 SHA256SUMS（main 就填 main，否则填 commit hash）
REV=main
curl -fsSL "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/$REV/SHA256SUMS" -o /tmp/SHA256SUMS
# 2) 在模型所在目录比对（SUMS 里是相对文件名）
cd /opt/models && sha256sum -c /tmp/SHA256SUMS
# 3) 核对 artifact-manifest.json 的 bytes 字段与本地文件大小
curl -fsSL "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/$REV/artifact-manifest.json"
stat -c%s /opt/models/qwen3_6_35b_a3b.ninfer
```

> **额外提醒**：不同 revision 的 artifact 可能不兼容（引擎只认自己支持的 container_version）。hash 对上 SUMS 只说明"下载完整"，不保证"引擎能加载"——**最终以 ninfer-serve 实际启动成功为准**（见第 5 节验证）。

## 4. 环境变量与服务参数

### 4.1 容器环境变量（事实来源：entrypoint.sh）

以下默认值与 entrypoint.sh 实际内容逐字一致。

| 变量 | 默认值 | 对应参数 | 说明 / 什么时候该调 |
|---|---|---|---|
| NINFER_MODEL | /models/qwen3_6_35b_a3b.ninfer | 位置参数（artifact 路径） | 容器内模型文件路径，文件名不同必须覆盖 |
| NINFER_PORT | 8000 | --port | 容器内监听端口；改了它，-p 右侧端口要同步改 |
| NINFER_HOST | 0.0.0.0 | --host | 容器内监听地址；容器场景保持 0.0.0.0（对外可见性由 -p 控制） |
| NINFER_CONC | 8 | --max-concurrency | 最大同时生成请求数，合法范围 1..8；显存紧张或追单请求低延迟时调小，纯吞吐保持 8 |
| NINFER_MAX_CONTEXT | 262144 | --max-context | 每序列逻辑上下文上限（prompt+completion，token 数）；显存不够时优先调小，不跑超长上下文时调小可显著省显存并加快启动 |
| NINFER_KV_DTYPE | rk4v4-e8 | --kv-dtype | KV cache 量化格式：bf16/int8/rk8v4/rk4v4/rk4v4-e8/rk2v4-e8；取舍见第 7 节 |
| NINFER_VISION_MAX_TOKENS | 65536 | --vision-max-tokens | 视觉 scratchpad token 容量，约束单请求视觉 token 数（进而约束帧数）；喂超大图/长视频且显存紧张时调小 |
| NINFER_MAX_MEDIA_ITEMS | 64 | --max-media-items | 单请求图片/视频内容条目数上限；批量多图按需调大 |
| NINFER_MAX_VIDEO_PIXELS | 4294967296（2^32） | --max-video-pixels | 采样视频帧的解码像素总预算；长视频解码/解码后体积超限时调小 |
| NINFER_VIDEO_MAX_PIXELS | 67108864（64 MP） | --video-max-pixels | 采样后所有保留帧的 resize 目标像素总体积，决定每帧清晰度与帧数的取舍；设 0 表示沿用 artifact 内 video_preprocessor_config.json 的值（entrypoint 此时不传该参数）。注意它不放大视觉显存预算，要放大需同时调大 NINFER_VISION_MAX_TOKENS |
| NINFER_DRAFT_TOKENS | 3 | --draft-tokens | MTP 每轮 draft token 数（1..15）；调大提速但接受率下降，3 是较稳默认 |
| NINFER_PREFILL_CHUNK | 1024 | --prefill-chunk | prefill 分块大小（128 的倍数）；一般不用动 |
| NINFER_MODEL_ID | qwen3.6-35b-a3b | --model-id | 对外公开模型别名，请求 model 字段必须与它一致；接入要求特定 model 名的客户端时改它 |
| NINFER_API_KEY | 空（不鉴权） | --api-key | 设置后 OpenAI 风格需 Authorization: Bearer <key>，Anthropic 风格需 x-api-key；/health 免鉴权 |
| NINFER_THINKING | off | off 时追加 --no-thinking | 默认思考开关，只接受 off/on（含 false/true/0/1），非法值容器直接退出；单请求仍可用 reasoning_effort / enable_thinking 覆盖 |

### 4.2 entrypoint 写死、无环境变量对应的参数

这些由 entrypoint.sh 固定传入；需要改动时在 docker run 末尾追加原生参数覆盖（追加参数原样透传，同名参数后者生效）：

| 参数 | 值 | 含义 |
|---|---|---|
| --device | 0 | 容器内 CUDA 设备号；多卡用 --gpus "device=N" 把物理卡映射进容器 0 号 |
| --kv-capacity | auto | 权重加载后用剩余显存尽量开大 KV 池（保留 1 GiB 余量），见第 7 节 |
| --vision | 开 | 固定启用图片/视频输入 |
| --spec mtp --lm-head-draft | 开 | 固定启用 MTP 投机解码 + 专用 proposal head |
| --max-pending-requests | 64 | 准入队列深度（超出返回 429 server_overloaded） |
| --pending-timeout-ms | 900000 | 排队+准备超时 15 分钟（超时返回 503 request_queue_timeout） |

示例——显式限死 KV 容量、关 CUDA Graph：

```bash
docker run -d --name ninfer --gpus "device=0" \
  -v /opt/models:/models:ro -p 8000:8000 \
  ghcr.io/yorkane/ninfer-4090:latest \
  --kv-capacity 131072 --no-cuda-graph
```

### 4.3 ninfer-serve 原生参数全表

下表是 ninfer-serve 二进制的完整原生选项（以 --help 为准）。"默认"列是二进制的原生默认值；容器化部署时实际生效的是 entrypoint 传入的值（4.1 / 4.2 节），直接跑二进制时则按此表默认。

**服务与网络**

| 参数 | 含义 | 默认 |
|---|---|---|
| --host <H> | HTTP 监听地址 | 127.0.0.1 |
| --port <N> | HTTP 监听端口 | 8080 |
| --api-key <KEY> | Bearer token 鉴权（OpenAI Bearer / Anthropic x-api-key）；省略则不鉴权 | 未设置 |
| --cors | 浏览器 CORS 响应头 | 关 |
| --ui / --no-ui | 内嵌 Web UI（GET / 提供 SPA） | 开 |
| --model-id <ID> | 覆盖 /v1/models 的公开模型别名（不选择/不改变 artifact） | artifact 的 identity.model_id |
| --max-request-mib <N> | JSON 解析前的请求体大小上限（MiB） | 384 |
| --request-log-jsonl <FILE> | 追加全精度请求/遥测 JSONL 记录（父目录须已存在；路径不能指向模型文件；不含响应正文与 api-key 值） | 关 |
| --log-stats-interval-ms <N> | 周期性吞吐/引擎统计日志间隔；0 关闭 | 5000 |

**并发与准入**

| 参数 | 含义 | 默认 |
|---|---|---|
| --max-concurrency <N> | 最大活跃并行 decode 槽位（1..8） | 1 |
| --max-pending-requests <N> | 准入 FIFO 队列容量 | 16 |
| --pending-timeout-ms <N> | 准备+排队总超时（超时 503） | 30000 |
| --device <N> | CUDA 设备号 | 0 |

**状态会话与 Response 存储**

| 参数 | 含义 | 默认 |
|---|---|---|
| --response-store-max-records <N> | 进程内保留的 Responses 对象上限（LRU） | 1024 |
| --response-store-max-mib <N> | Response envelope/Item/context 内存预算（MiB） | 256 |

**上下文、KV 与量化**

| 参数 | 含义 | 默认 |
|---|---|---|
| --max-context <N> | 每序列逻辑上下文上限（prompt + completion，token 数） | 8192 |
| --kv-capacity <N\|auto> | 显式共享 Main Text KV 容量（token 数，向上取整到 64 token 页），或 auto：权重加载后按剩余显存最大化（保留 1 GiB sizing 余量） | 跟随 --max-context |
| --prefill-chunk <N> | 文本 prefill 分块（128 的倍数） | 1024 |
| --kv-dtype bf16\|int8\|rk8v4\|rk4v4\|rk4v4-e8\|rk2v4-e8 | KV cache 存储格式；晶格/圆柱模式（rk4v4-e8、rk2v4-e8）有损但装下更多上下文，实测上限见第 7 节 | bf16 |
| --no-cuda-graph | 禁用 CUDA Graph 解码 | graph 开 |
| --no-prefix-reuse | 禁用跨请求 KV 前缀复用 | 前缀复用开 |
| --wddm-evictable-budget | Windows 专用：对独显按总显存做激进 WDDM 预算 | 关 |

**持久化 Prompt 缓存（DirectStorage 1.3，Windows）**

| 参数 | 含义 | 默认 |
|---|---|---|
| --disk-cache / --prompt-cache | 启用 NVMe→VRAM DMA 持久化多轮 prompt 缓存（CoW 页日志） | 关 |
| --no-disk-cache | 显式禁用 | — |
| --disk-cache-dir <DIR> | prompt 缓存目录 | %LOCALAPPDATA%/ninfer/cache/<profile> |
| --disk-cache-gb <N> | 磁盘缓存配额（GiB） | 30 |

**投机解码**

| 参数 | 含义 | 默认 |
|---|---|---|
| --spec mtp\|dflash | 投机后端；dflash 仅 35B-A3B text-only artifact，且不能与 --vision 同用 | 关 |
| --draft-tokens <N> | 每轮投机 draft token 数（MTP 1..15；DFlash 1..15，7 为当前实测推荐） | 未设置 |
| --lm-head-draft | 用专用 proposal head（artifact draft 词表上的小投影）代替完整 LM head 出 draft；接受率略降、每轮显著更快，验证仍用完整 head，输出文本不变（需要 --spec） | 关 |

**视觉与多模态**

| 参数 | 含义 | 默认 |
|---|---|---|
| --vision | 启用图片/视频视觉编码器并加载 Vision GPU 分配 | 关 |
| --vision-max-tokens <N> | 视觉 scratchpad token 容量，约束单请求视觉 token 数与帧数 | 8192 |
| --max-media-items <N> | 单请求图片/视频内容条目上限 | 16 |
| --max-video-pixels <N> | 采样视频帧的解码像素总预算 | 134217728（128 MP） |
| --video-max-pixels <N> | 采样后所有保留帧的 resize 目标像素总体积；省略则沿用 artifact 的 video_preprocessor_config.json（显式 0 被拒绝，entrypoint 用"不传"实现跟随）；调大不放大视觉 scratchpad，需同时调大 --vision-max-tokens | 沿用 artifact |

**推理与生成默认**

| 参数 | 含义 | 默认 |
|---|---|---|
| --default-max-tokens <N> | 请求省略 max_tokens 时的输出上限 | 8192 |
| --no-thinking | 全局默认关闭思考 | 思考开 |
| --preserve-thinking | 多轮历史中保留已关闭回合的 assistant 推理 | 关 |
| --reasoning-effort <low\|medium\|xhigh> | 客户端省略时的默认思考深度 | 模板默认 |

**采样兜底（请求参数 > 进程参数 > 注册预设；--greedy 最后强制 temperature 0）**

| 参数 | 含义 | 默认 |
|---|---|---|
| --temperature <F> | softmax 温度兜底 | 注册预设 |
| --top-p <F> | 核采样累积概率截断兜底 | 注册预设 |
| --top-k <N> | top-K 候选数兜底 | 注册预设 |
| --min-p <F> | 相对 top token 的最小概率兜底 | 注册预设 |
| --presence-penalty <F> | presence penalty 兜底 | 注册预设 |
| --frequency-penalty <F> | frequency penalty 兜底 | 注册预设 |
| --seed <N> | 请求省略 seed 时使用的种子 | 每请求随机 |
| --greedy | 强制精确 argmax（等价 --temperature 0） | 关 |

运行 ninfer-serve --help 可查看当前构建的精确选项契约。

### 4.4 引擎注册的采样预设

请求未显式指定、进程级参数也未覆盖时，Engine 按加载模型与渲染模式选择以下预设（frequency penalty 在所有预设中均为 0）：

| 模型 | 提示模式 | temperature | top-p | top-k | min-p | presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | 1.0 | 0.95 | 20 | 0 | 0 |
| Qwen3.6-27B | non-thinking | 0.7 | 0.80 | 20 | 0 | 1.5 |
| Qwen3.8-27B | thinking | 1.0 | 0.95 | 20 | 0 | 0 |
| Qwen3.8-27B | non-thinking | 0.7 | 0.80 | 20 | 0 | 1.5 |
| Qwen3.6-35B-A3B | thinking | 1.0 | 0.95 | 20 | 0 | 1.5 |
| Qwen3.6-35B-A3B | non-thinking | 0.7 | 0.80 | 20 | 0 | 1.5 |

## 5. 验证服务

```bash
# 1) 健康检查（免鉴权；worker 故障时返回 503 {"status":"error"}）
curl -s http://127.0.0.1:8000/health

# 2) 查看对外模型别名
curl -s http://127.0.0.1:8000/v1/models

# 3) 最小纯文本对话
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "max_tokens": 128
  }'
```

带图片（支持 HTTP(S) URL 或 base64 data URL；服务固定开启 --vision）：

```bash
IMG_B64=$(base64 -w0 /path/to/cat.jpg)
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qwen3.6-35b-a3b\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMG_B64\"}},
        {\"type\": \"text\", \"text\": \"描述这张图片\"}
      ]
    }],
    \"max_tokens\": 256
  }"
```

设置了 NINFER_API_KEY 时，以上 /v1/* 请求都要加 -H "Authorization: Bearer <key>"。

## 6. 多卡部署（每卡一个容器）

entrypoint 固定 --device 0，所以每卡一容器的正确姿势是用 --gpus "device=N" 把不同物理卡映射进各自容器的 0 号，再各占一个宿主机端口：

```bash
for i in 0 1 2 3; do
  docker run -d --name "ninfer-gpu$i" \
    --gpus "device=$i" \
    -v /opt/models:/models:ro \
    -p 820$i:8000 \
    ghcr.io/yorkane/ninfer-4090:latest
done
```

- 4 张 4090 得到 8200/8201/8202/8203 四个相同服务，可再接上游网关做负载均衡
- 多容器共用同一只读模型挂载没有问题
- 某张卡需要不同参数（如 24 GB 卡调小上下文）时，单独给该容器加 -e：
  docker run -d --name ninfer-gpu1 --gpus "device=1" -v /opt/models:/models:ro -p 8201:8000 -e NINFER_MAX_CONTEXT=131072 ghcr.io/yorkane/ninfer-4090:latest

## 7. KV / 显存调参

### --kv-dtype 可选值与取舍

| 值 | 格式 | 精度（vs FP32 余弦相似度） | 显存 | 适用 |
|---|---|---|---|---|
| bf16 | 16-bit 非压缩 | 最高 | 最大 | 一般不必用，24 GB 卡上下文会很短 |
| int8 | 8-bit 通道量化 | 约 99.8% | 中 | 高精度优先、上下文需求中等 |
| rk8v4 | 8-bit 秩压缩 K + 4-bit V | 约 99.4% | 中 | 介于 int8 与 rk4v4 之间 |
| rk4v4 | 4-bit Hadamard K + 4-bit V | 约 97.8% | 小 | 容量/精度平衡 |
| rk4v4-e8 | 4-bit E8 晶格 K + 4-bit V | 约 98.7% | 小 | **entrypoint 默认**，推荐起点 |
| rk2v4-e8 | 2-bit E8 圆柱 K + 4-bit V | 约 96.2% | 最小 | 超长上下文、显存极限场景 |

### 已验证上下文上限矩阵（RTX 4090，24 GB）

下表是 24 GB 卡上二分搜索得到的物理内存上限（取整到千位；Windows 下用 --wddm-evictable-budget 测得，该标志允许 WDDM 把后台应用驱逐到 DWM 显示底线；不开该标志时，桌面显卡上 DWM 与后台应用通常占用 1–3 GiB 显存，实际可用上下文要低 30,000–60,000 token 左右）。持续满吞吐运行时，建议把 --max-context 设在上限以下 20,000–30,000 token。Direct L1 缓存的 GQA 解码块表查找支持最高 1,048,576（1M）token 的原生上下文包络。

| 配置 | 投机 | KV 模式 | 物理上限 | 余弦相似度 | 建议安全值 |
|---|---|---|---:|---:|---:|
| 纯文本 | MTP0 | rk2v4-e8 | 567,000 | 96.2% | 500,000 |
| 纯文本 | MTP0 | rk4v4-e8 | 433,000 | 98.7% | 400,000 |
| 纯文本 | MTP0 | rk4v4 | 433,000 | 97.8% | 400,000 |
| 纯文本 | MTP0 | rk8v4 | 294,000 | 99.4% | 270,000 |
| 纯文本 | MTP0 | int8 | 223,000 | 99.8% | 200,000 |
| 纯文本 | MTP4 | rk2v4-e8 | 462,000 | 96.2% | 430,000 |
| 纯文本 | MTP4 | rk4v4-e8 | 352,000 | 98.7% | 320,000 |
| 纯文本 | MTP4 | rk4v4 | 352,000 | 97.8% | 320,000 |
| 纯文本 | MTP4 | rk8v4 | 239,000 | 99.4% | 210,000 |
| 纯文本 | MTP4 | int8 | 181,000 | 99.8% | 160,000 |
| 视觉（8k 默认） | MTP4 | rk2v4-e8 | 415,000 | 96.2% | 380,000 |
| 视觉（8k 默认） | MTP4 | rk4v4-e8 | 317,000 | 98.7% | 280,000 |
| 视觉（8k 默认） | MTP4 | int8 | 163,000 | 99.8% | 140,000 |
| 视觉（4k 小） | MTP4 | rk2v4-e8 | 434,000 | 96.2% | 400,000 |
| 视觉（4k 小） | MTP4 | rk4v4-e8 | 332,000 | 98.7% | 300,000 |

### --kv-capacity auto 的含义

entrypoint 固定传入 --kv-capacity auto：权重加载完成后，用**剩余显存自动开大**共享 KV 池，保留 1 GiB sizing 余量；启动日志会报出解析出的容量与余量。想显式限死（给同卡其他东西留显存）时，在 docker run 末尾追加 --kv-capacity <token数> 覆盖。

--max-context 与 --kv-capacity 是相互独立的两个上限：前者是每序列的逻辑上限，后者决定所有活跃请求共享的 Main Text KV 池。池在启动时定死，不会按请求增长，也不在请求间均分。显式容量永远不会被静默调小；启动时会拒绝"小于一整条序列"或"给每个配置槽位都分不到一页"的池子。

### 显存不够（OOM / 启动失败）时的降配顺序

1. 调小 NINFER_MAX_CONTEXT（默认 262144，很大；不跑超长上下文就先降它，如 131072 → 65536）
2. 换更激进的 NINFER_KV_DTYPE（rk4v4-e8 → rk2v4-e8）
3. 调小 NINFER_CONC（8 → 4）
4. 视觉场景再降 NINFER_VISION_MAX_TOKENS / NINFER_MAX_VIDEO_PIXELS / NINFER_VIDEO_MAX_PIXELS

## 8. HTTP API

ninfer-serve 在一个常驻 Engine 上暴露 OpenAI 与 Anthropic 兼容端点。请求的 model 字段必须等于公开模型别名（artifact 的 identity.model_id，或被 --model-id 覆盖后的值）；--model-id 只是 HTTP 别名，不选择也不改变加载的 artifact。

### 端点一览

| 方法与路径 | 行为 |
|---|---|
| GET /health | 进程健康（免鉴权） |
| GET /v1/models | 配置的 OpenAI 模型别名 |
| GET /v1/models/{id} | 查询配置的别名 |
| POST /v1/chat/completions | OpenAI 风格 chat 生成 |
| POST /v1/responses | OpenAI Responses Core（typed Items + 语义 SSE） |
| POST /v1/responses/input_tokens | 不生成，只做 prompt token 计数 |
| GET /v1/responses/{id} | 取本地存储的终态 Response |
| DELETE /v1/responses/{id} | 删除本地 Response |
| GET /v1/responses/{id}/input_items | 列出该 Response 归一化后的输入 Items（支持 after、limit 1..100 默认 20、order asc\|desc 默认 desc） |
| POST /v1/messages | Anthropic 风格消息生成 |
| POST /v1/messages/count_tokens | 用 checkpoint 原生展开做输入 token 计数 |
| GET / | 内嵌 Web UI（SPA） |
| GET /props、GET /slots、GET /metrics | 运行时属性 / 槽位 / Prometheus 指标 |

### OpenAI Chat Completions

支持：

- system、developer、user、assistant、tool 历史；
- 字符串 content，以及有序的 text、image_url、video_url parts（媒体为 HTTP(S) URL 或 base64 data URL）；
- max_completion_tokens 与旧拼法 max_tokens；
- temperature、top_p、top_k、presence/frequency penalty、非负 seed；
- 单个 stop 字符串或数组；
- 非流式响应与 SSE 流；stream_options.include_usage；
- function tools、tool choice、assistant tool-call 历史、tool 结果消息；
- 顶层 reasoning_effort 字段与 enable_thinking 扩展；
- chat_template_kwargs.preserve_thinking 与顶层 preserve_thinking 别名。

reasoning 内容单独放在 reasoning_content 返回，正文留在 content。

思考控制细节：启动时 NInfer 从加载 artifact 内嵌的 frontend/chat_template.jinja 解析 prompt 能力（不从请求的 model 字段推断）。已注册的 effort 模板暴露 low、medium、xhigh 三档，省略 effort 用模板声明的默认值；模板未暴露的 effort（如 OpenAI 协议的 minimal、high、max）在 prompt 准备前返回 400 reasoning_effort_not_supported。Chat Completions 里 reasoning_effort: "none" 关闭思考；enable_thinking 控制同一个新回合思考开关，与 reasoning_effort 矛盾时返回 conflicting_template_option。preserve_thinking 控制已关闭回合的推理是否留在后续 prompt 中，默认取服务器设置（未加 --preserve-thinking 时为 off）；两种拼法同时出现必须取值相同；未知的非 null chat_template_kwargs 会被拒绝。

流式：先送 assistant 角色的首个 chunk，再分别发 reasoning 与 content delta，然后是 finish-reason chunk 与 [DONE]；stream_options.include_usage 为 true 时，最后一个空 choices chunk 携带完整 usage。

usage 口径：usage.prompt_tokens_details.cached_tokens 是 Engine 复用的常驻 prompt 前缀精确长度（非流式响应与流式末帧 usage chunk 都报告）；usage.prompt_tokens 包含这些缓存 token。

### OpenAI Responses Core

NInfer 实现 Responses API 的 typed-Item 与语义事件核心，不是 OpenAI 托管工具、持久云存储、后台任务、Conversations、compaction 的全量等价物。OpenAI SDK 直接改 base_url 即可用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.6-35b-a3b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK 便捷属性；线协议上是 typed output Items
```

Create 请求字段契约：

| 字段 | 契约 |
|---|---|
| model | 必填非空字符串；须等于公开模型别名 |
| input | 必填字符串或非空 typed Item 数组 |
| instructions | 可选字符串，仅对本请求插入重建会话之前 |
| previous_response_id | 可选，本地保留的 Response ID |
| max_output_tokens | ≥16 的整数；默认取 --default-max-tokens |
| stream | true 走 Responses SSE，false 走 JSON 体 |
| store | 默认 true；控制本地可检索与续接状态 |
| temperature / top_p | [0,2] / [0,1] 有限数 |
| metadata | ≤16 对；键 ≤64 字符，值 ≤512 |
| reasoning.effort | none 关闭思考；low/medium/xhigh 取模板暴露档；minimal/high/max 返回 reasoning_effort_not_supported |
| preserve_thinking（含 chat_template_kwargs 形式） | 布尔；两种拼法取值必须一致 |
| text.format | 省略或 {"type":"text"} |
| tools | 扁平 Responses function 定义（见下） |
| tool_choice | auto 或 none |
| parallel_tool_calls | 省略或 true |
| truncation | 省略或 disabled；超长输入报错而非静默丢 Items |
| top_logprobs | 省略或 0 |
| service_tier | 省略、auto 或 default；响应报告 default |
| background | 省略或 false |
| include | 省略或空数组 |
| stream_options | 省略或 {"include_obfuscation":false} |

未知顶层字段返回 unknown_parameter；识别但不支持的特性返回对应字段的 400，而不是静默忽略。

输入 Item 契约：字符串 input 归一化为一条 user message（input_text part）。数组输入接受：

| Item | 支持形态 |
|---|---|
| message | 角色 user/assistant/system/developer；字符串 content 或 typed content 数组 |
| input_text / output_text | 用户 / assistant 回放消息内的字符串 part |
| input_image | user 消息内的 image_url（HTTP(S) 或 data URI）；detail 省略或 auto；需要服务器 --vision |
| input_video | NInfer 扩展：video_url（HTTP(S) 或 data URI）；需要 --vision |
| reasoning | 空 summary + reasoning_text content parts 的回放 Item |
| function_call | 完成的 assistant 调用；call_id、name 必填，id 可选，arguments 为 JSON 对象字符串 |
| function_call_output | 完成的工具结果；call_id 与字符串 output 必填 |

相邻 function-call Items 归入同一 assistant 历史回合；reasoning Item 挂到后续 assistant 消息或 function call 上。提供的 Item ID 被保留，否则生成；重复 ID 报错。input_file、input_audio、image file_id、非 auto 的 image detail、reasoning summary、加密 reasoning、message phase 等其他类型不支持。保存在响应链里的 HTTP 媒体 URL 在续接时会重新抓取；需要不可变历史媒体字节时用 data URI。

Function tools 是扁平定义（不是 Chat Completions 的嵌套 function 对象）：type/name/description/parameters/strict。NInfer 把定义渲染进 Qwen prompt，并把模型输出解析为 function_call 输出 Items（协议 Item id 为 fc_...，call_id 为 call_...）；客户端执行后在后续请求送 function_call_output。NInfer 不执行函数、不用约束解码强制 JSON Schema，因此 strict:true、tool_choice:required、命名 tool choice、托管工具、MCP 工具与自定义自由格式工具都会被拒绝。

终态响应：object 为 response，status 为 completed / incomplete / cancelled 之一，output 为 typed 数组（reasoning Item、含 output_text 的 assistant message、若干 function_call Item）。普通模型/字符串停止是 completed；输出 token 或上下文容量耗尽是 incomplete（incomplete_details.reason: "max_output_tokens"）；SSE 已开始后接受到的错误是 response.failed；验证与准备期错误仍是普通 HTTP 错误响应。

usage 是 checkpoint 原生的：input_tokens 含 chat template 与展开媒体 token，input_tokens_details.cached_tokens 是复用的常驻前缀，output_tokens 是接受的生成 token ID 数（含被扣留的 stop token），output_tokens_details.reasoning_tokens 在 Qwen 输出解码器中计数（不靠重分词估计）。

Responses 流式：stream:true 走语义 SSE，每帧同时带 SSE 事件名与匹配的 JSON type，每个 JSON 事件有单调递增的 sequence_number。生命周期：response.created → response.in_progress → output_item.added / content_part.added → 若干 reasoning_text.delta 或 output_text.delta → 对应的 *.done → 恰有一个 response.completed / response.incomplete / response.failed。函数参数走 response.function_call_arguments.delta 与 .done；拼接后的 deltas 与终态 Item 相等。Responses SSE 不发 Chat Completions 的 [DONE]。

本地 Response 状态：store 默认 true，存储只在本进程内、受 LRU 约束，重启即失，不是 OpenAI 的持久云存储。previous_response_id 在追加新输入前重建完整的已存输入/输出 Item 历史；当前 instructions 放最前但不存入续接上下文（符合 Responses 规则）；function 定义是请求配置而非会话 Item，tool 结果回合要重新发送。store:false 的 Response 不可检索、不可作 previous_response_id；LRU 驱逐与显式删除都会使 ID 失效；单条超出存储容量时返回 response_store_capacity_exceeded。POST /v1/responses/{id}/cancel 显式失败（不支持后台执行）；POST /v1/responses/compact 返回 compaction_not_supported。GET /v1/responses/{id} 对未知 ID 返回 404 response_not_found；DELETE 返回 response.deleted（其他 Response 已保留的后代上下文仍可用）。

POST /v1/responses/input_tokens 只接受 model 与 input，执行同样的 Item/template/媒体展开但不生成：

```bash
curl -s http://127.0.0.1:8000/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","input":"Count this prompt."}'
# → {"object":"response.input_tokens","input_tokens":11}
```

不支持的 Create 字段（Conversations、prompt templates、context management、托管审核、prompt-cache 控制、safety/user 标识、Structured Outputs/JSON mode、非空 include、后台执行、compaction、files/audio、OpenAI 托管/MCP/自定义工具）是兼容性边界，不是被静默接受的占位。

### Anthropic Messages

```bash
curl -s http://127.0.0.1:8000/v1/messages \
  -H 'x-api-key: <key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

支持 system 文本、user/assistant 历史、text 与 image blocks、thinking blocks、tool-use 历史、tool 结果、客户端自定义 tools、非流式响应与 Anthropic SSE 事件。thinking.type: "disabled" 关闭思考，其他支持值开启；顶层独立布尔 preserve_thinking 控制已关闭回合历史，否则取服务器默认。output_config.effort 接受协议值 low/medium/high/xhigh/max，再按与 OpenAI 端点相同的方式对照已加载 chat template（已注册模板暴露 low/medium/xhigh）；effort 与 thinking.type: "disabled" 组合会被拒绝。Anthropic 的 model 字段只作响应标签，不选择加载的 artifact。

POST /v1/messages/count_tokens 用 artifact 的 tokenizer、chat template 与媒体展开做计数，不跑 GPU 生成。

### 鉴权与 CORS

--api-key VALUE 后，OpenAI 风格需 Authorization: Bearer <value>，Anthropic 风格需 x-api-key 头；GET /health 与 CORS 预请求保持免鉴权。--cors 打开宽松的浏览器 CORS 响应头，默认关闭。

### 错误码速查

| HTTP | 代码 | 含义 |
|---|---|---|
| 400 | vision_disabled | 媒体请求打到未开 --vision 的服务器 |
| 400 | reasoning_effort_not_supported | 请求的 effort 未暴露在已加载模板 |
| 400 | conflicting_template_option | reasoning_effort 与 enable_thinking 矛盾 |
| 400 | unknown_parameter | 未知顶层请求字段 |
| 404 | response_not_found | GET/DELETE 未知 Response ID |
| 413 | media_budget_exceeded | 视觉展开超出 --vision-max-tokens 预算（编码前拒绝） |
| 429 | server_overloaded | 活跃+排队请求达到 max_concurrency + max_pending_requests |
| 503 | request_queue_timeout | 准备+排队超过 --pending-timeout-ms 未被准入 |
| 503 | response_store_capacity_exceeded 相关 | 单条 Response 超出本地存储容量（store 语义） |

### 执行行为要点

服务器拥有一个常驻 Engine，启动时固定 1..8 个活跃生成请求槽位。每个 decode 边界把所有 decode-ready 请求压缩成一个 batch，一次模型遍历（开 graph 时一次精确 batch 的 CUDA Graph 回放）完成；请求在其单请求 prefill 结束后才加入 batch，完成/取消后下一边界重建、不留空行。

--max-pending-requests 约束活跃集之后的排队请求；总生命周期容量 = max_concurrency + max_pending_requests（含仍在 CPU/媒体准备和已完成未释放结果的请求），满容量返回 429 server_overloaded。--pending-timeout-ms 是准备开始前的绝对期限，覆盖媒体获取与 Engine FIFO 等待，超时返回 503 request_queue_timeout；没有准入 ETA，也没有无界溢出队列。

准入时预留整段 prompt+有效输出的页配额，被准入的请求能在声明边界内跑完；剩余共享页不够时，后来者按 FIFO 等待，Engine 不会先准入再截断旧请求腾位。

兼容的常驻前缀对文本与多模态历史都复用（--no-prefix-reuse 关闭）。多模态命中要求 token 类型、三轴 MRoPE 位置、编码媒体摘要、grid 与 consumer spans 全部匹配——改前面的图/视频会重置前缀而不是复用占位 token 的 KV；完全落在命中前缀内的媒体跳过 Vision 执行，新后缀媒体正常编码；完成日志用 cache= 报告复用 token 数。共享运行时区分 full_reset / append_frontier / restore_turn_checkpoint 三条路径（JSONL 完成记录以 prefix_reuse_path 暴露）；改变 reasoning effort 会改变渲染 prompt，不复用 effort 指令不同的前缀。

投机解码是引擎选项，不改变协议输出形状、stop 行为或 usage 口径；stop 截断多 token MTP/DFlash 轮次时，Engine 提交精确的已接受目标前缀，后续兼容回合仍可复用。输出上限/上下文容量结束映射为 length / max_tokens；普通模型/字符串停止映射为 stop / end_turn。

### 结构化请求日志

--request-log-jsonl FILE 以追加模式打开（父目录须已存在，路径不能指向模型 artifact，打开失败即中止启动）。每行一个 ninfer_serve_request_log schema-v8 对象，全部事件带 timestamp_unix_ms 与进程级唯一 server_instance_id。事件：server_start（artifact/Engine 身份、采样默认与进程覆盖、KV sizing 账本、CUDA Graph 观察/许可字节、GPU 环境、脱敏 argv）、request_start（协议、解析后的采样与 seed、思考模式、输出预算、流/消息/工具形状）、request_done（finish reason、prompt/completion/cache/computed-prefill token 数、前缀复用路径、未取整阶段秒数、完整投机解码计数器；timings_seconds 含 prepare/ttft/vision/prefill/decode/total）、request_error、throughput（区间 token 增量与速率、调度器占用、decode 轮批统计）。文件不含生成正文，api-key 以 <redacted> 替代。默认每 5 秒在 stderr 输出聚合统计（--log-stats-interval-ms 0 关闭）：prefill 只计区间内实际计算的 prompt 后缀 token（不含前缀命中），decode 计 decode 轮最终提交的 token，avg_decode_batch 是 decode 行轮次除以 decode 轮数，running/prefilling/decode_ready/waiting 是区间末的调度器快照，全零空闲区间省略。

## 9. 命令行工具 ninfer

镜像内的 /usr/local/bin/ninfer（或源码构建的 build/apps/ninfer）对单个 .ninfer artifact 执行一次请求。恰好传 --prompt 与 --messages 之一。答案正文流式写到 stdout；推理、模型加载（含 target 与 weights_id）、计时、吞吐、显存、投机解码统计写 stderr，所以可以独立重定向：

```bash
ninfer /models/qwen3_6_35b_a3b.ninfer \
  --prompt "Return one sentence." --max-new 64 \
  > answer.txt 2> run.log
```

### 文本与消息输入

```bash
ninfer /models/qwen3_6_35b_a3b.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 16384 --max-new 256
```

--messages 接受非空 JSON 消息数组，或含 messages 与可选 tools 数组的对象。角色支持 system、developer、user、assistant、tool；content 可为字符串或有序数组，数组 part 为：

| 内容类型 | 字段 | 可接受的来源 |
|---|---|---|
| text | text | 字符串 |
| image / image_url | image 或 image_url | 本地路径、HTTP(S) URL、base64 data URI |
| video / video_url | video 或 video_url | 本地路径、HTTP(S) URL、base64 data URI |

image_url/video_url 可以是字符串，也可以是含字符串 url 的对象。assistant 历史可带 reasoning_content 与 tool_calls；tool 结果用角色 tool 与 tool_call_id。仓库 examples/cli/ 提交了文本、图片、视频、混合媒体、thinking、长解码与长上下文的成套输入（manifest.json 列出每个用例的预期观察与 token 预算），从仓库根目录运行即可使用其仓库相对媒体路径。

### 思考、采样与投机

思考默认开启。加载 artifact 的 chat template 暴露 effort 时，--reasoning-effort low|medium|xhigh 可选（省略用模板默认；模板不暴露 effort 时该选项被拒绝）；--no-thinking 直接关闭，不能与 --reasoning-effort 同用；--greedy 独立地选精确 argmax。采样省略时取第 4.4 节注册预设；--stop-token-id / --stop / --reasoning-stop 可重复出现；--raw-output 暴露前端原始输出流，--print-token-ids 在诊断里带出 token ID。

投机解码默认关闭。MTP 取 1..15 draft 位，DFlash（35B-A3B text-only）取 1..15（7 为当前实测推荐）；--lm-head-draft 选专用 proposal head，必须先选后端。MTP 与 DFlash 不能同时启用；DFlash 与 --vision 互斥。

```bash
ninfer /models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

启动内存画像在 Engine 启动时冻结：不传 --spec 就不加载 MTP/DFlash 权重与状态；--vision 才加载视觉权重、scratch 与请求瞬态分配；DFlash 与 Vision 互斥。完整 .ninfer 清单仍会被校验，这些不是懒加载——纯文本 Engine 拒绝媒体请求，且之后无法再开视觉。

### 常用选项

| 选项 | 含义 | 默认 |
|---|---|---|
| --max-context <N> | 每序列逻辑上下文上限 | 2048 |
| --kv-capacity <N\|auto> | 显式共享 KV 容量或 auto（同 serve 语义） | 2048 |
| --prefill-chunk <N> | prefill 分块（128 的倍数） | 1024 |
| --max-new <N> | 请求的输出 token 上限 | 128 |
| --device <N> | CUDA 设备号 | 0 |
| --kv-dtype bf16\|int8\|rk8v4\|rk4v4\|rk4v4-e8\|rk2v4-e8 | KV cache 存储（同 serve，第 7 节） | bf16 |
| --spec mtp\|dflash | 投机后端 | 关 |
| --draft-tokens <N> | MTP 1..15；DFlash 1..15 | 未设置 |
| --lm-head-draft | 专用 proposal head | 关 |
| --vision | 启用图片/视频输入并加载 Vision GPU 分配 | 关 |
| --no-cuda-graph | 禁用 CUDA Graph 解码 | graph 开 |
| --no-thinking | prompt 渲染关闭思考 | 思考开 |
| --reasoning-effort low\|medium\|xhigh | 选已加载模板暴露的 effort | 模板默认 |
| --greedy | 精确 argmax 解码 | 关 |
| --temperature / --top-p / --top-k / --min-p / --presence-penalty / --frequency-penalty | 采样覆盖（省略取注册预设） | 注册预设 |
| --seed <N> | 采样种子 | 0 |

运行 ninfer --help 可查看当前构建的精确选项契约。

## 10. 性能参考

### 标准产品基准（RTX 4090 24 GB，Qwen3.8-27B，CUDA 13.3，ninfer_bench）

| 用例 | 配置 | 吞吐 | 备注 |
|---|---|---:|---|
| Prefill pp2048 | chunk 1024，INT8 KV | 2,146.3 ± 3.0 tok/s | 计算饱和 |
| Prefill pp4096 | chunk 1024，INT8 KV | 2,637.6 ± 3.3 tok/s | 深分块 prefill |
| Prefill pp512 | chunk 1024，INT8 KV | 1,971.5 ± 6.4 tok/s | 低延迟浅 prefill |
| Decode 深上下文 MTP7 | pp32768+tg128，greedy，rk4v4-e8 | 272.5 ± 0.7 tok/s | 100% 接受（8.00 tok/轮） |
| Decode 前缀缓存 MTP7 | pp2048+tg128，greedy，INT8 KV | 220.8 ± 23.5 tok/s | 88.0% 接受（7.11 tok/轮） |
| Decode 前缀缓存 MTP7 | pp2048+tg128，greedy，rk4v4-e8 | 226.0 ± 23.2 tok/s | 88.0% 接受（7.11 tok/轮） |
| Decode 冷语料 MTP4 | tg128，greedy，rk4v4-e8 | 79.5 ± 8.7 tok/s | 冷种子 28.4% 接受 |
| Decode 基线 MTP0 | tg128，无投机，INT8 KV，CUDA Graph | 51.9 ± 2.2 tok/s | 单 token 基线自回归 |
| DirectStorage 1.3 冷 DMA 恢复 | 77,615 prompt token（1.51 GiB） | 150 ms（10.1 GB/s） | 冷 TTFT 从 52.6 s 降到 1.86 s |
| 360k Needle-in-a-Haystack | 359,169 prompt token，rk2v4-e8 | 100%（5/5 命中） | 平均 prefill 666.7 tok/s |

### 并发 MTP3 解码饱和（RTX 5090，INT8 group-64 KV，--kv-capacity auto 恰解析为 C×16,384）

long_decode_aime26_15 夹具、thinking 开启、每请求 8,192 token 输出预算；只取 running=C、decode_ready=C、decode 轮全满 C 行的完整 1 秒区间。

| 模型画像 | C | 稳态 (s) | 平均 batch | 聚合 decode tok/s | MTP 接受率 | 相对 C1 加速 | 波次 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B groupwise-int | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| Qwen3.6-27B groupwise-int | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| Qwen3.6-27B groupwise-int | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| Qwen3.6-27B groupwise-int | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| Qwen3.6-27B nvfp4 | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| Qwen3.6-27B nvfp4 | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| Qwen3.6-27B nvfp4 | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| Qwen3.6-27B nvfp4 | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |
| Qwen3.6-35B-A3B groupwise-int | 1 | 12.00 | 1.00 | 593.0 | 67.2% | 1.00× | 13.75 |
| Qwen3.6-35B-A3B groupwise-int | 2 | 17.00 | 2.00 | 877.7 | 68.2% | 1.48× | 18.87 |
| Qwen3.6-35B-A3B groupwise-int | 4 | 26.01 | 4.00 | 1,166.0 | 69.8% | 1.97× | 28.43 |
| Qwen3.6-35B-A3B groupwise-int | 8 | 48.01 | 8.00 | 1,313.8 | 67.3% | 2.22× | 50.20 |

45 个请求全部达到输出上限（共 368,640 完成 token），608 个完整满批稳态区间，无请求/CUDA/OOM 失败。

### 单请求解码参考（RTX 5090，Qwen3.6-35B-A3B，MTP3）

| 场景 | decode tok/s | MTP 接受率 |
|---|---:|---:|
| 长推理（aime26 夹具，65k 输出预算） | 620–672 | 72.7–82.8% |
| 代码 | 657.6 ± 34.3 | 70.3% |
| 故事 | 456.2 ± 36.6 | 38.0% |
| 翻译 | 649.7 ± 33.0 | 67.6% |
| 结构化输出 | 770.9 ± 29.3 | 89.1% |

同工况下 DFlash block=8（k=7）与 MTP3 互有胜负（长推理 ±5% 内，代码 -14.5%，故事 -42.6%，结构化 +2.0%）；MTP3 是 Docker 镜像默认（--draft-tokens 3），代码/结构化任务优先，创意写作场景 DFlash 更差。

### 27B 推理精度参考（RTX 5090，MTP3，262,144 上下文，单样本）

| 权重画像 | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| groupwise-int | 86.67%（26/30） | 93.33%（28/30） | 86.87%（172/198） |
| nvfp4 | 93.33%（28/30） | 93.33%（28/30） | 84.34%（167/198） |

复现脚本在 tools/bench/（run_serve_corpus.py、run_serve_concurrency.py，支持 --mode mtp3 / dflash7 / mtp0、--sampling greedy）。

## 11. 从源码构建

### 构建自己的 Docker 镜像（推荐路径）

在源码仓库根目录（含 Dockerfile 与 entrypoint.sh）：

```bash
docker build -t ninfer-4090:local .
```

- 两阶段构建：nvidia/cuda:13.1.2-devel-ubuntu24.04 里 CMake（≥3.30）+ Ninja 编译 ninfer 与 ninfer-serve（CMAKE_CUDA_ARCHITECTURES=89），最终只把两个二进制拷进 nvidia/cuda:13.1.2-runtime-ubuntu24.04 运行层，**不打包模型**
- 编译耗时取决于宿主机 CPU 核数；构建机核数越多越快
- 最终镜像体积几个 GiB（CUDA runtime 底座 + 二进制），以 docker image ls 为准
- 本地构建后把第 2 节命令里的 ghcr.io/yorkane/ninfer-4090:latest 换成 ninfer-4090:local

### Windows 11 原生构建（RTX 4090，sm_89）

前置：Windows 11、CUDA Toolkit 13.x、Visual Studio 2022（MSVC x64）、CMake 与 Ninja。在 PowerShell 里初始化 MSVC x64 环境后：

```powershell
cmake -B build-ninja -G Ninja -DNINFER_BUILD_BENCHMARKS=ON
ninja -C build-ninja -j 32
```

跑完整测试套件（所有 CTest 必须零失败）：

```powershell
cd build-ninja
ctest --output-on-failure -j 8
```

只构建应用：

```powershell
ninja -C build-ninja apps/ninfer.exe apps/ninfer-serve.exe bench/ninfer_bench.exe -j 32
```

产物：build-ninja/apps/ninfer.exe（CLI）、build-ninja/apps/ninfer-serve.exe（服务）、build-ninja/bench/ninfer_bench.exe（基准）。

### RTX 3090（sm_86）构建差异

RTX 3090/3090 Ti 走 Ampere 路径：保持 CMAKE_CUDA_ARCHITECTURES=86，**不要改成 89**——4090 fork 使用 Ada 专用调度，对 3090 不适用。

Linux（Ubuntu 24.04）Docker 路径：

```bash
docker build --tag ninfer-3090:sm86 .
```

Linux 原生路径：CUDA Toolkit 12.8+，GCC 13，依赖 build-essential、cmake、ninja-build、pkg-config 及 FFmpeg/curl 开发包（libavcodec-dev、libavformat-dev、libavutil-dev、libswscale-dev、libcurl4-openssl-dev）；可选用仓库钉住的 vcpkg manifest 替代系统 FFmpeg/curl。

```bash
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
export CUDACXX=/usr/local/cuda/bin/nvcc CUDAHOSTCXX=/usr/bin/g++-13
cmake -S . -B build-sm86 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-sm86 --parallel
```

Windows 原生路径：Visual Studio 2022 + CMake + vcpkg：

```powershell
cmake -S . -B build-windows -G 'Visual Studio 17 2022' -A x64 \
  -DCMAKE_TOOLCHAIN_FILE="C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=x64-windows \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-windows --config Release --parallel
```

要点：

- 24 GB 3090 上建议 KV 容量显式指定（auto 会再预留 1 GiB 余量，可能拒掉本可行的紧凑 35B 配置）
- 35B-A3B 紧凑 artifact 不含 DFlash 权重，不要选 --spec dflash
- 3090 上 INT8 是推荐默认；rk8v4 能把 C1 自动 sizing 边界从 171,648 提到 226,560 token（MTP+Graph 关、1 GiB 余量），但未做质量等价验证，先在自己负载上验证再用
- scripts/ 下有各 Windows 脚本的 Bash 版（下载、启动、打包），NINFER_MODEL_DIR 可改落盘目录
- 成功的编译不等于性能合格；发布 Linux 实测前先记录 GPU、驱动、CUDA Toolkit、编译器、负载与结果

## 12. 常见问题

**容器启动即退出？**

```bash
docker logs --tail 200 ninfer
```

最常见输出是 FATAL: model artifact not found: ...——挂载路径或 NINFER_MODEL 没对上，按第 2 节检查。

**GPU 不可见（容器内 nvidia-smi 报错 / CUDA 初始化失败）？**

- 宿主机 nvidia-smi 是否正常？
- docker info | grep -iE 'runtimes|nvidia' 是否列出了 nvidia runtime？没有就重做第 1 节并 sudo systemctl restart docker
- docker run --rm --gpus all nvidia/cuda:13.1.2-runtime-ubuntu24.04 nvidia-smi 能否看到卡？
- --gpus "device=N" 的 N 是否超出物理卡数？换成 --gpus all 试试

**OOM / 显存不足？** 按第 7 节降配顺序处理；先看启动日志报的 KV 容量与"权重后剩余显存"，确认是 KV 池开不下还是其他分配失败。

**端口占用（"port is already allocated"）？** 换宿主机侧端口：-p 8200:8000（容器内 8000 不动，或同步改 NINFER_PORT）。多卡部署见第 6 节。

**请求被拒 429 / 503？** 429 server_overloaded = 活跃+排队请求达到 max_concurrency + max_pending_requests（容器默认 8+64）上限；503 request_queue_timeout = 排队+准备 15 分钟未被准入（通常是前面有超长 prefill 占位）。调小客户端并发，或接受默认值。

**请求报 model 不匹配？** 请求体 model 字段必须等于 NINFER_MODEL_ID（默认 qwen3.6-35b-a3b）。

**媒体请求 400 vision_disabled？** 该服务器没开 --vision。容器镜像固定开启；自己跑二进制时加 --vision。

**视觉请求 413 media_budget_exceeded？** 视觉展开超出 NINFER_VISION_MAX_TOKENS 预算，调大该值（会占更多显存）。

**拉镜像 403 / unauthorized？** 该 GHCR 包现在是 **public** 的，匿名 docker pull 即可，不需要 docker login。若仍 403，先确认 tag 拼写（ghcr.io/yorkane/ninfer-4090:latest），再检查公司网络代理/镜像仓库拦截；只有推送镜像才需要登录 GitHub 账号（CI 用 GHCR_TOKEN）。

**媒体 URL 的历史行为？** 用 store 保留的 Response 链做续接时，链里存的 HTTP 媒体 URL 会重新抓取；需要历史媒体字节不可变时改用 base64 data URI。

## 13. 版本历史

> 以下由仓库根目录的 8 份版本发布说明（v0.6.0 ~ v1.2.0）压缩合并；早期版本（v0.6.x）属于 NInfer-3090（sm_86）线，v0.7.0 起为 NInfer-4090（sm_89）线。

### v1.2.0

针对 RTX 4090（sm_89 / AD102）全面特化运行时与算子流水线：

- T=1 draft head 走 Ada Tensor Core MMA；Split-K 整数波调度（128 SM 无尾波浪费）；注意力输入 Q4/Q5 双流 fork-join；双缓冲 small-T MMA 流水线；Q5 SIMT 两行共享激活摊销（PRMT 指令 -50%、全局负载扇区 -48%）
- WDDM 驻留加固：D3D12 驻留探测失败时的崩溃安全回退、物理显存下限（专用显存 - 512 MiB DWM 底线）、1 ms 快速容量拒绝；--wddm-evictable-budget 从环境变量改为显式 CLI 标志（默认关）
- 检查点兼容：更新版 Qwen3.8-27B-NInfer 中 66 个 dflash2/* 张量走 ValidateOnly 桩消费校验（不占显存）；删除全部 Blackwell SM120/NVFP4 代码（45+ 文件），代码库严格收敛到 sm_89
- 服务：MTP 兼容的 grammar 约束结构化 JSON 输出；多轮中段 system 消息就地渲染（恢复 prompt 缓存复用）；W3C 标准最大子段 UTF-8 错误恢复（流式跨块截断字节替换 U+FFFD）；/health 改为查询 worker 真实健康（故障时 503）；瞬态准入页短缺不再锁定执行器失败；default_max_tokens 动态默认、-1 表示不限
- DirectStorage 1.3 prompt 缓存加固：32 页有界快照 staging（24 GB 卡深上下文不再 OOM）、MTP KV 恢复页跨步修正、取消时发布已完成的页写
- 构建：-rdc=false 全程序设备优化 + --split-compile=0 多线程并行 CUDA 编译

### v1.1.0

- DirectStorage 1.3 NVMe→VRAM DMA prompt 缓存（--disk-cache）：CoW 页日志、去重 64-token 页、GDN/MTP 循环状态持久化、压缩守卫与 LRU 批量驱逐
- MTP draft 窗口上限从 K=5 扩到 K=15（W ≤ 16）
- 硬件 SFU 超越函数（ex2/lg2/rcp，~40–50 周期降到 5 条指令）、E8 根码 bfi.b32 汇编加速、BF162 打包向量化
- 消除小 T GEMM 与采样内核的 DRAM 寄存器溢出（launch bounds 调 128 寄存器/线程、采样栈移入共享内存）
- GQA decode 网格对齐 128-SM 完整波（消除 34.4% 空闲尾波）
- YaRN RoPE 长上下文扩展（锚定 1,048,576 token，动态频率表更新）
- ninfer-serve 内嵌 WebUI（GET /）、/props、/slots、Prometheus /metrics、流式逐 token 计时、TCP_NODELAY + SSE 2 秒心跳
- 基准（Qwen3.8-27B，16.67 GiB）：prefill pp2048 2,093.5 tok/s；MTP7@2k 218.3 tok/s（88% 接受）；MTP7@32k 229.9 tok/s（100% 接受）；DirectStorage 冷恢复 150 ms

### v1.0.0

- D3D12 WDDM 驻留管理与后台驱逐（RESIDENCY_PRIORITY_MAXIMUM + DENY_OVERBUDGET、DXGI LUID 适配器匹配、按物理显存 - 512 MiB DWM 底线规划，可用显存解锁到 ~23.5 GiB）
- prefill 注意力因果瓦片分区（内部块无分支 + 边界块精确掩码，~76 µs → 56–60 µs）
- --vision-max-tokens 可配置（默认 8192，省 1.3–1.5 GiB 静态显存；超预算 413 media_budget_exceeded）
- 量化 GEMM 反量化用 bfe.s32、零 bank conflict staging；GDN 循环状态蝴蝶 all-reduce；2D 异步内存批处理（96 次驱动调用 → 2 次）
- 上下文扩展验证到文本 567k / 视觉+MTP4 380k token；359,169-token NIAH 100% 召回（5/5）

### v0.9.0

- E8 晶格/圆柱 KV 量化：rk4v4-e8（136 B/tok，98.67% 余弦，1M token NIAH 100% 命中）、rk2v4-e8（100 B/tok，24 GB 装 350k+ token）
- rk4v4 旋转 4-bit KV（132 B/tok，~200k 上下文）
- GQA decode 块表从共享内存静态数组改为 L1 缓存直接查找；原生上下文包络从 256k 提到 1M token
- warp 协作量化（3 步蝴蝶 shuffle，编码 2.7× 提速）、码表移 L1、32 位向量化负载
- 统一命名：int8 / rk8v4 / rk4v4 / rk4v4-e8 / rk2v4-e8

### v0.8.0

- 72 MB persisting L2 缓存钉住（MTP proposal 权重保持 L2 热态）
- 分层 N-gram prompt-lookup 投机（N=5→2，结构化代码/schema 接受率 67–75%）
- 高优先级 CUDA 流调度（降低 WDDM 下的派发/排队延迟）
- K=4 投机窗口（--draft-tokens 4，代码生成 GPU 轮次最多 -13%；110.0–110.4 tok/s）
- 多轮 checkpoint 恢复 TTFT 从 3.80 s 降到 1.79–2.05 s

### v0.7.0

首个 RTX 4090（sm_89）版本：

- 原生 Ada SASS（无 Ampere/Blackwell 兼容垫片）、W8 small-T MMA 双缓冲 cp.async.cg DMA、T=48 精确瓦片调度（128 SM）
- MTP3 线性投机解码（代码/数学 126.4–148.2 tok/s）+ ReplaySSM 循环状态事务（跨轮 949 ms TTFT 前缀复用）
- rk8v4 分页深上下文 KV（24 GB 装 140k–170k 常驻 token；134k 深度持续 prefill 1,179 tok/s、decode 92.8 tok/s）

### v0.6.1（3090 线）

- RTX 3090/3090 Ti 的 Linux 源码与 Docker 构建（Ubuntu 24.04、CUDA 13.1、GCC 13）
- 支持钉住的 vcpkg manifest；新增 Bash 下载/启动/打包脚本
- 不改变模型执行与已合格的 Windows 性能画像

### v0.6.0（3090 线）

- 注册 qwen3.8-27b/groupwise-int；ReplaySSM 降低投机循环状态显存
- 24 GB RTX 3090 上验证 C1–C4@4K 与 C8@8K（MTP3）：C8/8K 聚合 114.73 tok/s，峰值 21,818 MiB
- Qwen3.8 low/medium/xhigh reasoning effort；分页 KV 支持 BF16/INT8

## 14. 参考

- 架构与模型维护文档（英文，面向维护者）：docs/maintainer/，包括并发推理架构、分页 KV 缓存、Op 准入与契约、ReplaySSM GDN、E8 张量格式、存储布局、各 Qwen 模型/artifact 契约
- CLI 示例集：examples/cli/（文本、多模态、thinking、长解码、长上下文输入及 manifest.json）
- 基准工具：bench/、tools/bench/
- 仓库 README（英文）：构建说明与 4090 基准矩阵

---

*本文档依据仓库中 entrypoint.sh、Dockerfile、src/serve/serve_options.cpp、原 docs/ 用户文档（HTTP 服务、CLI、性能数据、三份硬件构建指南与文档索引）及根目录 8 份版本发布说明的实际内容合并编写；标注“以实际为准”处请以你机器上的真实输出为准。*
