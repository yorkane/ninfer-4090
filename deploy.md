# NInfer-4090 Docker 部署指南

本指南说明如何用 **GHCR 镜像 + 挂载本地模型目录** 的方式在 NVIDIA RTX 4090（sm_89）上启动 ninfer-4090 推理服务。

- 镜像：`ghcr.io/yorkane/ninfer-4090:latest`
- 服务：OpenAI 兼容（`/v1/chat/completions`、`/v1/responses`）+ Anthropic 兼容（`/v1/messages`）
- 模型：Qwen3.6-35B-A3B 视觉版（约 21 GiB 的 `.ninfer` 文件），**不打进镜像**，必须挂载
- 入口脚本内置了已调优的默认参数（视觉、MTP 投机解码、KV 自动容量等），全部可用环境变量覆盖，也可以在 `docker run` 末尾追加原生 `ninfer-serve` 参数（原样透传）

## 1. 前置条件

| 项 | 要求 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090（sm_89）。24 GB 或 48 GB 显存均可；显存越小，需要把上下文 / KV 量化调得越激进（见第 8 节） |
| 驱动 | 需支持 CUDA 13.1（镜像运行阶段基于 `nvidia/cuda:13.1.2-runtime-ubuntu24.04`）。两台已实测可用的 4090 机器驱动分别为 **595.58.03**（CUDA 13.2）与 **610.43.02**，均正常工作；最低驱动版本请以 NVIDIA 官方 CUDA 13.1 兼容性矩阵为准 |
| Docker | 任意较新版本 Docker Engine |
| nvidia-container-toolkit | 必需。安装后 `docker info` 的 Runtimes 里应能看到 `nvidia` |
| 磁盘 | 模型约 21 GiB（约 22.8 GB）+ 镜像（运行阶段为 CUDA runtime 底座，预计几个 GiB，以 `docker image ls` 实际输出为准） |
| 网络 | 能访问 `ghcr.io` 拉镜像、能访问 `huggingface.co` 下载模型 |

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

## 2. 获取并校验模型

官方 HF 仓库：`https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer`，文件名 `qwen3_6_35b_a3b.ninfer`。

> **重要：该仓库存在多个不同 revision 的模型文件，大小和 hash 都不同。**
> 撰写本文档时已知至少三个版本（以下为快照，可能随仓库更新变化，请勿当作唯一标准）：
>
> | 来源 | 字节数 | SHA256 |
> |---|---|---|
> | `main` 分支（`artifact-manifest.json` 与 `SHA256SUMS` 一致） | 22,790,484,480 | `3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84` |
> | 仓库自带下载脚本钉住的 revision `c8b8c1c0` | 22,373,184,256 | `9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163` |
> | 一份更早 revision 的历史下载 | 22,783,246,080 | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |
>
> 正确做法：**先认准你要用的 revision，再在同一 revision 上取 `SHA256SUMS` 做比对**，避免"文件 hash 对不上 SUMS"的困惑。

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

`scripts/download-qwen36-35b-vision.sh` 使用 `curl -L -C -` 断点续传，中断后重新运行即可继续。注意该脚本钉住了 revision `c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c`（校验时请对同一 revision 的 `SHA256SUMS`）：

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

> **额外提醒**：不同 revision 的 artifact 可能不兼容（引擎只认自己支持的 `container_version`）。hash 对上 SUMS 只说明"下载完整"，不保证"引擎能加载"——**最终以 `ninfer-serve` 实际启动成功为准**（见第 6 节验证）。

## 3. 一步到位的最小可运行命令

假设模型在宿主机 `/opt/models/qwen3_6_35b_a3b.ninfer`（换成你的实际路径）：

> **注意：该 GHCR 包目前是 private 的**（匿名 `docker pull` 会返回 403）。先登录对该包有读权限的 GitHub 账号再拉：

```bash
docker login ghcr.io
# Username: 你的 GitHub 用户名
# Password: 有 read:packages 权限的 PAT
```

CI 环境可用 `echo "$GHCR_TOKEN" | docker login ghcr.io -u <用户名> --password-stdin`。若日后该包改为 public，匿名 `docker pull` 即可。

```bash
docker run -d --name ninfer \
  --gpus "device=0" \
  -v /opt/models:/models:ro \
  -p 8000:8000 \
  ghcr.io/yorkane/ninfer-4090:latest
```

说明：

- `-v /opt/models:/models:ro`：模型目录只读挂载到容器 `/models`（`NINFER_MODEL` 默认值就是 `/models/qwen3_6_35b_a3b.ninfer`）
- `-p 8000:8000`：容器内服务默认监听 8000
- `--gpus "device=0"`：entrypoint 固定使用容器内 `--device 0`，所以想跑在物理 N 号卡上，就把 `device=N` 填进去；单卡机也可以用 `--gpus all`
- 本服务是单进程 C++ 服务，不依赖 NCCL，通常**不需要** `--ipc=host` 或 `--shm-size`
- 首次启动需加载约 21 GiB 权重并构建 CUDA Graph，耗时以实际为准；`docker logs -f ninfer` 观察，看到服务就绪 / 开始监听即完成

## 4. 模型目录挂载

映射关系：**宿主机目录 → 容器 "/models"**。entrypoint 要求 `NINFER_MODEL` 指向容器内一个已存在的文件，找不到会直接退出并打印：

```
FATAL: model artifact not found: /models/qwen3_6_35b_a3b.ninfer
Mount the directory holding the .ninfer file, e.g.
  -v /path/to/models:/models:ro
```

三种常见情况：

1. **文件名就是 `qwen3_6_35b_a3b.ninfer`**：`-v <宿主机目录>:/models:ro` 即可，其他什么都不用做。
2. **文件名不同**（如 `qwen3_6_35b_a3b_v2.ninfer`）：加 `-e NINFER_MODEL=/models/qwen3_6_35b_a3b_v2.ninfer`。
3. **想挂到容器内其他路径**：`-v /opt/models:/data/m:ro -e NINFER_MODEL=/data/m/qwen3_6_35b_a3b.ninfer`。

挂载加 `ro`（只读）是推荐做法，服务不会写模型文件。

## 5. 环境变量全表

以下默认值与 `entrypoint.sh` 实际内容一致。

| 变量 | 默认值 | 对应参数 | 说明 / 什么时候该调 |
|---|---|---|---|
| `NINFER_MODEL` | `/models/qwen3_6_35b_a3b.ninfer` | 位置参数（artifact 路径） | 容器内模型文件路径，文件名不同必须覆盖 |
| `NINFER_PORT` | `8000` | `--port` | 容器内监听端口；改了它，`-p` 右侧端口要同步改 |
| `NINFER_HOST` | `0.0.0.0` | `--host` | 容器内监听地址；容器场景保持 0.0.0.0（对外可见性由 `-p` 控制） |
| `NINFER_CONC` | `8` | `--max-concurrency` | 最大同时生成请求数，合法范围 `1..8`；显存紧张或追单请求低延迟时调小，纯吞吐保持 8 |
| `NINFER_MAX_CONTEXT` | `262144` | `--max-context` | 每序列逻辑上下文上限（prompt+completion，token 数）；显存不够时优先调小，不跑超长上下文时调小可显著省显存并加快启动 |
| `NINFER_KV_DTYPE` | `rk4v4-e8` | `--kv-dtype` | KV cache 量化格式：`bf16`/`int8`/`rk8v4`/`rk4v4`/`rk4v4-e8`/`rk2v4-e8`；取舍见第 8 节 |
| `NINFER_VISION_MAX_TOKENS` | `65536` | `--vision-max-tokens` | 视觉展开后的 token 上限；喂超大图/长视频且显存紧张时调小 |
| `NINFER_MAX_MEDIA_ITEMS` | `64` | `--max-media-items` | 单请求媒体条目数上限；批量多图按需调大 |
| `NINFER_MAX_VIDEO_PIXELS` | `4294967296`（2^32） | `--max-video-pixels` | 视频总像素预算（所有帧合计）；长视频显存不够时调小 |
| `NINFER_VIDEO_MAX_PIXELS` | `67108864`（64 MP） | `--video-max-pixels` | 采样后**所有保留帧的合计像素体积**上限（会按各帧摊分，从而决定每帧清晰度）；设 `0` 表示沿用 artifact 内 `video_preprocessor_config.json` 的值。注意它不改变视觉显存上限，要同时调 `NINFER_VISION_MAX_TOKENS` 才能放大显存预算 |
| `NINFER_DRAFT_TOKENS` | `3` | `--draft-tokens` | MTP 每轮 draft token 数（1..15）；调大提速但接受率下降，3 是较稳默认 |
| `NINFER_PREFILL_CHUNK` | `1024` | `--prefill-chunk` | prefill 分块大小（128 的倍数）；一般不用动 |
| `NINFER_MODEL_ID` | `qwen3.6-35b-a3b` | `--model-id` | 对外公开模型别名，请求 `model` 字段必须与它一致；接入要求特定 model 名的客户端时改它 |
| `NINFER_API_KEY` | 空（不鉴权） | `--api-key` | 设置后 OpenAI 风格需 `Authorization: Bearer <key>`，Anthropic 风格需 `x-api-key`；`/health` 免鉴权 |
| `NINFER_THINKING` | `off` | `off` 时追加 `--no-thinking` | 默认思考开关，只接受 `off/on`（含 `false/true/0/1`），非法值容器直接退出；单请求仍可用 `reasoning_effort`/`enable_thinking` 覆盖 |

### entrypoint 中写死、无环境变量对应的参数

这些由 `entrypoint.sh` 固定传入；需要改动时在 `docker run` 末尾追加原生参数覆盖（追加参数原样透传，同名参数后者生效）：

| 参数 | 值 | 含义 |
|---|---|---|
| `--device` | `0` | 容器内 CUDA 设备号；多卡用 `--gpus "device=N"` 把物理卡映射进容器 0 号 |
| `--kv-capacity` | `auto` | 权重加载后用剩余显存尽量开大 KV 池（保留 1 GiB 余量），见第 8 节 |
| `--vision` | 开 | 固定启用图片/视频输入 |
| `--spec mtp --lm-head-draft` | 开 | 固定启用 MTP 投机解码 + 优化 draft head |
| `--max-pending-requests` | `64` | 准入队列深度（超出返回 429 `server_overloaded`） |
| `--pending-timeout-ms` | `900000` | 排队+准备超时 15 分钟（超时返回 503 `request_queue_timeout`） |

示例——显式限死 KV 容量、关 CUDA Graph：

```bash
docker run -d --name ninfer --gpus "device=0" \
  -v /opt/models:/models:ro -p 8000:8000 \
  ghcr.io/yorkane/ninfer-4090:latest \
  --kv-capacity 131072 --no-cuda-graph
```

## 6. 验证服务

```bash
# 1) 健康检查
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

带图片（支持 HTTP(S) URL 或 base64 data URL；服务固定开启 `--vision`）：

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

设置了 `NINFER_API_KEY` 时，以上 `/v1/*` 请求都要加 `-H "Authorization: Bearer <key>"`。

## 7. 多卡部署（每卡一个容器）

entrypoint 固定 `--device 0`，所以每卡一容器的正确姿势是用 `--gpus "device=N"` 把不同物理卡映射进各自容器的 0 号，再各占一个宿主机端口：

```bash
for i in 0 1 2 3; do
  docker run -d --name "ninfer-gpu$i" \
    --gpus "device=$i" \
    -v /opt/models:/models:ro \
    -p 820$i:8000 \
    ghcr.io/yorkane/ninfer-4090:latest
done
```

- 4 张 4090 得到 `8200/8201/8202/8203` 四个相同服务，可再接上游网关做负载均衡
- 多容器共用同一只读模型挂载没有问题
- 某张卡需要不同参数（如 24 GB 卡调小上下文）时，单独给该容器加 `-e`：
  `docker run -d --name ninfer-gpu1 --gpus "device=1" -v /opt/models:/models:ro -p 8201:8000 -e NINFER_MAX_CONTEXT=131072 ghcr.io/yorkane/ninfer-4090:latest`

## 8. KV / 显存相关调参

### --kv-dtype 可选值与取舍

| 值 | 格式 | 精度（vs FP32 余弦相似度，README 实测口径） | 显存 | 适用 |
|---|---|---|---|---|
| `bf16` | 16-bit 非压缩 | 最高 | 最大 | 一般不必用，24 GB 卡上下文会很短 |
| `int8` | 8-bit 通道量化 | 约 99.8% | 中 | 高精度优先、上下文需求中等 |
| `rk8v4` | 8-bit 秩压缩 K + 4-bit V | 高 | 中 | 介于 int8 与 rk4v4 之间 |
| `rk4v4` | 4-bit Hadamard K + 4-bit V | 约 97.8% | 小 | 容量/精度平衡 |
| `rk4v4-e8` | 4-bit E8 晶格 K + 4-bit V | 约 98.7% | 小 | **entrypoint 默认**，推荐起点 |
| `rk2v4-e8` | 2-bit E8 圆柱 K + 4-bit V | 约 96.2% | 最小 | 超长上下文、显存极限场景 |

参考：README 给出的 24 GB RTX 4090 文本-only（MTP4）物理上限/建议安全值约为——`rk2v4-e8` 462k/430k、`rk4v4-e8` 352k/320k、`int8` 181k/160k；视觉（8k 默认）下 415k/380k、317k/280k、163k/140k。48 GB 卡按剩余显存自然放大，以启动日志报出的实际 KV 容量为准。

### --kv-capacity auto 的含义

entrypoint 固定传入 `--kv-capacity auto`：权重加载完成后，用**剩余显存自动开大**共享 KV 池，保留 1 GiB sizing 余量；启动日志会报出解析出的容量与余量。想显式限死（给同卡其他东西留显存）时，在 `docker run` 末尾追加 `--kv-capacity <token数>` 覆盖。

### 显存不够（OOM / 启动失败）时的降配顺序

1. 调小 `NINFER_MAX_CONTEXT`（默认 262144，很大；不跑超长上下文就先降它，如 131072 → 65536）
2. 换更激进的 `NINFER_KV_DTYPE`（`rk4v4-e8` → `rk2v4-e8`）
3. 调小 `NINFER_CONC`（8 → 4）
4. 视觉场景再降 `NINFER_VISION_MAX_TOKENS` / `NINFER_MAX_VIDEO_PIXELS` / `NINFER_VIDEO_MAX_PIXELS`

## 9. 常见问题

**容器启动即退出？**

```bash
docker logs --tail 200 ninfer
```

最常见输出是 `FATAL: model artifact not found: ...`——挂载路径或 `NINFER_MODEL` 没对上，按第 4 节检查。

**GPU 不可见（容器内 nvidia-smi 报错 / CUDA 初始化失败）？**

- 宿主机 `nvidia-smi` 是否正常？
- `docker info | grep -iE 'runtimes|nvidia'` 是否列出了 nvidia runtime？没有就重做第 1 节并 `sudo systemctl restart docker`
- `docker run --rm --gpus all nvidia/cuda:13.1.2-runtime-ubuntu24.04 nvidia-smi` 能否看到卡？
- `--gpus "device=N"` 的 N 是否超出物理卡数？换成 `--gpus all` 试试

**OOM / 显存不足？** 按第 8 节降配顺序处理；先看启动日志报的 KV 容量与"权重后剩余显存"，确认是 KV 池开不下还是其他分配失败。

**端口占用（"port is already allocated"）？** 换宿主机侧端口：`-p 8200:8000`（容器内 8000 不动，或同步改 `NINFER_PORT`）。多卡部署见第 7 节。

**请求被拒 429 / 503？** `429 server_overloaded` = 活跃+排队请求达到 `NINFER_CONC + 64` 上限；`503 request_queue_timeout` = 排队 15 分钟未被准入（通常是前面有超长 prefill 占位）。调小客户端并发，或接受默认值。

**请求报 model 不匹配？** 请求体 `model` 字段必须等于 `NINFER_MODEL_ID`（默认 `qwen3.6-35b-a3b`）。

**拉镜像 403 / unauthorized / 限流？** 该包目前是 **private** 的：`docker login ghcr.io` 登录的账号必须对该包有读权限（包 owner 或已被授权的用户）才能拉，匿名请求一律 403；有权限账号同样可能遇到限流，登录后重试即可。

## 10. 构建自己的镜像（可选）

在源码仓库根目录（含 `Dockerfile` 与 `entrypoint.sh`）：

```bash
docker build -t ninfer-4090:local .
```

- 两阶段构建：`nvidia/cuda:13.1.2-devel-ubuntu24.04` 里 CMake+Ninja 编译 `ninfer` 与 `ninfer-serve`，最终只把两个二进制拷进 `nvidia/cuda:13.1.2-runtime-ubuntu24.04` 运行层，**不打包模型**
- 编译耗时取决于宿主机 CPU 核数（并行度），以实际为准；构建机核数越多越快
- 最终镜像体积预计几个 GiB（CUDA runtime 底座 + 二进制），以 `docker image ls` 实际输出为准
- 本地构建后可直接把第 3 节命令里的 `ghcr.io/yorkane/ninfer-4090:latest` 换成 `ninfer-4090:local`

---

*本文档依据仓库中 `entrypoint.sh`、`Dockerfile`、`docs/serving.md`、`README.md` 的实际内容编写；标注"以实际为准"处请以你机器上的真实输出为准。*
