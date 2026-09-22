# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
# CMake >= 3.30 is required (CMP0169 in CMakeLists.txt); the distro cmake on noble is 3.28.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/cmake.tar.gz \
        https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.tar.gz \
    && mkdir -p /opt/cmake \
    && tar --strip-components=1 -C /opt/cmake -xzf /tmp/cmake.tar.gz \
    && rm -f /tmp/cmake.tar.gz

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        git \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN export PATH=/opt/cmake/bin:$PATH \
    && cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

# OCI metadata. The source label is what links this package to its GitHub
# repository on GHCR, so keep it in step with the origin remote.
LABEL org.opencontainers.image.title="ninfer-4090" \
      org.opencontainers.image.description="NInfer sm_89 inference server for Qwen3.6-35B-A3B (vision, MTP, rk4v4-e8 KV)" \
      org.opencontainers.image.source="https://github.com/yorkane/ninfer-4090" \
      org.opencontainers.image.url="https://github.com/yorkane/ninfer-4090" \
      org.opencontainers.image.documentation="https://github.com/yorkane/ninfer-4090/blob/main/docs/deploy.md" \
      org.opencontainers.image.licenses="Apache-2.0"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY entrypoint.sh /usr/local/bin/ninfer-entrypoint
RUN chmod +x /usr/local/bin/ninfer /usr/local/bin/ninfer-serve /usr/local/bin/ninfer-entrypoint

WORKDIR /workspace

# The model artifact is mounted, never baked in (it is ~21 GiB).
VOLUME ["/models"]

EXPOSE 8000
STOPSIGNAL SIGTERM

# Bring up the tuned serving profile against the mounted artifact.
# Override any of it with NINFER_* env vars, or append raw ninfer-serve flags.
ENTRYPOINT ["/usr/local/bin/ninfer-entrypoint"]
