# Kimi K3 KDA TPU kernel benchmarks — hermetic build from repo contents
# (kernel code comes from the sglang-jax submodule, not a network clone)
# Build (repo root): docker buildx build --platform linux/amd64 -f docker/Dockerfile -t ghcr.io/junyi-99/kda-tpu-bench:latest .
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/Junyi-99/kda-tpu-bench"
LABEL org.opencontainers.image.description="Kimi K3 KDA prefill kernel benchmarks on TPU v6e (SGLang-JAX, pinned)"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pinned JAX TPU stack
RUN pip install --no-cache-dir "jax[tpu]==0.11.1" "libtpu==0.0.46" "xprof==2.23.1" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --no-cache-dir transformers tiktoken blobfile matplotlib

# Kernel + integration from the reviewed submodule (hermetic)
COPY sglang-jax /root/sglang-jax

# Bench scripts reference ~/venv-sgl/bin/python — shim to system python
RUN mkdir -p /root/venv-sgl/bin && ln -s /usr/local/bin/python /root/venv-sgl/bin/python

# 双内核：baseline = 上游 main（钉死）；final = submodule 最新（构建时快照）
COPY kernels/baseline/ /root/kernels/baseline/
RUN cp -r /root/sglang-jax/python/sgl_jax/srt/kernels/kda /root/kernels/final

COPY models/ /root/models/
COPY bench/ /root/bench/
COPY docker/reproduce.sh /root/reproduce.sh
RUN chmod +x /root/reproduce.sh

ENV HOME=/root
ENV JAX_COMPILATION_CACHE_DIR=/root/jax_kernel_cache
WORKDIR /root
ENTRYPOINT ["/root/reproduce.sh"]
CMD ["help"]

# Published results shipped in-image
COPY results/ /root/results/
