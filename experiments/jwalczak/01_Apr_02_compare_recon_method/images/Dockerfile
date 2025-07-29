FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv wget curl git bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ../ /app

RUN chmod +x scripts/*.sh

RUN bash conf/setup.sh

ENV VIRTUAL_ENV=/app/conf/exp1
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir -p /app/results

CMD bash -c "scripts/download_blend_mean_temperature.sh && \
             python scripts/prepare_ecad_observations.py && \
             bash scripts/run_experiments.sh"
