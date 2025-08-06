FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    wget curl git unzip bash \
    && rm -rf /var/lib/apt/lists/*

COPY conf ./conf/
RUN bash ./conf/setup.sh

ENV PATH="/app/conf/exp1/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ./conf/requirements.txt

COPY images/ ./images/
COPY scripts/ ./scripts/

RUN mkdir -p /app/results

RUN chmod +x ./images/scripts/entrypoint.sh && \
    find . -name "*.sh" -type f -exec chmod +x {} +

ENTRYPOINT ["./entrypoint.sh"]