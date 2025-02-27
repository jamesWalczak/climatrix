FROM python:3.12-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Upgrade pip and install the package
RUN pip install --upgrade pip \
    && pip install .

CMD ["python", "-m", "climatrix"]
