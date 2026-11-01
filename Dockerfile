FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libopus0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep installed packages outside /app: Bothost can mount the source folder there.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --only-binary=:all: -r /tmp/requirements.txt \
    && python -c "import ctypes.util, discord.opus; discord.opus.load_opus(ctypes.util.find_library('opus')); assert discord.opus.is_loaded()"

WORKDIR /app
COPY bot.py ./bot.py
COPY music/ ./music/

CMD ["python", "bot.py"]
