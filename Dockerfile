# Reproducible environment for the Hendricks retail analytics pipeline.
#
# Based on an NVIDIA CUDA runtime image so the tracking pass can use a GPU.
# The pipeline runs on CPU as well -- the code selects the device
# automatically -- but the one-time tracking pass over both videos takes
# roughly 8 minutes on an RTX 4060 versus considerably longer on CPU.
#
# Build:
#   docker build -t hendricks-analytics .
#
# Run (GPU), mounting the videos in and the results out:
#   docker run --gpus all \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/outputs:/app/outputs" \
#     hendricks-analytics
#
# Run (CPU only): drop --gpus all.
#
# The default command runs the whole pipeline end to end. To run a single
# stage instead, append it, e.g.:
#   docker run --gpus all -v "$(pwd)/data:/app/data" hendricks-analytics \
#     python src/task1_interest_conversion.py

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Ultralytics otherwise tries to write settings/telemetry into a
    # non-writable home directory inside the container.
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# libgl1 + libglib2.0 are required by opencv-python; ffmpeg provides the video
# decoders OpenCV shells out to for mp4.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# PyTorch is installed from its own index, not PyPI: the CUDA builds carry a
# +cu121 local version suffix that a plain `pip install -r requirements.txt`
# cannot resolve. requirements.txt therefore deliberately omits torch and
# torchvision. Installed first, and in its own layer, so the large download is
# cached across rebuilds.
RUN pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model weights are baked into the image so the container needs no network at
# run time. Ultralytics would otherwise download them on first use.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m-pose.pt')" \
    && mkdir -p models \
    && mv yolov8m-pose.pt models/ 2>/dev/null || true

COPY src/ ./src/
COPY configs/ ./configs/
COPY README.md .

# data/ and outputs/ are expected as mounts. Created here so the pipeline can
# run (and fail with a clear "video not found") even if a mount is forgotten.
RUN mkdir -p data/raw_videos data/cache outputs

# Build the tracking cache once, then every task reads from it.
CMD ["bash", "-c", "\
python src/build_cache.py && \
python src/task1_interest_conversion.py && \
python src/task2_shelf_interest.py --video interior && \
python src/task2_shelf_interest.py --video entrance && \
python src/task3_staff_interaction.py && \
python src/render_entrance_video.py && \
python src/render_task2_video.py && \
echo 'Pipeline complete. Results in outputs/.'"]
