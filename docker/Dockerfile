# Vesper sim image. Step 0: Isaac Lab base + vesper package + ffmpeg.
# Bump the tag deliberately -- STACK.md pins the 2.3.x line (Isaac Sim 5.1).
ARG ISAAC_LAB_TAG=2.3.0
FROM nvcr.io/nvidia/isaac-lab:${ISAAC_LAB_TAG}

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y

# ffmpeg stitches capture frames into MP4s
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Step 1: PX4 SITL (slow layer, keep above the repo COPY for caching) ---
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git sudo lsb-release \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v1.16.0 --recursive \
    https://github.com/PX4/PX4-Autopilot.git /px4/PX4-Autopilot
RUN /px4/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools \
    && cd /px4/PX4-Autopilot && DONT_RUN=1 make px4_sitl_default

# --- Step 1: Pegasus Simulator + MAVSDK ---
ENV ISAACSIM_PATH=/isaac-sim \
    ISAACSIM_PYTHON_EXE=/isaac-sim/python.sh
RUN git clone --depth 1 --branch v5.1.0 \
    https://github.com/PegasusSimulator/PegasusSimulator.git /pegasus
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -e /pegasus/extensions/pegasus.simulator \
    && /isaac-sim/python.sh -m pip install --no-cache-dir mavsdk

# vesper package (compose bind-mounts the live repo over this path at runtime)
COPY . /workspace/vesper
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -e /workspace/vesper

WORKDIR /workspace/vesper
ENTRYPOINT ["/workspace/vesper/docker/entrypoint.sh"]
CMD ["bash"]
