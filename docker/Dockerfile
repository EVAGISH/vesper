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

# vesper package (compose bind-mounts the live repo over this path at runtime)
COPY . /workspace/vesper
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -e /workspace/vesper

# --- Step 1 layers, not yet enabled ---
# Pegasus Simulator v5.1.0: clone + /isaac-sim/python.sh -m pip install -e extensions/pegasus.simulator
# PX4-Autopilot v1.16: deps + make px4_sitl_default

WORKDIR /workspace/vesper
ENTRYPOINT ["/workspace/vesper/docker/entrypoint.sh"]
CMD ["bash"]
