# Radxa Orion O6 NPU Frigate Setup

This guide details how to leverage the Orion O6's NPU (Neural Processing Unit) for hardware-accelerated object detection in Frigate. By offloading detection to the NPU, you can process high-resolution streams with minimal CPU usage.

*Photo by [Shir Danieli](https://www.pexels.com/photo/urban-street-scene-with-pedestrians-and-building-35465130/), modified to look "security camera-ish"*

![](./npu_sample.jpg)

---

## 1. Host System Preparation

Before running Frigate, you must install the necessary drivers and AI models on the Orion O6 host operating system.

### Install CIX SDK Drivers

Follow the [Radxa NPU SDK Installation guide](https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence/npu-introduction). You strictly need the runtime drivers, not the compilation tools.

1. Download the SDK and extract it.
2. Install the User Mode Driver (UMD) and Kernel Driver:

```bash
sudo dpkg -i cix-npu-driver_xxx_arm64.deb
sudo dpkg -i cix-noe-umd_xxx_arm64.deb

```

*(Ignore packages related to "compiler" or "builder" unless you plan to compile models yourself)*

### Install CIX AI Model Hub

This repository contains the pre-compiled `.cix` models required for inference.

Follow [these instructions to install the CIX AI Model Hub](https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence/ai-hub).

1. **Install Git LFS** (Crucial for large model files):

```bash
sudo apt-get update && sudo apt-get install git-lfs
git lfs install

```

2. **Clone the Model Hub**:
Navigate to your intended configuration directory *for Frigate* (e.g., `/home/radxa/frigate/config`) and clone the repo listed in the instructions.
*Note: You may delete models in subdirectories you do not plan to use to save space.*

3. **Verify Model Functionality**:
Test that the NPU is working by running a sample inference script directly on the host:
```bash
python3 inference_npu.py

```

---

## 2. Directory Structure

Ensure your project folder matches this structure. This keeps your configuration, custom plugins, and media storage organized.

```bash
radxa@RadxaOrionO6 ~/Docker_Containers/frigate
.
├── docker-compose.yml
├── config
│   ├── config.yaml
│   ├── ai_model_hub_25_Q3/  <-- Cloned Repo
│   └── model_cache
│       └── oriono6
│           └── yolox_m.cix  <-- Copy your preferred model here for easy access
├── plugins
│   └── detectors
│       └── oriono6.py       <-- Custom plugin file
└── storage
    ├── clips
    ├── exports
    └── recordings

```

---

## 3. Docker Configuration

Configure your `docker-compose.yml` to map the NPU devices and your custom plugin directories. The "ffmpeg hardware transplant" relies on mounting the host's specific ffmpeg binaries or libraries if you are not using a custom-built image.

```yaml
services:
  frigate:
    container_name: frigate
    privileged: true # Required for hardware access
    restart: unless-stopped
    image: ghcr.io/blakeblackshear/frigate:stable
    shm_size: "128mb" # Recommended for multiple cameras

    # CRITICAL: Tell Frigate to check the custom Radxa folders for libraries
    environment:
      - LD_LIBRARY_PATH=/opt/cixgpu-compat/lib/aarch64-linux-gnu:/opt/cixgpu-pro/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:/usr/local/lib

    ports:
      - "8971:8971"
      - "8554:8554" # RTSP feeds
      - "8555:8555/tcp" # WebRTC
      - "8555:8555/udp" # WebRTC

    volumes:
      # --- Config & Storage ---
      - ./config:/config
      - ./storage:/media/frigate
      - type: tmpfs
        target: /tmp/cache
        tmpfs:
          size: 1000000000

      # --- HARDWARE TRANSPLANT ---
      - /usr/bin/ffmpeg:/usr/lib/custom_ffmpeg/bin/ffmpeg:ro
      - /usr/bin/ffprobe:/usr/lib/custom_ffmpeg/bin/ffprobe:ro
      - /lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:ro
      - /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro
      - /opt/cixgpu-compat:/opt/cixgpu-compat:ro
      - /opt/cixgpu-pro:/opt/cixgpu-pro:ro

      # --- PLUGIN & SDK MOUNTS ---
      # 1. Map your custom plugin folder
      - ./plugins/detectors/oriono6.py:/opt/frigate/frigate/detectors/plugins/oriono6.py

      # 2. Map the Radxa AI Hub Code (so we can import 'utils')
      - /home/radxa/Downloads/ai_model_hub_25_Q3:/opt/radxa_ai_hub:ro

      # 3. Map the Venv Libraries (so we can import 'libnoe')
      - /home/radxa/Downloads/ai_model_hub_25_Q3/venv/lib/python3.11/site-packages:/opt/radxa_venv:ro
```

---

## 4. Frigate Config (`config.yaml`)

Update your Frigate config to use the custom detector.

```yaml
ffmpeg:
  path: /usr/lib/custom_ffmpeg
  hwaccel_args:
    - -hwaccel
    - drm
    - -hwaccel_device
    - /dev/dri/renderD128

# Frigate reads this to set up the hardware video pipeline correctly
model:
  path: /config/model_cache/oriono6/yolox_m.cix
  width: 640
  height: 640
  input_tensor: nhwc
  input_pixel_format: bgr
  input_dtype: int # Metadata only, we handle float conversion in script

detectors:
  orion:
    type: oriono6

cameras:
  LivingRoom:
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://PLACEHOLDER:RTSP_STREAM@192.168.X.X
          roles:
            - detect
            - record
    detect:
      enabled: true
      width: 1920
      height: 1080
      fps: 5

objects:
  track:
    - person

detect:
  enabled: true
  fps: 5
version: 0.16-0

```

---

## 5. Verification & Performance

### Startup Check

Launch the container:

```bash
sudo docker compose up -d

```

Check the logs to confirm the plugin loaded:

```bash
sudo docker logs frigate -f

```

Look for the confirmation line:
`oriono6.py: --- ORION YOLOX (OPTIMIZED) --- Ready.`

### Hardware Stats

The Orion O6 is capable of impressive performance. Using `yolox_m.cix` on a 1080p stream typically results in roughly **40ms inference speeds**, leaving the CPU idle for other tasks.

The docker compose also sets up an `ffmpeg` hardware transplant to enable hardware encoding such that it will use `ffmpeg version 5.1.6-0+deb12u1+cix` instead of the default ffmpeg installation.

![Detection](detection_overview.png)

![Inference](detector_inference_speed.png)

![Stats](detector_stats.png)