# Python NPU Demos for RK3588 Devices

This directory contains Python demonstration applications for running neural network models on RK3588 NPU hardware using the RKNN Lite API.

## Available Demos

### 1. Face Mask Detection (`face_mask/`)
- **face_mask.py** - Face mask detection on static images
- **face_mask_cap.py** - Real-time face mask detection using webcam
- Detects whether people are wearing face masks using an anchor-based detection model
- Outputs bounding boxes with "Mask" (green) or "NoMask" (red) labels

### 2. InceptionV3 (`inceptionv3/`)
- Image classification using the InceptionV3 model
- Demonstrates quantized model inference
- Outputs top-5 classification results with confidence scores

### 3. ResNet18 (`resnet18/`)
- Image classification using the ResNet18 model
- Optimized for RK3588 NPU
- Returns top-5 predictions with softmax scores

### 4. SSD MobileNet (`ssd/`)
- Object detection using SSD MobileNet V1 trained on COCO dataset
- Detects 91 object classes
- Performs Non-Maximum Suppression (NMS) for optimal results

## Installation

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver, written in Rust. It's significantly faster than pip and provides better dependency resolution.

#### Install uv

```sh
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Setup Runtime Library

```sh
# Copy RKNN runtime library
sudo cp ../C++/runtime/librknn_api/aarch64/librknnrt.so /usr/lib
```

#### Install Dependencies with uv

```sh
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev

# Create virtual environment and install dependencies using uv
uv venv
source .venv/bin/activate
uv pip install numpy opencv-python

# Install RKNN Toolkit Lite
uv pip install ./wheel/rknn_toolkit_lite2-1.3.0-cp310-cp310-linux_aarch64.whl
```

### Option 2: Using pip (Traditional)

```sh
# Copy RKNN runtime library
sudo cp ../C++/runtime/librknn_api/aarch64/librknnrt.so /usr/lib

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y python3-opencv python3-numpy

# Install RKNN Toolkit Lite
pip3 install ./wheel/rknn_toolkit_lite2-1.3.0-cp310-cp310-linux_aarch64.whl
```

## Running the Demos

### ResNet18 - Image Classification

```sh
cd resnet18
python3 resnet18.py
```

**Expected Output:**
```
--> Load RKNN model
done
--> Init runtime environment
I RKNN: [17:07:10.282] RKNN Runtime Information: librknnrt version: 1.3.0
done
--> Running model
resnet18
-----TOP 5-----
[812]: 0.9996383190155029
[404]: 0.00028062646742910147
[657]: 1.632110434002243e-05

done
```

### InceptionV3 - Image Classification

```sh
cd inceptionv3
python3 inceptionv3.py
```

Runs inference on `data/img/goldfish_299x299.jpg` and outputs top-5 predictions.

### SSD MobileNet - Object Detection

```sh
cd ssd
python3 ssd.py
```

Detects objects in `data/img/road.bmp` and saves annotated output to `out.jpg`.

### Face Mask Detection - Static Image

```sh
cd face_mask
python3 face_mask.py
```

Processes `data/img/face.jpg` and saves the result to `out.jpg` with bounding boxes indicating mask/no-mask detection.

### Face Mask Detection - Real-time Webcam

```sh
cd face_mask
python3 face_mask_cap.py --device 0
```

**Arguments:**
- `--device` - Video device number (e.g., 0 for /dev/video0)

**Controls:**
- Press `q` to quit

Displays real-time face mask detection with:
- Green boxes: Person wearing mask
- Red boxes: Person not wearing mask

## Project Structure

```
Python/
├── pyproject.toml              # uv/pip package configuration
├── README.md                   # This file
├── face_mask/
│   ├── face_mask.py           # Static image detection
│   ├── face_mask_cap.py       # Webcam detection
│   └── data/
│       ├── model/             # RKNN model files
│       └── img/               # Test images
├── inceptionv3/
│   ├── inceptionv3.py
│   └── data/
├── resnet18/
│   ├── resnet18.py
│   └── data/
├── ssd/
│   ├── ssd.py
│   └── data/
│       ├── box_priors.txt     # Anchor box priors
│       └── coco_labels_list.txt
└── wheel/
    └── rknn_toolkit_lite2-*.whl  # RKNN Lite wheel
```

## Notes

- All demos require the RKNN runtime library (`librknnrt.so`) to be installed
- Pre-trained RKNN models are located in each demo's `data/model/` directory
- Models are optimized for RK3588 NPU hardware
- Some demos use specific NPU cores (e.g., `NPU_CORE_0`) for better performance

## Troubleshooting

**Model loading fails:**
- Ensure `librknnrt.so` is properly installed in `/usr/lib`
- Verify the model file exists in the `data/model/` directory

**Camera not found (face_mask_cap.py):**
- Check your video device number with `ls /dev/video*`
- Try different device numbers: `--device 0`, `--device 1`, etc.

**Import errors:**
- Ensure all dependencies are installed via uv or pip
- Activate your virtual environment if using one
