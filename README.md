# Edge AI Fruit Classifier on ESP32-CAM (TinyML)

An end-to-end TinyML computer vision system designed to train, quantize, and deploy an ultra-lightweight Convolutional Neural Network (CNN) on an **AI-Thinker ESP32-CAM** to perform fruit classification across three target classes: **Apple**, **Banana**, and **Orange**.

## Hardware Specifications

* **Target Microcontroller:** AI-Thinker ESP32-CAM (Xtensa Dual-Core 32-bit LX6 running at 240 MHz, 520 KB internal SRAM, 4 MB external PSRAM).
* **Camera Module:** Omnivision OV2640 Image Sensor.
* **Development Machine:** Apple Silicon M4.
* **Core Frameworks:** TensorFlow, Keras, TensorFlow Lite, Edge Impulse Python SDK, Arduino Core for ESP32.
* **Model Format:** Full 8-bit Integer Quantization (INT8).
* **Input Resolution:** 96 x 96 single-channel Grayscale.
* **Target Classes:** Apple, Banana, Orange.

## Key Architecture & Engineering Decisions

### Grayscale Conversion Over RGB

* Capturing and processing single-channel grayscale data reduces the input tensor size by 66.7% compared to RGB.
* This dramatically cuts down memory consumption inside the ESP32’s limited SRAM while retaining essential geometric shapes, contours, and edge textures required for fruit recognition.

### Three-Block Convolutional Pipeline

* The feature extractor utilizes three progressive convolution stages (scaling from 16 to 32 to 64 filters).
* This provides sufficient representational depth to identify curvature and textural boundaries while preventing the parameter explosion common in deeper networks.

### Global Average Pooling Instead of Flattening

* Replacing a standard flattening layer with Global Average Pooling collapses two-dimensional feature maps into single summary values.
* This drastically slashes total trainable parameters from tens of thousands down to a fraction, actively preventing overfitting on compact datasets and reducing the memory footprint.

## Dataset Pipeline & Verification Strategy

### Unbiased Verification Quarantine

* Dataset was taken from Kaggle with around 120 images per category.
* Before training, a dedicated verification subset (10 unseen images per class, 30 images total) was completely isolated into a separate directory.
* These images were excluded from both training iterations and quantization calibration loops to provide an honest, unbiased benchmark of post-quantization performance.

### In-Memory Training Augmentation

* To prevent the network from memorizing static backgrounds and orientations, dynamic data augmentations (random horizontal/vertical flips, subtle rotations, translations, and zoom adjustments) were applied during training to force the model to learn invariant geometric shapes.

## 4. INT8 Quantization Methodologies

Crushing 32-bit floating-point weights and activation tensors down to 8-bit integers is mandatory to achieve low-latency execution and minimal memory usage on microcontrollers lacking hardware floating-point units.

### Method A: Local Mac Quantization (TensorFlow Lite)
Modern Keras 3 versions caused `NoneType` tracing bugs during standard conversion. To bypass this, the trained weights were extracted and compiled into a raw concrete execution graph.
* **Calibration:** 100 images were passed through the converter to calculate precise integer scale factors.
* **Verification:** The local `.tflite` model achieved **83.33% accuracy** on a strictly isolated verification dataset using integer math clipping.

### Method B: Edge Impulse SDK
To deploy directly to the ESP32 without manually writing a TFLite Micro interpreter:
* **Clean Graph Extraction:** Dynamic training layers (like Data Augmentation) were programmatically stripped.
* **EON Compiler:** The model and calibration data were sent via the Edge Impulse Python SDK to generate a raw C++ library, utilizing hardware-accelerated ESP-NN integer instructions.

## Performance & Benchmark Metrics

* **Float32 Model Validation Accuracy:** 86.24%
* **INT8 Quantized Verification Accuracy:** 83.33% (25 out of 30 isolated verification images correctly classified)
* **Quantized Model Size:** ~33 KB

## 5. Known Limitations & Future Improvements

While the model achieved high accuracy on the isolated test dataset and successfully executes on hardware (~35–60 ms inference time), real-time physical testing revealed a domain gap.

* **Explicit Background Class:** Introduce a fourth training category containing images of empty desks, walls, hands, and ambient noise to eliminate false-positive classifications on empty frames.
* **The Dataset Distribution Shift:** The CNN was trained on high-quality, web-scraped images of fruits. The ESP32-CAM OV2640 sensor produces highly compressed, noisy, and differently lit images. Because the model never learned the specific sensor noise profile of the ESP32-CAM, real-time accuracy degrades heavily. Collect a custom dataset by capturing images of the fruits directly through the ESP32-CAM web server.
* **Quantization-Aware Training (QAT):** Emulate 8-bit precision limits during the training phase to bridge the remaining accuracy gap between floating-point and integer representations.
* **Illumination Control:** Integrate pulse-width modulation control for the onboard high-power flash LED to stabilize brightness levels in low-light environments.
* **Real-Time Web Overlay:** Implement a lightweight web streaming server that overlays model classification predictions and confidence scores onto live camera video.