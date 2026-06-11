---
title: Deepfake Detector
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# deepfake-detector

**Beyond Binary Detection — XAI Deepfake Localization via Frequency-Aware Detection**

A research prototype that detects AI-generated images and explains *where* in the image the evidence was found using Grad-CAM heatmaps and DCT frequency maps.

## How It Works

Every uploaded image goes through two parallel streams:
- **RGB stream** — spatial textures, blending artefacts
- **DCT stream** — frequency fingerprints of AI generation

## Training
- CIFAKE (diffusion objects)
- FaceForensics++ C23 (6 manipulation types)
- Zero-shot tested: ArtiFact (25 generators)
