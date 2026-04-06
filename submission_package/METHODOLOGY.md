# Auto-Annotation Submission Method Description
**Generated on**: 2026-04-06

## Overview
Our approach focuses on high-precision layout segmentation and baseline extraction for Indic manuscripts (Standard Indic & Ramcharitmanas).

## Technical Implementation
1. **Preprocessing**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and Adaptive Gaussian Thresholding to handle bleed-through and fading common in palm-leaf and aged paper manuscripts.
2. **Layout Parsing**: Utilizes an enhanced contour-based segmentation with horizontal dilation bias to tightly wrap text lines. Includes heuristics for:
   - `marginalia/notes`: Detected via edge-proximity thresholds.
   - `illustration/diagram`: Detected via aspect-ratio and area density checks.
   - `damage/hole`: Detected via irregular blob analysis.
3. **Reading Order**: Implements a "Smart Reading Order" graph that priorities main text flow over marginal components, significantly reducing the "Human Effort Score" (E).
4. **Baselines**: Generates horizontal polylines (baselines) through the geometric center of text regions for Track 2 compliance.

## Reproducibility
The pipeline is modular and can be run using `python main.py <input>`. It includes support for a pre-trained YOLOv8-segmentation weights (`--model`) once fine-tuned on the provided seed set.
