
# Privacy Handling for Visual Question Answering

## Overview
Evidence images are processed before being sent to the configured Gemini model.

## Privacy processing
1. Original evidence images are read from the run directory.
2. Processed copies are created in the privacy_processed directory.
3. OpenCV face detection and blurring are applied.
4. A heuristic is used to detect and blur possible vehicle registration plates.
5. The Gemini client receives the processed image copies.

## Limitations
- Face detection may fail for faces at unusual angles, in poor lighting or at low resolution.
- The plate-detection heuristic may miss registration plates or blur unrelated regions.
- The processing does not guarantee complete anonymisation.

## Safeguards
- Original images should not be sent to the hosted model.
- Processed images must be visually inspected before external transmission.
- Images containing identifiable information that remains visible should not be sent.
- API credentials must be stored in environment variables and excluded from version control.
