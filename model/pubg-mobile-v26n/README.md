# PUBG Mobile YOLO26n

`model.tflite` is a float32 TFLite export of an Ultralytics YOLO26n network trained on the project's own PUBG Mobile dataset (`/datasets/pubg-mobilev1iyolo26-1-1c2d0e8d/data.yaml` at training time). It detects one class, `head-person`, which this store maps to the `body` role: the box is outlined, tracked and aimed at.

- Source: trained and exported with the project training pipeline; embedded metadata records Ultralytics 8.4.212, exported 2026-11-14.
- Weights license: AGPL-3.0 (`LICENSE`), matching the project license.
- Input: `[1, 3, 640, 640]` float32, channels-first, pixels scaled 0 to 1. The app's TFLite loader detects the channels-first layout itself and feeds normalized pixels.
- Output: `[1, 300, 6]` float32, end-to-end: each row is a corner-form box in model pixels followed by score and class. No NMS in the app is needed.
- File size: 9,859,196 bytes.

The file's TFLite FlatBuffer, tensor types, dimensions and quantization were inspected with LiteRT. Host inference completed and produced finite output; a blank warm-up gives only near-zero scores, so the end-to-end layout cannot be mistaken for raw anchors. The coordinate and score conventions follow the same Ultralytics end-to-end export as the `yolo26n-pose` model.

No accuracy claim has been verified on labelled game captures. No physical Android device, GPU/NNAPI delegate or inference latency has been measured; a float32 640-pixel model is heavier than the INT8 256-pixel `valorant-256-v26n` and needs device testing before latency is promised.
