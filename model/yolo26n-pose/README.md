# YOLO26n Pose

`model.tflite` is a float32 conversion of the official Ultralytics `yolo26n-pose` ONNX model. It is an optional download, not an APK asset. Its general COCO person-pose training is not specific to a game and does not make every detected person an enemy in every game.

- Source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.onnx
- License: GNU AGPL-3.0, supplied in `LICENSE` and at https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Input: `[1, 640, 640, 3]`, float32 NHWC RGB, scaled from 0 to 1.
- Output: `[1, 300, 57]`, float32: corner-form box, person score, class, then 17 COCO `(x, y, confidence)` triples. Coordinates are model pixels; the export already selects final detections.
- File size: 12,233,156 bytes.
- SHA-256: `2ff48db73e247675a77bfe2fb2b1ab4d3d88cb6584402f428fb210f907f226ce`.

The model was converted locally with `onnx2tf` and verified on the host using LiteRT. On the Ultralytics bus photograph, four people had a person score above 0.25. Those retained ONNX and TFLite rows agreed to a maximum absolute difference of 0.000184. Low-confidence rows can differ substantially, so the store entry must be regenerated if the model is reconverted; the large difference in discarded rows is not a retained-detection accuracy measurement.

The float16 conversion was not selected: its input is float16, whereas the Android runner accepts float32 or quantized integer input. No physical-device latency or GPU/NNAPI result has been measured. Do not describe this model as guaranteed sub-12 ms or as the fastest model without a device benchmark.
