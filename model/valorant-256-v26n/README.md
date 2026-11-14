# Valorant 256 YOLO26n INT8

`model.tflite` is an unmodified model from the open-source Auto-aim_android-yolo repository, made available as an optional model-store download rather than an APK asset. It is intended for the supported Valorant appearance described by its source; its training data, validation set and enemy/team separation have not been independently audited.

- Source: https://github.com/xiangsu1145/Auto-aim_android-yolo/blob/1e3ad894a11aad3503efaec5e03acc58920bbb3c/app/src/main/assets/valorant_256_v26n.tflite
- Source revision: `1e3ad894a11aad3503efaec5e03acc58920bbb3c`.
- Model metadata: https://github.com/xiangsu1145/Auto-aim_android-yolo/blob/master/app/src/main/assets/models.json
- Source license: GNU GPL-3.0, supplied verbatim in `LICENSE`. No separate weights-specific license was found.
- Input: `[1, 256, 256, 3]`, INT8 NHWC RGB; scale 0.003921568859368563, zero point -128. This corresponds to pixels scaled from 0 to 1.
- Output: `[1, 6, 1344]`, INT8; scale 0.004291026387363672, zero point -128. After dequantization, each channels-first anchor is a normalized centre-form box followed by `body` and `head` scores. The raw anchors require NMS.
- File size: 2,716,654 bytes.
- SHA-256: `075372fb054a6ef52bca87977645294d74ecdaaa1e16c9bf9e6c89da5fd7740d`.

The file's TFLite FlatBuffer, tensor types, dimensions and quantization were inspected with LiteRT. Host inference completed and produced finite dequantized output. The coordinate and score conventions are confirmed by the upstream `litert_engine.cpp` decoder, which reads channels-first dequantized boxes and per-class probabilities in the same order as this app. The supplied `model.json` states them explicitly so a blank loading probe cannot choose the wrong scale.

No accuracy claim has been verified on labelled game captures. No physical Android device, GPU/NNAPI delegate, inference latency or end-to-end aiming result has been measured. The model's small file and INT8 representation do not by themselves establish sub-12 ms execution or superiority to other models.
