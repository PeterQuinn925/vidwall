##!/bin/bash
python -m tf2onnx.convert --tflite keypoint_classifier.tflite --output keypoint_classifier.onnx --opset 11
python -m onnxsim keypoint_classifier.onnx keypoint_classifier.onnx
