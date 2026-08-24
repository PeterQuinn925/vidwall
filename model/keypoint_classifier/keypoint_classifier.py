#!/usr/bin/env python

import onnxruntime
import numpy as np
from typing import (
    Optional,
    List,
)

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path: Optional[str] = 'model/keypoint_classifier/keypoint_classifier.onnx',
        providers: Optional[List] = [
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """KeyPointClassifier

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for Palm Detection

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]


    def __call__(
        self,
        landmarks: np.ndarray,
    ):
        """KeyPointClassifier

        Parameters
        ----------
        landmarks: np.ndarray
            Landmarks [N, 42]

        Returns
        -------
        class_ids: np.ndarray
            int64[N]
            ClassIDs of Hand Signatures
        confidences: np.ndarray
            float32[N]
            Confidence (winning class's score) for each prediction.
            When using the stock merged model (argmax fused into the
            graph), raw scores aren't available, so confidence is
            reported as 1.0 for every prediction -- thresholding has no
            effect in that case.
        """
        scores = self.onnx_session.run(
            self.output_names,
            {input_name: landmarks for input_name in self.input_names},
        )[0]

        if scores.ndim == 2:
            # Raw per-class scores (our own tf2onnx + onnxsim conversion,
            # without the argmax-merge step) -- reduce to class indices,
            # and take the winning score itself as the confidence. The
            # model's final op is a Softmax, so these are already proper
            # 0-1 probabilities -- no extra normalization needed.
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1).astype(np.float32)
        else:
            # Already class indices (the stock repo's onnx, which has
            # argmax fused into the graph via make_argmax.py + snc4onnx).
            # No raw scores are available in this case.
            class_ids = scores.astype(np.int64)
            confidences = np.ones_like(class_ids, dtype=np.float32)

        return class_ids, confidences