import os
import numpy as np
import onnxruntime as ort


class ONNXEngine:
    """
    ONNX Runtime Inference Engine wrapper optimized for TensorRT FP16
    and 512MB max workspace allocation for Jetson Nano.
    """

    def __init__(self, model_path, cache_dir="./trt_cache"):
        os.makedirs(cache_dir, exist_ok=True)

        trt_options = {
            'device_id': 0,
            'trt_max_workspace_size': 536870912,  # 512 MB
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,
        }

        providers = [
            ('TensorrtExecutionProvider', trt_options),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"💡 ONNXEngine running on execution provider: {self.session.get_providers()[0]}")

    def infer(self, input_tensor):
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0]
