import time
import warnings
import numpy as np
import onnxruntime as ort


class Executor:
    def __init__(self):
        self.session = None

    def create_session(self, model_path):
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options)
        return self.session

    def run(self, inputs):
        input_feed = {}
        is_scalar = True

        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                input_feed[name] = value.astype(np.float32)
                is_scalar = False
            elif isinstance(value, (list, tuple)):
                input_feed[name] = np.array(value, dtype=np.float32)
                is_scalar = False
            else:
                input_feed[name] = np.array([value], dtype=np.float32)

        result = self.session.run(None, input_feed)[0]

        if is_scalar:
            return float(result.flat[0])
        return result

    def benchmark(self, inputs, runs=1000):
        input_feed = {}
        is_scalar = True

        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                input_feed[name] = value.astype(np.float32)
                is_scalar = False
            elif isinstance(value, (list, tuple)):
                input_feed[name] = np.array(value, dtype=np.float32)
                is_scalar = False
            else:
                input_feed[name] = np.array([value], dtype=np.float32)

        self.session.run(None, input_feed)

        start = time.perf_counter()
        for _ in range(runs):
            self.session.run(None, input_feed)
        end = time.perf_counter()

        total_time = end - start
        avg_time = total_time / runs

        num_elements = 1
        for v in input_feed.values():
            num_elements = max(num_elements, v.size)

        return {
            'total_time': total_time,
            'avg_time': avg_time,
            'runs': runs,
            'runs_per_second': runs / total_time,
            'elements_per_second': (runs * num_elements) / total_time
        }
