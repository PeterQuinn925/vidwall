import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model/keypoint_classifier/keypoint_classifier.onnx")
landmark = np.array([[0.0, 0.0, -0.242236, -0.049689, -0.440994, -0.142857, -0.596273, -0.229814, -0.645963, -0.322981, -0.366460, -0.409938, -0.552795, -0.490683, -0.590062, -0.422360, -0.590062, -0.335404, -0.279503, -0.490683, -0.434783, -0.695652, -0.559006, -0.782609, -0.670807, -0.850932, -0.167702, -0.521739, -0.217391, -0.751553, -0.279503, -0.888199, -0.341615, -1.0, -0.037267, -0.484472, 0.006211, -0.689441, 0.006211, -0.819876, -0.006211, -0.931677]], dtype=np.float32)

input_name = sess.get_inputs()[0].name
scores = sess.run(None, {input_name: landmark})[0]
print(scores)
