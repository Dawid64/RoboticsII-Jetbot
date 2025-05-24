import numpy as np

history = []

# a = np.array([0.5, 1.3, 5.0, -10.0])
# a[a > 1] = a[a > 1] * 0 + 1.0
# print(a)

def postprocess_with_smoothing(detections: np.ndarray, hist_len = 4) -> np.ndarray:
    global history
    if len(history) > max(5, hist_len - 1):
            history = history[-(hist_len - 1):]

    history.append(detections)
    print(history)
    mult = np.exp(np.arange(1, min(len(history), hist_len) + 1, 1, dtype=float))
    mult /= np.sum(mult)
    mult = mult.reshape(-1, 1)
    print(mult)

    return np.sum(np.array(history[-(min(4, len(history))):]) * mult, axis=0)

for i in range(10):
    print(postprocess_with_smoothing(np.array([i + 1, i * 2])))

# print(np.multiply(np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1], [2], [3]])))