import cv2
import onnxruntime as rt

from pathlib import Path
import yaml
import numpy as np

from PUTDriver import PUTDriver, gstreamer_pipeline


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
 
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

        self.history = []
        self.robot_config = config["robot"]["postprocessing"]

        self.counter = 0
        self.mult = 0

    def preprocess(self, img: np.ndarray, roi=2) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        roi_height = height // roi
        roi = gray[height - roi_height:height, :]
        mean_val = np.mean(roi)
        std_val = np.std(roi)
        lower_threshold = max(0, mean_val - std_val)
        upper_threshold = min(255, mean_val + std_val)
        edges = cv2.Canny(roi, lower_threshold, upper_threshold)
        result = np.zeros((height, width), dtype=np.uint8)
        result[height - roi_height:height, :] = edges
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.dilate(result, kernel, iterations=1)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        result = np.expand_dims(result.transpose(2, 0, 1), axis=0)
        return result

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        print(detections)
        detections = detections[0]
        assert detections.shape == (2,)
        detections[detections > 1.0] = detections[detections > 1.0] * 0 + 1.0
        detections[detections < -1.0] = detections[detections < -1.0] * 0 - 1.0

        if self.robot_config["smooth"]:
            detections = self.postprocess_smoothing(detections, self.robot_config["hist_len"])

        if self.robot_config["speed"] == "mid":
            #if detections[0] > 0.01 and detections[0] < 0.75:
            #    detections[0] = min(np.sqrt(detections[0]), 0.75)
            #elif detections[0] > 0.75:
            #    detections[0] = 0.75
            if detections[0] > 0.0:
                detections[0] = np.sqrt(detections[0])
        elif (self.robot_config["speed"] == "max") and (detections[0] > 0.0):
            detections[0] = 1.0
        if detections[1] < 0.1 and detections[1] > -0.1:
            detections[0] *= 1.4
            if detections[0] > 1:
                detections[0] = 1
        if self.robot_config["centre"]:
            if detections[1] < 0.1 and detections[1] > -0.1:
                detections[1] = 0
                detections[0]
                if detections[0] > 1: detections[0] = 1
            else:
                detections[0]

        self.counter += 1
        if self.counter > 0:
            if self.mult == 0:
                self.mult = 1
            else:
                self.mult = 0
            
            self.counter = 0

        #detections *= self.mult
        
        print(type(detections))
        print(detections)

        return detections
    
    def postprocess_smoothing(self, detections: np.ndarray, hist_len = 4) -> np.ndarray:
        if len(self.history) > max(1000, hist_len - 1):
            self.history = self.history[-(hist_len - 1):]

        self.history.append(detections)
        mult = np.exp(np.arange(1, min(len(self.history), hist_len) + 1, 1, dtype=float))
        mult /= np.sum(mult)
        mult = mult.reshape(-1, 1)

        return np.sum(np.array(self.history[-(min(4, len(self.history))):]) * mult, axis=0, dtype=np.float32)

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)
        
        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, 224, 224)
        
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        print(outputs)
        assert outputs.max() <= 1.0
        assert outputs.min() >= -1.0

        return outputs


def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    # model warm-up
    ret, image = video_capture.read()
    if not ret:
        print(f'No camera')
        return
    
    _ = ai.predict(image)

    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    counter = 0
    import time
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print(f'No camera')
            break
        counter += 1
        if (counter % 10) != 0:
            continue
        start = time.time()
        forward, left = ai.predict(image)
        print(time.time() - start)

if __name__ == '__main__':
    main()
