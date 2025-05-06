import cv2
import cv2.data
import numpy as np
from _system import log_config

class FaceDetector():
    
    def __init__(self):
        self.logger = log_config.setup_logger(name=__name__)
        self.logger.info(msg=f'FaceDetector <{id(self)}> created')
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
    def detect(self, frame:np.ndarray) -> tuple:
        self.logger.info(msg=f'Detecting on frame: {frame.shape}')
        detected_face_points = self.face_classifier.detectMultiScale(
            image=frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40,40)
        )
        self.logger.info(msg=f'Face found on: {detected_face_points}')
        return detected_face_points