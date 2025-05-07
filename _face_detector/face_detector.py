import cv2
import cv2.data
import numpy as np
from _system.log_config import setup_logger

class FaceDetector():
    
    def __init__(self):
        self.logger = setup_logger(name=__name__)
        self.logger.info(msg=f'FaceDetector <{id(self)}> created')
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
    def detect(self, frame:np.ndarray) -> tuple:
        detected_faces_points = self.face_classifier.detectMultiScale(
            image=frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40,40)
        )
        return detected_faces_points