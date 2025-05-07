import cv2
import numpy as np
import pathlib as pl
from _face_detector.face_detector import FaceDetector
from _system.log_config import setup_logger

class DataProcessor():
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.logger = setup_logger(name=__name__)
    
    def extract_faces_from_dir(self, in_dir:pl.Path, out_dir:pl.Path) -> None:
        detected_faces = []
        for img_path in in_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                img = cv2.imread(filename=img_path)
                collected_faces_from_frame = self.extract_faces_from_image(img=img)
                detected_faces.extend(collected_faces_from_frame)
        detected_faces = np.array(detected_faces)
        self.save_face_samples(detected_faces_samples=detected_faces, out_dir=out_dir)
                
    def extract_faces_from_image(self, img:np.ndarray) -> np.ndarray[np.ndarray]:
        collected_faces = []
        detected_faces = self.face_detector.detect(frame=img)
        for x,y,h,w in detected_faces:
            detected_face = self.crop_face_from_image(img=img, face_coordinates=(x,y,h,w))
            collected_faces.append(detected_face)
        return np.array(collected_faces)
       
    def save_face_samples(self, detected_faces_samples:np.ndarray[np.ndarray], out_dir:pl.Path) -> None:
        for i, sample in enumerate(detected_faces_samples):
            cv2.imwrite(
                    filename=pl.Path(out_dir, f'sample_{i}.jpg'),
                    img=sample
                )
                 
    def crop_face_from_image(self, img:np.ndarray, face_coordinates:np.ndarray[np.ndarray]) -> np.ndarray:
        x,y,h,w = face_coordinates
        cropped_img = img[y:y+h, x:x+w]
        cropped_img = cv2.resize(cropped_img, (128,128))
        return cropped_img
        
    def capture_faces_with_webcam(self, out_dir:pl.Path, max_captures:int=100) -> None:
        cap = cv2.VideoCapture(4)
        detected_faces = []
        i = 0
        while i < max_captures:
            success, frame = cap.read()
            cv2.imshow("Capture", frame)
            key = cv2.waitKey(20)
            match(key):
                case 27:
                    break
                case 99:
                    collected_faces_from_frame = self.extract_faces_from_image(img=frame)
                    if collected_faces_from_frame.size != 0:
                        detected_faces.extend(collected_faces_from_frame)
                        i += 1
        self.save_face_samples(detected_faces_samples=np.array(detected_faces), out_dir=out_dir)
                    