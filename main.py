from _face_detector import face_detector
from _system.data_processor import DataProcessor
import pathlib as pl
import cv2
def face_detector_main():
    fd = face_detector.FaceDetector()
    frame = cv2.imread(filename=pl.Path('_face_detector', 'test_image.jpg'))
    detected_face_points = fd.detect(frame=frame)
    for x,y,h,w in detected_face_points:
        frame = cv2.rectangle(img=frame, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0),thickness=4)
    
    cv2.imshow(winname='Detected', mat=frame)
    key = None
    while True and key != 27:
        key = cv2.waitKey(20)
        c
def data_processor_main():
    data_processor = DataProcessor()
    data_processor.extract_faces_from_dir(in_dir=pl.Path('_dataset','unprepared_other'), out_dir=pl.Path('_dataset', 'other'))
    data_processor.capture_faces_with_webcam(out_dir=pl.Path('_dataset', 'me'))

if __name__ == "__main__":
    pass
