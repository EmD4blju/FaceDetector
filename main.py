from _face_detector.face_detector import FaceDetector
from _system.data_processor import DataProcessor
from _model.model_init import ModelManager
import pathlib as pl
import cv2
import numpy as np

model_manager = ModelManager()
data_processor = DataProcessor()

def face_detector_main():
    fd = FaceDetector()
    frame = cv2.imread(filename=pl.Path('_face_detector', 'test_image.jpg'))
    detected_face_points = fd.detect(frame=frame)
    for x,y,h,w in detected_face_points:
        frame = cv2.rectangle(img=frame, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0),thickness=4)
    
    cv2.imshow(winname='Detected', mat=frame)
    key = None
    while True and key != 27:
        key = cv2.waitKey(20)
        
def data_processor_main():
    data_processor = DataProcessor()
    data_processor.extract_faces_from_dir(in_dir=pl.Path('_dataset','unprepared_other'), out_dir=pl.Path('_dataset', 'dataset'), file_name_prefix='other')
    data_processor.capture_faces_with_webcam(out_dir=pl.Path('_dataset', 'dataset'), file_name_prefix='me')
    
def model_manager_main():
    model_manager = ModelManager()
    data_processor = DataProcessor()
    model_manager.set_dataset(dataset=data_processor.load_labeled_face_samples(in_dir=pl.Path('_dataset','dataset')))
    model_manager.init_model()
    model_manager.train_model()
    model_manager.save_model()
    
def video_face_capture():
    model = ModelManager.load_model()
    face_detector = FaceDetector()
    data_processor = DataProcessor()
    model.summary()
    cap = cv2.VideoCapture(4)
    key = None
    while key != 27:
        key = cv2.waitKey(10)
        _, frame = cap.read()
        detected_faces_points = face_detector.detect(frame=frame)
        edited_frame = frame.copy()
        for x,y,h,w in detected_faces_points:
            edited_frame = cv2.rectangle(img=frame, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0), thickness=4)
            face = data_processor.crop_face_from_image(img=edited_frame, face_coordinates=(x,y,h,w))
            face = np.expand_dims(face, axis=0)
            prediction = np.argmax(model.predict(face, verbose=0))
            class_labels = ['me', 'other']
            predicted_label = class_labels[prediction]
            edited_frame = cv2.putText(img=frame, text=f'Predicted:{predicted_label}', org=(x-10,y-10), fontFace=2, fontScale=0.5, color=(252, 169, 3), thickness=2)
        cv2.imshow(winname='Face Recognition', mat=edited_frame)
        
    
    
    

if __name__ == "__main__":
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    # data_processor_main()
    # model_manager_main()
    video_face_capture()
    
  
