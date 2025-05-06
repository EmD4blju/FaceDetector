from _face_detector import face_detector
import pathlib as pl
import cv2
def main():
    fd = face_detector.FaceDetector()
    frame = cv2.imread(filename=pl.Path('_face_detector', 'test_image.jpg'))
        
    detected_face_points = fd.detect(frame=frame)
    for x,y,h,w in detected_face_points:
        frame = cv2.rectangle(img=frame, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0),thickness=4)
    
    cv2.imshow(winname='Detected', mat=frame)
    key = None
    while True and key != 27:
        key = cv2.waitKey(20)

if __name__ == "__main__":
    main()
