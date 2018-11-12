import numpy as np
import cv2
from skimage.feature import hog
# from utils import rgb2gray
import sys

# cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # features = hog(gray, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL,
    #                cells_per_block=CELLS_PER_BLOCK, visualise=VISUALISE, normalise=NORMALISE)
    features = hog(gray, visualise=True)
    print(features)

    # hog = cv2.HOGDescriptor()
    # hog.setSVMDetector(cv2.HOGDescriptor_detect())
    # rects, weights = hog.detectMultiScale(frame, winStride=(5, 5), padding=(16, 16), scale=1.05,
    #                                         useMeanshiftGrouping=False)
    # print(rects, weights)




    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    # )

    # Draw a rectangle around the faces
    # for (x, y, w, h) in features:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     print("Face location " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))

    # Display the resulting frame
    cv2.imshow('Video', frame)
    for image in features:
        cv2.imshow('feature', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if cv2.waitKey(1) & 0xFF == ord('c'):
    #     cframe = frame
    #     video_capture.release()
    #     cv2.imshow('freezeframe', cframe)

video_capture.release()
cv2.destroyAllWindows()
