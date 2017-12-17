# Chandru, Jeyanth
# 1001-359-339
# 2017-11-27
# Assignment_06_02

from keras.models import model_from_json
import mxnet as mx
from Chandru_06_04 import MtcnnDetector
import numpy as np
import os
import cv2

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

directory = "Images"
class_images = os.listdir(directory)


json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
directory = "Images"
class_images = os.listdir(directory)


def load_images_from_folder(folder):
    prediction = []
    for i, filename in enumerate(os.listdir(folder)):
            if filename.endswith(".DS_Store"):
                continue
            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(folder, filename))
                img = cv2.resize(img, (224,224))
                if img is not None:
                    img1 = img / 255.0
                    img1 = np.expand_dims(img1, 0)
                    predictions = loaded_model.predict(img1)
                    max_image = np.argmax(predictions)
                    prediction.append(max_image)
                    class_read_image = cv2.imread(os.path.join(directory, class_images[max_image]))
                    im2 = class_read_image.copy()
                    results = detector.detect_face(img)

                    if results is None:
                        continue

                    total_boxes = results[0]
                    points = results[1]

                    draw = img.copy()
                    for b in total_boxes:
                        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0))
                    cv2.imshow("Test Images VS Predicted", np.hstack((draw, im2)))
                    cv2.waitKey(0)

            else:
                for file in os.listdir(os.path.join(folder, filename)):
                    if file.endswith(".DS_Store"):
                        continue
                    if file.endswith(".jpg"):
                        img = cv2.imread(os.path.join(folder, filename, file))
                        img = cv2.resize(img, (224, 224))
                        if img is not None:
                            img1 = img / 255.0
                            img1 = np.expand_dims(img1, 0)
                            predictions = loaded_model.predict(img1)
                            max_image = np.argmax(predictions)
                            prediction.append(max_image)
                            class_read_image = cv2.imread(os.path.join(directory, class_images[max_image]))
                            im2 = class_read_image.copy()
                            cv2.namedWindow("Test Images VS Predicted")
                            results = detector.detect_face(img)

                            if results is None:
                                continue

                            total_boxes = results[0]
                            points = results[1]

                            draw = img.copy()
                            for b in total_boxes:
                                cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0))
                            cv2.imshow("Test Images VS Predicted", np.hstack((draw,im2)))
                            cv2.waitKey(300)

input_test_image = input("Enter the full path for Test Images : ")
load_images_from_folder(input_test_image)
print("Open the CV2 Display window and hit Enter to continue...")
cv2.waitKey(0)

# cv2.destroyAllWindows()

camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture('Video.mp4')

while True:
# while(camera.isOpened()):
    grab, frame = camera.read()
    img = cv2.resize(frame, (224,224))

    results = detector.detect_face(img)

    if results is None:
        continue

    total_boxes = results[0]
    points = results[1]

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0))
    chips = detector.extract_image_chips(img, points, 144, 0.37)
    chips = cv2.resize(chips[0], (224,224))
    img1 = img / 255.0
    img1 = np.expand_dims(img1, 0)
    predictions = loaded_model.predict(img1)
    pred = sorted(predictions[0],reverse=True)
    max_images = np.argsort(-predictions[0])
    class_read_image = cv2.imread(os.path.join(directory, class_images[max_images[0]]))
    class_read_image1 = cv2.imread(os.path.join(directory, class_images[max_images[1]]))
    class_read_image2 = cv2.imread(os.path.join(directory, class_images[max_images[2]]))
    cv2.putText(class_read_image,str(pred[0]),(10,200),font,2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(class_read_image1,str(pred[1]),(10,200),font,2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(class_read_image2,str(pred[2]),(10,200),font,2,(255,255,255),2,cv2.LINE_AA)

    vimstack = np.vstack((class_read_image, class_read_image1))
    vimstack = np.vstack((vimstack, class_read_image2))
    imstack = np.hstack((draw, chips))
    imstack = cv2.resize(imstack, (672,672))
    imstack = np.hstack((imstack,vimstack))
    cv2.imshow("Assignment 10_Chandru", imstack)
    cv2.waitKey(30)

camera.release()
cv2.destroyAllWindows()