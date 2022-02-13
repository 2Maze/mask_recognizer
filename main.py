import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# choose camera or webcamera
cap = cv2.VideoCapture(0)

mask_label = {0:'MASK', 1:'NO MASK'}

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)

    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        crop = image[y:y + h, x:x + w]
        crop = cv2.resize(crop, (150, 150))
        crop = np.reshape(crop, [1, 150, 150, 3]) / 255.0
        mask_result = model.predict(crop)

        if mask_result.argmax() == 1:
            # without mask
            cv2.putText(image, mask_label[mask_result.argmax()], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (66, 76, 235), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (66, 76, 235), 3)
        elif mask_result.argmax() == 0:
            # with mask
            cv2.putText(image, mask_label[mask_result.argmax()], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (120, 200, 80), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (120, 200, 80), 3)
    cv2.imshow('img', image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()