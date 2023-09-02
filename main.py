import sys
import numpy as np
import tensorflow as tf

def predict(model, img_path, class_names):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    print(img_array)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

model = tf.keras.models.load_model(sys.argv[1])

class_names = [0,1,2,3,4,5,6,7,8,9]

for i in range(2, len(sys.argv)):
    predicted_class, confidence = predict(model,sys.argv[i],class_names)
    print(sys.argv[i], predicted_class, confidence)
