import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image, ImageOps
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)
loss,accuracy=model.evaluate(x_test,y_test)
print("Test Accuracy",accuracy)
index=99
prediction=model.predict(x_test[index:index+1])
predicted_digit=tf.argmax(prediction[0]).numpy()
plt.imshow(x_test[index],cmap='gray')
plt.title("Predicted Digit" +str(predicted_digit))
plt.axis('off')
plt.show()
img = Image.open("my_digit.jpeg").convert("L")
img = ImageOps.invert(img)
img = img.crop(img.getbbox())
img = img.resize((28, 28))
plt.imshow(img, cmap="gray")
plt.title("Processed Image")
plt.axis("off")
plt.show()
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 28, 28)
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction[0])
print("Your handwritten digit is:", predicted_digit)
print("Actual Digit",2)
print("Predicted Digit",predicted_digit)
