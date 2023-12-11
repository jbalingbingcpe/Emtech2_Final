import streamlit as st
from tensorflow import keras
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import img_to_array
import numpy as np
import os

# Get the absolute path to the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the file name
model_file_path = os.path.join(script_directory, 'model_checkpoint.h5')

# Load the saved model
best_model = keras.models.load_model(model_file_path)

# Function to preprocess an image for prediction
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize the image to match the model's expected input shape
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    return img_array

def create_custom_model(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))  # Input layer

    # Iterate through layers in the original model
    for layer in best_model.layers[1:]:
        if isinstance(layer, keras.layers.Conv2D):
            # Create a new convolutional layer with the same parameters
            new_conv = keras.layers.Conv2D(layer.filters, layer.kernel_size, activation=layer.activation, input_shape=input_shape)
            model.add(new_conv)
        elif isinstance(layer, keras.layers.MaxPooling2D):
            model.add(layer)
        elif isinstance(layer, keras.layers.Flatten):
            model.add(layer)
        elif isinstance(layer, keras.layers.Dense):
            # Create a new dense layer with the same parameters
            new_dense = keras.layers.Dense(layer.units, activation=layer.activation)
            model.add(new_dense)

    # Add a new dense layer with the desired output shape
    model.add(keras.layers.Dense(1, activation='sigmoid', name='custom_dense'))

    return model

def predict(img):
    # Preprocess the image
    preprocessed_image = preprocess_image(img)

    # Create a custom model with the desired architecture
    input_shape = preprocessed_image.shape[1:]
    custom_model = create_custom_model(input_shape)

    # Make predictions using the custom model
    predictions = custom_model.predict(preprocessed_image)

    # Get the predicted class (0 for cat, 1 for dog, assuming 2-class classification)
    predicted_class = np.argmax(predictions)

    return "Cat" if predicted_class == 0 else "Dog"

def main():
    st.title("IS IT A CAT OR DOG?")

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        img = keras_image.load_img(uploaded_file, target_size=(128, 128))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("PREDICT"):
            prediction = predict(img)
            st.write(f"PREDICTION: {prediction}")

if __name__ == "__main__":
    main()
