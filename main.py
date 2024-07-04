import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time


class FoodVision:
    def __init__(self, img=None):
        self.img = img
        # Define your food categories
        self.food_categories = [
            'chicken_curry',
            'chicken_wings',
            'fried_rice',
            'grilled_salmon',
            'hamburger',
            'ice_cream',
            'pizza',
            'ramen',
            'steak',
            'sushi'
        ]
        # Load the trained model
        self.model = tf.keras.models.load_model('model/food_vision_model.h5')

    # Function to preprocess the uploaded image
    def preprocess_image(self):
        img = self.img.resize((224, 224))
        img_array = np.asarray(img)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img = img.reshape(1, 224, 224, 3)
        return img

    # Function to make predictions using the loaded model
    def predict(self):
        img = self.preprocess_image()
        prediction = self.model.predict(img)
        predicted_label_idx = np.argmax(prediction)
        predicted_label = self.food_categories[predicted_label_idx]
        return predicted_label

    # Main function for Streamlit app
    def run(self):
        st.set_page_config(page_title="Food Vision App")
        st.markdown("""
                <div style="display: flex; justify-content: center;">
                    <img src="https://png.pngtree.com/png-vector/20220705/ourmid/pngtree-food-logo-png-image_5687686.png" alt="Logo" width="150" style="border-radius: 20px;">
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>Food Vision</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Discover the food in your image!</h3>", unsafe_allow_html=True)

        st.markdown("<h4 style='text-align: center;'>Upload an image and let the app identify your meal.</h4>",
                    unsafe_allow_html=True)

        # File uploader widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            self.img = image
            # st.image(image, caption="Uploaded Image", use_column_width=True)

            # Classify button
            if st.button("Classify"):
                progress_text = "Classifying the image. Please wait..."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                my_bar.empty()
                label = self.predict()
                st.success(f"Great! Today, You are having : {label}")


# App entry point
if __name__ == "__main__":
    model = FoodVision()
    model.run()
