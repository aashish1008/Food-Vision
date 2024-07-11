# Food Vision App
Food Vision is a deep learning application implemented using transfer learning with the EfficientNetV2B0 model. This project leverages TensorFlow and Keras to classify images of various food items.

## Overview
The purpose of this project is to build a robust food classification model using state-of-the-art transfer learning techniques. By using EfficientNetV2B0, the model is able to achieve high accuracy with relatively low computational cost.

## Technologies Used
- Python 3.10 
- TensorFlow 2.15.0
- Keras
- EfficientNetV2B0
- Streamlit

## Project Structure
- `model/` : Directory containing the trained model of the food vision dataset.
- `README.md` : Project documentation.
- `food-vision.ipynb` : Jupyter notebook demonstrating the implementation and training of the model.
- `main.py` : Main script for running the model.
- `requirements.txt` : List of required Python packages.

## Installation
1. Clone the repository:
   ``` bash
   git clone https://github.com/aashish1008/Food-Vision.git
   cd Food-Vision
2. Install the required packages:
   ``` bash
   pip install -r requirements.txt

## Usage
1. **Running the Jupyter Notebook :**
   Open and run the `food-vision.ipynb` notebook to see step-by-step implementation details, including data preprocessing, model training, and evaluation.

2. **Running the Script :**
   To classify images using the pre-trained model, run the `main.py` script
   ``` bash
   streamlit run main.py
   
## Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss your ideas.
