# ECEN4493 Final Project - Spring 2025
# Created by Evan Acree

All resources and code for the Oklahoma State ECEN4493 final project can be found here. This project is a YOLOv8-based image classification model which can identify cars used within the Stanford Cars dataset. There are 196 total classes which can be identified, and image submission & prediction is done through a local web interface hosted with Flask in webgui.py.

The two main files are yolo_training.ipynb and webgui.py, which are used to train and interact with
the model. 

The dataset (Stanford Cars Dataset) and its train/test split can be found in the /datasets/ folder.

The latest YOLOv8 model and its weights can be found as /runs/classify/train/weights/last.pt.

All images related to the web-based GUI can be found in /static/, and the .html templates
for the website display can be found in /templates/.

Sources used for reference in creating this project:

https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset - Used as the train/test dataset for the project and defines all classes (cars) available to be identified.

https://github.com/jeffprosise/Deep-Learning - Used as a template to start, but later transferred to a simple YOLO v8 model with Keras.

https://github.com/bhargav-joshi/Image-Classification-on-Flask - Used as guidance for creating the Web UI which interacts with the trained model.
