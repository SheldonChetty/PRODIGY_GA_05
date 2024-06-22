# **Neural Style Transfer**
This project demonstrates how to perform neural style transfer using TensorFlow. Neural style transfer is a technique that combines the content of one image and the style of another image to create a new image that blends both content and style. This project will guide you through setting up the environment, installing the necessary packages, and running a script to perform neural style transfer.
## Introduction
Neural style transfer is an exciting application of deep learning where a model is used to blend the content of one image with the style of another. By leveraging TensorFlow, we can easily implement and experiment with neural style transfer.
### The applications of neural style transfer include:
•	Artistic creation  
•	Photo enhancement  
•	Creating new visual content for media  
# Example 
**Content Image:**

![tubingen](https://github.com/SheldonChetty/PRODIGY_GA_05/assets/118753773/4633ccec-7dfc-46aa-ae65-00d1164bdf09)

**Style Image:**

![seated-nude](https://github.com/SheldonChetty/PRODIGY_GA_05/assets/118753773/7f31d06a-2894-401c-9c1a-865dff4754ec)



**This is the resultant combined image of the Content and Style Images:**

![tubingen-style](https://github.com/SheldonChetty/PRODIGY_GA_05/assets/118753773/b75905a5-ed18-41b7-b1f0-6d42c7c60248)

## In this project, we will:
•	Set up a Python virtual environment.  
•	Install the necessary dependencies.  
•	Write a Python script to perform neural style transfer.  
•	Execute the script to create and save the styled image.  

# Let's get started!  
**Prerequisites**
•	Download and install Anaconda (recommended).  

## Setup
1. Create a Folder and Activate a Virtual Environment  
First, we need to create a folder for the project and set up a Python virtual environment in it to manage our project's dependencies. We'll use conda for this purpose:
#Create a folder  
*mkdir neural_style_transfer*  
*cd neural_style_transfer*  

#Create a virtual environment  
*conda create --name nst_env python=3.8*  

#Activate the virtual environment  
*conda activate nst_env*  

2. Install Required Packages  
Next, we'll install the necessary Python packages:  
•	tensorflow  
•	matplotlib  
•	numpy  
•	Pillow  
### You can install these dependencies using the following command:  
*pip install -r requirements.txt*  

3. Obtain Content and Style Images  
You need two images for this project:  
•	A content image (the image whose content you want to keep).  
•	A style image (the image whose style you want to apply to the content image).  
Save these images in your project directory with the names content.jpg and style.jpg.

4. Run the Neural Style Transfer Script  
Create a Python script named neural_style_transfer.py in your project folder and add the provided code. This script will perform neural style transfer on the content and style images you have prepared.  
(Go to NST_Code.py file)  

# Notes
•	Ensure your environment is properly set up with all dependencies installed.  
•	Experiment with different content and style images to explore various artistic effects.  



