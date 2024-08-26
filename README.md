# MicroAI-Toolbox

This branch contains the implementation of my graduation project "Modeling of Microwave Devices Based on Deep Learning and Interface Design(基于深度学习的微波器件建模及界面开发)" and installation of MicroAI toolbox.
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
Welcome to MicroAI Toolbox! This is a toolbox for training deep learning models, where you can upload your own training dataset for model training. Additionally, we offer an interface for adjusting certain hyperparameters, allowing you to modify aspects such as the number of hidden layers in an MLP, the number of neurons per layer, activation functions, and more according to your training needs. After training is complete, you can upload a testing dataset to evaluate your model. You also have the option to upload a trained model file (.h5 file) for testing or making predictions on inputs.
        

欢迎使用MicroAI Toolbox! 这是一个深度学习模型训练的工具箱，您可以选择上传自己的训练数据集来进行模型的训练。同时，我们提供了调整部分超参数的接口，您可以根据训练需求调整MLP的隐藏层层数、每层的神经元个数、激活函数等。训练完成后您可以上传测试数据集进行对模型的测试。您也可以选择上传训练好的模型文件（.h5文件），从而进行测试或对输入进行预测。




<div align="center">
<img src=https://github.com/qdh-2002/MicroAI-Toolbox/blob/main/img/img/Main_Window.png width="500px">
<div align="left">
        
<p align="center">
  <img src="https://github.com/qdh-2002/MicroAI-Toolbox/blob/main/img/img/Train_window.png" width="300">
  <img src="https://github.com/qdh-2002/MicroAI-Toolbox/blob/main/img/img/Training.png" width="300">
</p>
<p align="center">
  <img src="https://github.com/qdh-2002/MicroAI-Toolbox/blob/main/img/img/test_window.png" width="300">
  <img src="https://github.com/qdh-2002/MicroAI-Toolbox/blob/main/img/img/training_results.png" width="300">
</p>


<div align="left">
        
## Usage

The figure below shows the overall design diagram of MicroAI.  
<div align="center">
<img src=https://github.com/qdh-2002/MicroAI-Toolbox/blob/main/img/img/structure.png width="500px">
<div align="left">

The main interface of MicroAI includes two main buttons: Enter (to access the interface) and Exit (to terminate the program). Clicking on Exit will immediately terminate the entire program.

The figure below shows the menu window that appears after clicking Enter on the main interface. The upper part of the window contains an introduction and usage instructions for MicroAI. The two buttons at the bottom correspond to the two main functions of MicroAI: 1. Training a model based on the training dataset uploaded by the user; 2. Loading the weight files of a pre-saved deep learning model from the local storage and using the loaded model for analysis and result prediction.
<div align="center">
<img src=https://github.com/qdh-2002/MicroAI-Toolbox/blob/main/img/img/Main_Window.png width="500px">
<div align="left">
        
After selecting to train a new model, the system will pop up a window allowing the user to choose the neural network architecture：

<div align="center">
<img src=https://github.com/qdh-2002/MicroAI-Toolbox/blob/main/img/img/Main_Window.png width="500px">
<div align="left">
       
The figure below shows the main interface for model training. The canvas in the upper left corner automatically displays the training loss curve after training is completed; the canvas in the upper right corner automatically shows the comparison curve of the predicted results versus the actual values based on the training dataset, allowing users to quickly get a visual understanding of the model's performance; the canvas in the lower left corner is used to visualize the MLP structure. Once the user inputs parameters such as the number of input neurons, output neurons, number of hidden layers, and number of neurons per layer into the text boxes, the MLP structure can be visualized. After clicking the Apply button, the current MLP model structure is drawn on the canvas. The training information panel in the lower right corner is used to display various model evaluation results after training is completed.

<div align="center">
<img src=https://github.com/qdh-2002/MicroAI-Toolbox/blob/main/img/img/Main_Window.png width="500px">
<div align="left">



## Install

We use Python v3.9.12, Tensorflow v2.9.2 for the implementation.

Step 1. Create an environment and activate it.
```
python3 -m venv my_toolbox
source my_toolbox/bin/activate
```
Step 2. To use the toolbox, install it from source:
```
git clone https://github.com/qdh-2002/MicroAI-Toolbox.git
cd MicroAI-Toolbox
# To install all dependencies listed in the requirements.txt file.
pip install -r requirements.txt
python May8.py
```






## Contributing

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

Small note: If editing the Readme, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.


