## Interactive PyTorch MNIST
:round_pushpin: This program modified from the [official PyTorch MNIST example](https://github.com/pytorch/examples/blob/master/mnist/main.py).  
### :file_folder: Files in CNN folder:
Name | Details
------------ | -------------
model.py | Define a convolutional neural network and how we train & test it.
app.py | Interactive predictor for users.
mnist_cnn.pt| This pre-trained model will be overriden when you start training.
test_n.png| Sample images for using interactive predictor, and you can also try your images.

### :wrench: How to use:
### :small_orange_diamond: _**Windows**_
 Prerequisite:  Python version 3.8 64 bit is preferable. _**(Note: PyTorch doesn't work with 32 bit)**_
 ### 1. clone project
 - If you have installed `Git` already, just enter below command in terminal.
 ```
 $ git clone https://github.com/rachelpeichen/Interactive_Pytorch_MNIST.git
  ```
  
 - You can also download the zip file and extract it without using `Git`:
 
 ```
 $ cd "download file path"
 ```  
 ### 2. install dependencies
 - Please go to [PyTorch site](https://pytorch.org/get-started/locally/) to select your preferences and run the install command. 
 - You can refer to the install command I use:
 ```
 $ pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
 ```
  - _Please be noted that I don't install CUDA because my computer doesn't have CUDA._
### 3. train & test model
 - Enter below command in terminal:
 ```
 $ python -m CNN.model
 ```
 - You may encounter this error : C++ IS NOT ..., please download and executate [this file](https://aka.ms/vs/16/release/vc_redist.x64.exe) from Microsift website and try above command again.
 - **Please be noted that the model you trained will be auto-saved, if you want to change the training settings, type below command in terminal:**
```
$ -m CNN.model help

`usage: model.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
                [--lr LR] [--gamma M] [--no-cuda] [--dry-run] [--seed S]
                [--log-interval N] [--save-model]`
                
$ --epochs N (N is the number of epochs to train, default = 10)

$ --save-model (defalut = True)
```
### 4. test model interactively
```
$ python3 -m CNN.app --image="the path of tested image"
```
  
  
    
### :small_orange_diamond: _**macOS**_
### 1. clone project
```
$ git clone https://github.com/rachelpeichen/Interactive_Pytorch_MNIST.git
    
$ cd Interactive_Pytorch_MNIST
```
### 2. install dependencies
```
$ pip3 install -r mac_requirements.txt
```
### 3. train & test model
```
$ python3 -m CNN.model
```
 - **Please be noted that the model you trained will be auto-saved, if you want to change the training settings, type below command in terminal:**
```
$ -m CNN.model help

usage: model.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
                [--lr LR] [--gamma M] [--no-cuda] [--dry-run] [--seed S]
                [--log-interval N] [--save-model]
                
$ --epochs N (N is the number of epochs to train, default = 10)

$ --save-model (defalut = True)
```
### 4. test model interactively
```
$ python3 -m CNN.app --image="the path of tested image"
```
