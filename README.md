# Interactive Pytorch MNIST
This program modified from the [official PyTorch MNIST example](https://github.com/pytorch/examples/blob/master/mnist/main.py).


### Files:
 * CNN
   * model.py        Define a cnn model and how we train & test it.
   * app.py.         Interactive predictor
   * mnist_cnn.pt    Pretrained model, this model will be overriden when you start training.
   * test_n.png      Sample images for using interactive predictor, and you can also try using your own images
 
#### How to use:

#### **Windows**

 Python version: 3.8 64 bit (can't higher and Pytorch doesn't work with 32 bit)
 
 #### 1. clone project
 - If you have installed `git` already: `$ git clone https://github.com/rachelpeichen/Interactive_Pytorch_MNIST.git`
 
 - You can download the zip file and extract it: `$cd "file path"`
 
 #### 2. install dependencies
 go to Pytorch site https://pytorch.org/get-started/locally/
 
 `pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`
 
 we don't install CUDA because we don't have CUDA

#### 3. train & test model

`$ python -m CNN.model`
You may encounter this error : C++ IS NOT ......

Download and executate this should be OK https://aka.ms/vs/16/release/vc_redist.x64.exe 

**Please be noted that the model you trained will be auto-saved, if you want to change the training settings, please type below command in terminal:**

`-m CNN.model help`

`usage: model.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
                [--lr LR] [--gamma M] [--no-cuda] [--dry-run] [--seed S]
                [--log-interval N] [--save-model]`
                
`--epochs N (N is the number of epochs to train, default = 10)`

`--save-model (defalut = True)`


#### 4. test model interactively

`$ python3 -m CNN.app --image="the path of tested image"`



 - **For macOS:** 
#### 1. clone project
  
`$ git clone https://github.com/rachelpeichen/Interactive_Pytorch_MNIST.git`
    
`$ cd Interactive_Pytorch_MNIST`


#### 2. install dependencies

 
`$ pip3 install -r mac_requirements.txt`


#### 3. train & test model

`$ python3 -m CNN.model`

**Please be noted that the model you trained will be auto-saved, if you want to change the training settings, please type below command in terminal:**

`-m CNN.model help`

`usage: model.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
                [--lr LR] [--gamma M] [--no-cuda] [--dry-run] [--seed S]
                [--log-interval N] [--save-model]`
                
`--epochs N (N is the number of epochs to train, default = 10)`

`--save-model (defalut = True)`


#### 4. test model interactively

`$ python3 -m CNN.app --image="the path of tested image"`


