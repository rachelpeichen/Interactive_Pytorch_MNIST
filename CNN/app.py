
from __future__ import print_function
import argparse
from PIL import Image # This pakage processes images files in Python. ex: read, save, transformation
import torch
from torchvision import transforms
# Import our model
import os 
from .model import Net 

'''Settings'''
package_dir = os.path.dirname(os.path.abspath(__file__)) # Find current directory of file
default_img_path = os.path.join(package_dir,'test_6.png')

parser = argparse.ArgumentParser(description='A simple implement of CNN via PyTorch MNIST dataset', epilog="Please refer the doc: .....")
parser.add_argument('--image', type=str, default=default_img_path, metavar='IMG',
                            help='image for prediction (default: {})'.format(default_img_path))
args = parser.parse_args()

'''Make Prediction'''
# Load model
model_path = os.path.join(package_dir,'mnist_cnn.pt')
model = Net()
model.load_state_dict(torch.load(model_path))

# Load & transform image (pre-treatment)
ori_img = Image.open(args.image).convert('L') # Open images with L mode (8-bit pixels, black & white)
t = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img = torch.autograd.Variable(t(ori_img).unsqueeze(0))
ori_img.close()

'''Predict'''
model.eval()
output = model(img)
pred = output.data.max(1, keepdim=True)[1][0][0]
print('Prediction: {}'.format(pred))