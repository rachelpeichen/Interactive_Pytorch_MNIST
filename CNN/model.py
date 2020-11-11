from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F # For ReLu & Pooling
import torch.optim as optim # Optimize parameters
from torchvision import datasets, transforms # Torchvision is a package for images
from torch.optim.lr_scheduler import StepLR
#
# (1) Define a cnn model
#
class Net(nn.Module):  # Inherit from parent class "nn.Module", and we only need to define __init__ & forward.
    def __init__(self): # Define the parameters for out network
        super(Net,self).__init__()

        # Define two 2D convolutional layers (1 x 32 and 32 x 64 with kernel size of 3x3)
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)

        # Define dropout layers
        self.dropout1 = nn.Dropout(0.25) # Probability of an element to be zeroed
        self.dropout2 = nn.Dropout(0.5)

        # Define fully connected layers
        self.fc1 = nn.Linear(9216, 128) # (Input features, Output features)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x): # Define the network's transformation; it's the mapping that maps an input tensor to a prediction output tensor
        
        x = self.conv1(x)
        x = F.relu(x) # Activate function: transform the summed weight input into output 

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2) # Pooling layers reduce the dimensions of the data, typically 2 X 2 window
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        # Apply the log_softmax function to an n-dimensional input tensor, re-scale all slices along dim in the range(0,1) and sum to 1
        output = F.log_softmax(x, dim=1) 
        return output

    # Input image size: 28x28, input channel: 1 ; bacth size: defaut is 64 
    # Input (batch size x 1 x 28 x 28) -> Conv1 (64 x 32 x 24 x 24) -> ReLu  -> Conv2(64 x 64 x 20 x 20) -> ReLu ->...
    # ... -> MaxPooling(64 x 64 x 10 x 10) -> Dropout1 ->  Flatten1(64 x 9216) -> FC1 (64 x 128) --> Dropout2 -> FC2 (64 x 10) 

#
# (2) Define how we train our model
#
def train(args, model, device, train_loader, optimizer, epoch): # Device: CPU or GPU
    model.train() # State that we are training the model

    for batch_index, (data, target) in enumerate(train_loader): # Iterate over train data batches
        
        data, target = data.to(device), target.to(device) # Data are images, target is label
        
        # Clear the gradients
        optimizer.zero_grad()
        
        # Forward propagation
        output = model(data)

        # Calcualte the negative log likelihood loss
        loss = F.nll_loss(output, target)

        # Backpropragate(backprop) to calculate gradients, backprop is a widely used algorithm in training feedforward neural networks for supervised learning
        loss.backward() 

        # Optimizer implement step() method to update the gradients
        optimizer.step()

        # Print loss messages for each training epoch
        if batch_index % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_index * len(data), len(train_loader.dataset),
                    100. * batch_index / len(train_loader), loss.item()))
        if args.dry_run:
            break
#
# (3) Define how we test our model
#
def test(model, device, test_loader):
    model.eval() # State that we are testing the model, amd we only do forward no backward when testing

    test_loss = 0 # Initialize loss & correct predicting accumulators
    correct = 0

    with torch.no_grad(): # The codes within "with" don't backward and update the gradients
        
        for data, target in test_loader: # Iterate over testing data batches
            
            data, target = data.to(device), target.to(device)
            
            # Retrieve
            output = model(data)
            
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)  
            
            # Increment correct predicting accumulator if correct
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Average test loss
    test_loss /= len(test_loader.dataset)
    
    # Print average test loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 
# (4) Implement the model
#
def main():
    # Set training settings from command line or use defalut settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For saving the current model')
    args = parser.parse_args()

    # Check if we can run in GPU since Pytorch allows accelarting by GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Provide seed for the pseudorandom number generator s.t. the same results can be reproduced
    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Fetch data
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    # Choose Adadelta as the optimizer, initialize it with the parameters & settings
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Decay the learning rate of each parameter group by gamma every step_size epochs
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # Train and test our model
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # Save the model for future use
    if args.save_model:
        torch.save(model.state_dict(), "CNN/mnist_cnn.pt")

if __name__ == '__main__':
    main()