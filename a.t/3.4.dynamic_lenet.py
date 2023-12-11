# import the modules used here in this recipe
import torch
import torch.quantization
import torch.nn as nn
import copy
import os
import time
import torch.nn.functional as F

# define a very, very simple LSTM for demonstration purposes
# in this case, we are wrapping ``nn.LSTM``, one layer, no preprocessing or postprocessing
# inspired by
# `Sequence Models and Long Short-Term Memory Networks tutorial <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html`_, by Robert Guthrie
# and `Dynamic Quanitzation tutorial <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`__.
if __name__ == '__main__':

    class Net(nn.Module):

        def __init__(self): #here you define the functions you use in the forward section
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.conv3 = nn.Conv2d(16,120,5)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(120 * 1 * 1, 84)
            self.fc2 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.tanh(self.conv1(x)))
            x = self.pool(F.tanh(self.conv2(x)))
            x = F.tanh(self.conv3(x))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.tanh(self.fc1(x))
            x = self.fc2(x)
            return x
        

    float_lenet = Net()
    float_lenet.load_state_dict(torch.load('./models/lenet.pth'))
    torch.manual_seed(29592)  # set the seed for reproducibility

    # this is the call that does the work
    quantized_lstm = torch.quantization.quantize_dynamic(
        float_lenet, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )

    # show the changes that were made
    print('Here is the floating point version of this module:')
    print(float_lenet)
    print('')
    print('and now the quantized version:')
    print(quantized_lstm)

    def print_size_of_model(model, label=""):
        torch.save(model.state_dict(), "temp.p")
        size=os.path.getsize("temp.p")
        print("model: ",label,' \t','Size (KB):', size/1e3)
        os.remove('temp.p')
        return size

    # compare the sizes
    f=print_size_of_model(float_lenet,"fp32")
    q=print_size_of_model(quantized_lstm,"int8")
    print("{0:.2f} times smaller".format(f/q))

    # compare the performance
    startTime = time.time()
    float_lenet.forward(inputs, hidden)
    executionTime = (time.time() - startTime)

    print('Floating point execution time in seconds: ' + str(executionTime))

    startTime = time.time()
    quantized_lstm.forward(inputs,hidden)
    executionTime = (time.time() - startTime)

    print('Quanitzed int execution time in seconds: ' + str(executionTime))

    out1, hidden1 = float_lenet(inputs, hidden)
    mag1 = torch.mean(abs(out1)).item()
    print('mean absolute value of output tensor values in the FP32 model is {0:.5f} '.format(mag1))

    # run the quantized model
    out2, hidden2 = quantized_lstm(inputs, hidden)
    mag2 = torch.mean(abs(out2)).item()
    print('mean absolute value of output tensor values in the INT8 model is {0:.5f}'.format(mag2))

    # compare them
    mag3 = torch.mean(abs(out1-out2)).item()
    print('mean absolute value of the difference between the output tensors is {0:.5f} or {1:.2f} percent'.format(mag3,mag3/mag1*100))

            