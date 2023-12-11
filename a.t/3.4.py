import torch
import torch.nn as nn
torch.manual_seed(0)
class Demo(torch.nn.Module):
    def __init__(self):
        super(Demo, self).__init__()
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc(x)
        #print("in process x",x)
        return x
model_fp32=Demo()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
input_fp32 = torch.randn(1, 2) #input a*b nn.linear:b, c.  "c" is the dimension of output
print('input f32',input_fp32)
res_f32 = model_fp32(input_fp32)
print('You could observe that the results below are different')
print('result,fp32',res_f32)
res_int8 = model_int8(input_fp32)
print('result,int8',res_int8)
########## observe the model
#print("modelfp32 dict",model_fp32.state_dict())
#print("modelint8 dict",model_int8.state_dict())
###observe the weight's type'
print("You could observe that the weights' values and types below are different")
print('modelf32 weight',model_fp32.state_dict()['fc.weight'])
print('modelint8 weight',model_int8.state_dict()['fc._packed_params._packed_params'][0])
print('modelint8 weight (int_repr)',model_int8.state_dict()['fc._packed_params._packed_params'][0].int_repr())
# input f32 tensor([[0.3643, 0.1344]])
# You could observe that the results below are different
# result,fp32 tensor([[-0.5329]], grad_fn=)
# result,int8 tensor([[-0.5332]])
# You could observe that the weights' values and types below are different
# modelf32 weight tensor([[-0.0053,  0.3793]])
# modelint8 weight tensor([[-0.0060,  0.3778]], size=(1, 2), dtype=torch.qint8,
#        quantization_scheme=torch.per_tensor_affine, scale=0.0029750820249319077,
#        zero_point=0)
# modelint8 weight (int_repr) tensor([[ -2, 127]], dtype=torch.int8)