import torch
from torch.autograd import Variable

def zero_variable(size, use_cuda, volatile=False):
    if use_cuda:
        return Variable(torch.cuda.FloatTensor(*size).zero_(), volatile=volatile)
    else:
        return Variable(torch.FloatTensor(*size).zero_(), volatile=volatile)

