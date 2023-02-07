import torch
import torch.nn as nn

from Transformer import Transformer

from util.utils import *

class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.lambda_t = nn.Parameter(torch.tensor(0.0))

    def forward(self, d, x_t, fp_operator, fbp_operator):
        d_t = fp_operator(x_t)
        d_diff = d - d_t
        x_error = fbp_operator(d_diff)
        x_out = x_t + self.lambda_t * x_error
        return x_out

class IterBlock(nn.Module):
    def __init__(self):
        super(IterBlock, self).__init__()
        self.transformer = Transformer()
        self.projection = Projection()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, d, x, fp_operator, fbp_operator):
        tmp1 = self.projection(d, x, fp_operator, fbp_operator)
        tmp2 = self.transformer(x)
        x_out = self.relu(tmp1 + tmp2)
        return x_out

class LearnFormer(nn.Module):
    def __init__(self, fp_operator, fbp_operator, num_iter):
        super(LearnFormer, self).__init__()
        self.fp_operator = fp_operator
        self.fbp_operator = fbp_operator
        
        self.num_iter = num_iter
        self.iter_blocks = nn.ModuleList([IterBlock() for _ in range(self.num_iter)])

    def forward(self, d):
        x = normlize_tensor(self.fbp_operator(d))[None, ...]

        for iter_block in self.iter_blocks:
            x = iter_block(d, x, self.fp_operator, self.fbp_operator)
        
        return x