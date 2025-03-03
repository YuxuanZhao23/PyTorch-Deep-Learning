from torch import Tensor
from torch.nn import Module

class PlusOne(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1
    
p1 = PlusOne()
x = Tensor([1, 2, 3, 4])
print(p1(x))