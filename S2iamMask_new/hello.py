"""
from nest import register

@register(author='Yanzhao', version='1.0.0')
def hello_nest(name: str) -> str:
    

    return 'Hello ' + name
"""
import torch
import torch.nn as nn
a = torch.tensor([[[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]])
b = torch.tensor([[0.0, 0.0], [2.2, 3.3], [0, 0]])
c = b
print("c")
print(c)
b[0][0] = -100
print("changed")
print(c)
d = b.repeat(1, 16)
#d = b.view(-1, 1).repeat(0, 10).view(10, -1)
#print(a.size())
#print(d.size())
#print(d)
"""
z = torch.FloatTensor([[1,2,3,4],[5,6,8,9]])
print(z.size())
g = z.repeat(16,1).view(16,-1,4)
print(g.size())
print(g)

input = torch.randn(16,20,5,5, requires_grad=True)
m = nn.Sigmoid()
print(m(input))
loss = nn.BCELoss()
output = loss(m(input), target)
"""