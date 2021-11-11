import torch

a = torch.tensor([-1, 1, 1 , 0 ,1,-1])
print(a.eq(1))
print(a.eq(1).nonzero().squeeze())
print(a.eq(1).nonzero().size())

print("kee")