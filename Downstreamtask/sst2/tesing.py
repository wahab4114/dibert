from transformers import BertTokenizer, BertModel
import torch

a = torch.randn([2,2,2])
print(a.size())
b= a.split(1,dim=-1)
print(b[0].size())
print(b[1].size())

print(a)
print(b)