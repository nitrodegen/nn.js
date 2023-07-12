import numpy as np
import torch.nn as nn 
import torch
import os,io,sys



np.random.seed(1024)


data = np.array([
    [0.1,0.2],
    [0.2,0.3]
],dtype=np.float32)

dati = data.copy()

xd = np.array([[0.5,0.6],[0.7,0.8]])

#print(data @ xd)


z  =np.array([[ 0.4369,0.4151],
             [0.3,0.4]])

torch.manual_seed(1024)
lay = nn.Linear(2,4)
w = lay.weight.cpu().detach().numpy()

data =torch.tensor(data)

bias = lay.bias.cpu().detach().numpy()
my_w = [] 
lp = 0 

for bdat in dati:
    
    outz  =[] 
    for z in w:
#        print(bdat,z)
        out = np.dot(bdat,z)+bias[lp]
        
        lp+=1
        outz.append(out)
    lp = 0
    print(outz)


exit(1)
print("WEIGHTS:",lay.weight,lay.weight.shape)

out = lay(data)

print("OUT:",out,out.shape)


