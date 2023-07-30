import numpy as np
import torch.nn as nn 
import torch
import os,io,sys
import torchvision
import cv2



class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784,256)
        self.act = nn.ReLU()

        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,10)
        
        self.out = nn.Softmax(dim=0)
        self.sig = nn.Sigmoid()

    def forward(self,x):


        x = x.view(-1)

        x= self.fc(x)
        x= self.act(x)


        x = self.sig(self.fc2(x))
        x= self.fc3(x)
  

        return x


def linear_layer(x,w,bias):
    x=x.cpu().detach().numpy().reshape(1,784)


    bil = 0 
    outz = [] 
 
    for wi in w:

        dotz = np.dot(x[0],wi) + bias[bil]

        #relu 


        dotz = np.maximum(0,dotz)
       
        outz.append(dotz)

        bil+=1
        
    return outz

def save_data(x,fname):
    fd = open(fname,"w")

    dat = "" 
    for w in x:

        nz = "" 
        if(type(w) != np.float32):

            for num in w:
                nz+=str(num)+","
            
            nz =nz[:-1]


            #print(nz)
        else:
            nz+=str(w)    
        print(nz) 
        dat+=nz+"\n"

    #    print(dat)

    fd.write(dat)
    fd.close()
    
#https://gist.github.com/yang-zhang/217dcc6ae9171d7a46ce42e215c1fee0


def smax(x):
    return np.exp(x) / ( np.exp(x).sum(-1))

def log_smax(x):
    return np.log(smax(x))

def nll(input, target): return -input[range(target.shape[0]), target].mean()

def nloss(x,y):

    oudz = [] 
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            oudz.append(-x[i][y[j]])

    oudz = np.mean(np.array(oudz))
    return oudz

def celoss(real,targ):
    
    real = real.cpu().detach().numpy()
    targ = targ.cpu().detach().numpy()

    pot = log_smax(real)
    print(pot.shape)
    print(targ.shape)
    exit(1)
    out = nloss(pot,targ)
    print("O:",out)
    
    exit(1)

    return md


torch.manual_seed(1024)




def test_1():
    tr = torchvision.transforms.Compose([torchvision.transforms.Resize([28,28]),torchvision.transforms.ToTensor()])
    data = torchvision.datasets.MNIST(download=True,root="./data",transform=tr)
    #trainl = torch.utils.data.DataLoader(data,batch_size=1,shuffle=True)

    net = BasicNN()


    fd =open("./train.csv","r")

    data =fd.read().split("\n")[1].split(",")[1:]

    img =[] 
    for px in data:
        
        px = np.float32(px)
        px =  ( px - 0.00 )  / (255.00 )
        

        img.append(px)


    img = np.array(img)
    img = torch.tensor(img).float()
    fd.close() 
    x=  img

    y = torch.tensor(np.array([1]))

    lay = nn.Sigmoid()
 
    out = net(x).reshape(1,10)

    criterion = nn.MSELoss()

    od = linear_layer(x,net.fc.weight.cpu().detach().numpy(),net.fc.bias.cpu().detach().numpy())

    print(out)

    outd = criterion(out,y)
    print(outd)
    exit(1)
 #   print(outd)

   #print(celoss(out,y))
    #out = criterion(out,y)
    #print(out.backward())
  #  exit(1)

    weight = net.fc.weight.cpu().detach().numpy()
    bias = net.fc.bias.cpu().detach().numpy()

    print("hello?")


    save_data(weight,"weights.dat")
    save_data(bias,"bias.dat")
    exit(1)

def test_2():
    z = np.random.uniform(-1.0,1.0,[128,1])
    x = np.random.uniform(-1.0,1.0,[10,1])
    y = np.random.uniform(-1.0,1.0,[10])
    print(np.dot(z,x))
#test_1()
test_2()
exit(1)

EPOCHS = 100 
LR = 3e-4


torch.manual_seed(1024)

tr = torchvision.transforms.Compose([torchvision.transforms.Resize([28,28]),torchvision.transforms.ToTensor()])
data = torchvision.datasets.MNIST(download=True,root="./data",transform=tr)
trainl = torch.utils.data.DataLoader(data,batch_size=1,shuffle=True)


crit = nn.CrossEntropyLoss()
net = BasicNN()
optim = torch.optim.Adam(net.parameters(),lr=LR)

for e in range(EPOCHS):
    dls = 0 
    d = 0
    for b,(x,y) in enumerate(trainl):
        x=x.reshape(28,28)

        optim.zero_grad()

        out=  net(x)
        #y=y.float()
        out = out.reshape(1,10)
        ls = crit(out,y)

        ls.backward()

        optim.step()

        dls+=ls.item()
        d+=1    
        print(dls/d)    

torch.save(net.state_dict(),"./basic.pth")
