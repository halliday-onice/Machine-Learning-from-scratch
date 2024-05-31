import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand
#x = torch.rand(5,3)

#print(x)

class LineNetwork(nn.Module):
      def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(1,1)) ## sequential vai utilizar uma sequencia de camadas


            pass
      #como a rede computa
      def forward(self,x):
            #x nesse caso eh o dado que entra
            return self.layers(x)

class AlgebraicDataset(Dataset):
      def __init__(self, f, interval, nsamples):
            X = urand.Uniform(interval[0], interval[1]).sample([nsamples])
            self.data = [(x,f(X)) for x in X]

      def __len__ (self):
            return len(self.data)
            
      
      def __getitem__(self, idx):
            return self.data[idx]
      
line = lambda x: 2*x + 3
interval = (-10, 10)
train_nsamples = 1000
test_nsamples = 100


train_dataset = AlgebraicDataset(line,interval, train_nsamples)
test_dataset = AlgebraicDataset(line, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, batch_size = train_nsamples, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = test_nsamples, shuffle = True)

#Hiperparametros de otimizacao
#como ela aprende de fato


model = LineNetwork()
#pra ela saber como de fato aprender, ela precisa de um parametro pra isso
#a Loss function- essa loss function utiliza o mean squared error- o erro quadratico medio
lossfunc = nn.MSELoss()
#vamos utilizar o gradiente descendente- pq ele vai me dar a direcao q a funcao MENOS CRESCE
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3) #lr: learning rate


def train(model, dataloader,lossfunc, optimizer):
      model.train()
      cummulative_loss = 0.0
      for X, y in dataloader:
            X = X.unsqueeze(1).float()
            y = y.unsqueeze(1).float()

            pred = model(X)
            loss = lossfunc(pred, y)# oq eu predisse e qual eh a verdade
            #zero os gradientes acumulados
            optimizer.zero_grad()
            #computa os gradientes
            loss.backward() # faz o calculo dos gradientes
            #ando de fato na direcao que reduz o erro local
            optimizer.step()
            cummulative_loss += loss.item()
      return cummulative_loss / len(dataloader)


#pra testar nao precisa otimizar
def test(model, dataloader, lossfunc):
      model.eval()
      cummulative_loss = 0.0
      with torch.no_grad():
            for X, y in dataloader:
                  X = X.unsqueeze(1).float()
                  y = y.unsqueeze(1).float()

                  pred = model(X)
                  loss = lossfunc(pred, y)# oq eu predisse e qual eh a verdade
                  #zero os gradientes acumulados
                  
                  cummulative_loss += loss.item()
            return cummulative_loss / len(dataloader)
      
            
      
#treinando de fato
epochs = 101
for t in range(epochs):
      train_loss = train(model, train_dataloader, lossfunc, optimizer)
      if t % 10 == 0:
            print(f"Epoch: {t}; train loss: {train_loss}")

test_loss = test(model, test_dataloader, lossfunc)
print(f"Epoch: {t}; test loss: {test_loss}")