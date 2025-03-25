# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 21:58:09 2025

@author: mytkn
"""

import torch  
import torch.nn as nn   
import numpy as np  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import time 

output_size=16 

class InnerNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, output_size),
            nn.ReLU(), 
            nn.Linear(output_size, 1)
        )
        
    def forward(self,x):
        return self.net(x)
 
       
class OuterNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, output_size),
            nn.ReLU(),
            nn.Linear(output_size, 1)
        )
    
    def forward(self, x):  
        return self.net(x)
    
class KAN(nn.Module):
    
    def __init__(self,input_dim = 4,K=3):
        """   
        input_dim : iris feature 
        K : class type count of iris.  
        """
        super().__init__()
        self.input_dim = input_dim
        self.K = K
        
        self.inner_nets = nn.ModuleList([
            nn.ModuleList([
                    InnerNet() for _ in range(input_dim)
                ]) for _ in range(K)
            ])
        
        self.outer_nets = nn.ModuleList([OuterNet() for _ in range(K)])
        
    def forward(self, x ):
        outputs = []
        for k in range(self.K):
            
            inner_sums = []
            
            for i in range(self.input_dim):
                xi = x[:,i].unsqueeze(1)
                yi = self.inner_nets[k][i](xi) 
                inner_sums.append(yi)
            
            sum_k = torch.stack(inner_sums,dim=0).sum(dim=0)
            
            zk = self.outer_nets[k](sum_k)
            outputs.append(zk) 
        final_out = torch.cat(outputs,dim=1) 
        return final_out 
         
        
iris = load_iris()  
X = iris.data.astype(np.float32)   # (150, 4)  
y = iris.target                    # (150,)  
    
# Train/Test split  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, train_size=0.8, random_state=42 )  

 
X_train_t = torch.from_numpy(X_train)  
y_train_t = torch.from_numpy(y_train).long()  
X_test_t  = torch.from_numpy(X_test)  
y_test_t  = torch.from_numpy(y_test).long() 
        
         
model = KAN(input_dim=4, K=3) 
  
        
def fit(model):
    epochs = 50
    loss_arr = []
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(),lr = 0.002)
    
    for epoch in range(epochs):
        ypred = model(X_train_t)
        loss = loss_fn(ypred,y_train_t)
        loss_arr.append(loss.item())
        loss.backward()
        optim.step()
        optim.zero_grad() 
 


try_count = 10
total_accuracy = 0

total_training_time = 0
total_test_time = 0 

for i in range(try_count):
    start_time = time.perf_counter()
    
    model = KAN()  
    fit(model) 
     
    total_training_time = (time.perf_counter() - start_time)/try_count

    start_time = time.perf_counter()
 
    ytest_pred = model(X_test_t)
    newytest = torch.argmax(ytest_pred, dim=1) 
    
    total_test_time = (time.perf_counter()-start_time)/try_count


    accuracy = accuracy_score(newytest.cpu(), y_test_t)
    total_accuracy +=accuracy/try_count
    
    print("Accuracy:", accuracy)
    #print("Confusion Matrix:\n", confusion_matrix(newytest.cpu(), y_test))

print("Total Accuracy: ",total_accuracy)   
print(f"Time for training in seconds : {total_training_time:.6f}")
print(f"Time for test in seconds : {total_test_time:.6f}")