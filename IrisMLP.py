# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 22:13:31 2025

@author: mytkn
"""

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import time



iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)


output_size=64 #64
hidden_output_size=16 # 16
 

class IrisMLP(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(4, output_size)
        self.act1=nn.ReLU()
        self.l2=nn.Linear(output_size, hidden_output_size)
        #self.drop = nn.Dropout(0.2)
        self.act2 = nn.ReLU()
        self.l3=nn.Linear(hidden_output_size, 3)
        
    def forward(self,x):
        #forward pass
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        #x = self.drop(x)
        x = self.act2(x)
        x = self.l3(x)
        return x
        

def fit(model):
    epochs = 50
    loss_arr = []
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(),lr = 0.002)
    
    for epoch in range(epochs):
        ypred = model(X_train_tensor)
        loss = loss_fn(ypred,y_train_tensor)
        loss_arr.append(loss.item())
        loss.backward()
        optim.step()
        optim.zero_grad()
    plt.plot(loss_arr)
    plt.show() 


try_count = 10
total_accuracy = 0


total_training_time = 0
total_test_time = 0 

for i in range(try_count):
    start_time = time.perf_counter()
    
    model = IrisMLP()  
    fit(model)
    
    total_training_time = (time.perf_counter() - start_time)/try_count
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    start_time = time.perf_counter()
    
    ytest_pred = model(X_test_tensor)
    newytest = torch.argmax(ytest_pred, dim=1)
    
    total_test_time = (time.perf_counter()-start_time)/try_count
    
    accuracy = accuracy_score(newytest.cpu(), y_test)
    total_accuracy += accuracy/try_count
    
    print("Accuracy:", accuracy) 
    #print("Confusion Matrix:\n", confusion_matrix(newytest.cpu(), y_test))


print("Total Accuracy: ",total_accuracy)
print(f"Time for training in seconds : {total_training_time:.6f}")
print(f"Time for test in seconds : {total_test_time:.6f}")