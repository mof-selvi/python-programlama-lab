#!/usr/bin/env python
# coding: utf-8

# In[106]:


# Muhammed Ömer Faruk Selvi

#%%
import numpy as np

def rnd(x):
    return round(x,2)
def rndarr(arr,precision=2):
    return np.round_(arr, decimals=precision)
def prnt(mtr):
    print(mtr,"\n", rndarr(globals()[mtr], 4))



# sonlu farklar yaklaşımı ile autodiff
def derivative(f, x, y, axis=1, x_eps=0.00001, y_eps=0.0):
    results = []
    for i in range(len(y)):
        for j in range(len(y[0])):
            results.append(derivative_val(f,np.array([x[i][j]]),np.array([y[i][j]])))
    return np.array([results])

# kaynak: "Scikit-Learn, Keras ve Tensorflow ile Uygulamalı Makine Öğrenmesi", Aurelien Geron, s.814
def derivative_val(f, x, y, axis=1, x_eps=0.00001, y_eps=0.0):
    return (f(x+x_eps, y+y_eps) - f(x,y))/(x_eps + y_eps)




# sigmoid aktivasyon fonksiyonu
def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))

# sigmoid'in türevi (biri A diğeri Z için kullanılabilir)
def sigmoid_d_z(val):
    return val * (1.0 - val)

def sigmoid_d(val):
    return sigmoid(val) * (1.0 - sigmoid(val))




# mult = np.sum(np.dot(dataset_x, W_hl.T), axis=1) 
# mult2 = mult + B_hl

def loss_function_mse(y_prediction, y_truth):
    len_ = len(y_truth)
    return np.sum((y_prediction-y_truth)**2) / len_


# def loss_function_mse_d(y_prediction, y_truth):
# #     len_ = len(y_truth)
#     return (y_prediction-y_truth)

def loss_function_mse_d(y_prediction, y_truth):
    return 2*(y_prediction-y_truth)


# kaynak: https://stackoverflow.com/a/67616451
def loss_function_bce(y_truth, y_prediction):
    term_0 = (1-y_truth) * np.log(1-y_prediction + 1e-7)
    term_1 = y_truth * np.log(y_prediction + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def loss_function_bce_d(y_prediction, y_truth):
    return -(y_truth/y_prediction + 1e-7) + (1-y_truth)/(1-y_prediction + 1e-7)
# kaynak: http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
# def loss_function_bce_d(y_prediction, y_truth):
#     return derivative(loss_function_bce,y_prediction,y_truth)


def prediction(data_x, data_y):
    global Z_hl, A_hl, Z_op, A_op
    
    perceptron_value = np.dot(data_x, W_hl) + B_hl
    Pred_hl = sigmoid(perceptron_value)
    
    Z_hl = perceptron_value
    A_hl = Pred_hl

    # print(perceptron_value)
    # print(prediction)
        
#     print("W_op",W_op)
#     print("Pred_hl",Pred_hl)
    
    perceptron_value2 = np.dot(Pred_hl, W_op) + B_op
#     print("perceptron_value2",perceptron_value2)
    # Pred_op = sigmoid(perceptron_value2)
    Pred_op = perceptron_value2
    
    Z_op = perceptron_value2
    A_op = Pred_op
    prnt("A_op")
    print(data_y)
    
#     print("Loss:",loss_function_mse(Pred_op,data_y))

    


################ DATA
dataset_x = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

dataset_y = [
    [0,0],
    [1,1],
    [1,1],
    [0,1]
]

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)

sample_count = dataset_y.shape[0]

learning_rate = 0.1

batch_size = 1 #stochastic


# In[108]:


# Network 1
# Çıkışta aktivasyon yok
# Loss: MSE

print("#"*50)
print("Network 1")
print("#"*50)


# bir gizli katmanımız ve bir çıktı katmanımız var
# gizli katman 3 nöronlu; yani her bir girdi nöronundan üçer bağlantı başlar

W_hl = rndarr(np.random.random((2, 3))) # gizli katmana giden ağırlıklar
B_hl = np.random.random((3)) # gizli katman için bias
Z_hl = np.zeros((3)) # gizli katman sonucu (inaktif)
A_hl = np.zeros((3)) # gizli katman sonucu
D_hl = np.random.random((2, 3)) # gizli katmana giden ağırlıklarının deltaları

W_op = rndarr(np.random.random((3, 2))) # çıktı katmanına giden ağırlıklar
B_op = np.random.random((2)) # çıktı katmanı için bias
Z_op = np.zeros((2)) # çıktı katmanı sonucu (inaktif)
A_op = np.zeros((2)) # çıktı katmanı sonucu
D_op = np.random.random((3, 2)) # çıktı katmanına giden ağırlıkların deltaları

####### DATA




print("Weights and biases before starting:")
prnt("W_op")
prnt("W_hl")
prnt("B_op")
prnt("B_hl")
print("\n\n")

for epoch in range(1,51):
    print("\t Training... Epoch",epoch)
    for bno in range(0,sample_count,batch_size):
        
        print("Passing samples in the range of",bno,":",bno+batch_size)
        i_low = bno
        i_high = bno+batch_size
        if i_high > sample_count:
            i_high = sample_count
        batch_x = dataset_x[i_low:i_high]
        batch_y = dataset_y[i_low:i_high]
        # print(batch_x,batch_y,"\n\n\n")
        
    
        prediction(batch_x, batch_y)

        # print("A_op",A_op)
        # error = (batch_y[i] - A_op)**2
        error = loss_function_mse(A_op, batch_y)
        D_op = loss_function_mse_d(A_op, batch_y)
#         prnt("A_op")
#         prnt("batch_y")
#         prnt("D_op")
#         D_op = derivative(loss_function_mse,A_op, batch_y) # aynı sonucu veriyor
#         prnt("D_op")
        # prnt("delta_by")
        # print("A_hl",A_hl)
        delta_wy = np.array([np.outer(D_op[a],A_hl[a]) for a in range(len(D_op))])
        delta_wy = np.average(delta_wy,axis=0).T
        # print("delta_wy",delta_wy)
        delta_by = np.average(D_op,axis=0).T


        D_hl = np.dot(D_op, W_op.T)
        D_hl = np.multiply(D_hl, sigmoid_d(A_hl))
        # print("D_hl",D_hl)
        delta_wh = np.array([np.outer(D_hl[a],batch_x[a]) for a in range(len(D_hl))])
        delta_wh = np.average(delta_wh,axis=0).T
        # print("delta_wh",delta_wh)
        delta_bh = np.average(D_hl,axis=0).T


        # print("W_op",W_op)
        # print("W_hl",W_hl)
        # error2 = loss_function_mse(A_op, batch_y[i])
        # delta_op = np.dot(batch_x, error * sigmoid_d(A_op))


        W_op -= learning_rate * delta_wy
        W_hl -= learning_rate * delta_wh


        B_op -= learning_rate * delta_by
        B_hl -= learning_rate * delta_bh

    
    prnt("W_op")
    prnt("W_hl")
    prnt("B_op")
    prnt("B_hl")
    prediction(dataset_x, dataset_y)
    print("Loss:",loss_function_mse(A_op,dataset_y))
    print("\n")


print("\n\nDone! Final prediction result:")
prediction(dataset_x, dataset_y)
print("Loss:",loss_function_mse(A_op,dataset_y))
# print(A_op)


# In[110]:


# Network 2
# Çıkışta sigmoid var
# Loss: Binary Cross Entropy
# bir gizli katmanımız ve bir çıktı katmanımız var
# gizli katman 3 nöronlu; yani her bir girdi nöronundan üçer bağlantı başlar

print("#"*50)
print("Network 2")
print("#"*50)

W_hl = rndarr(np.random.random((2, 3))) # gizli katmana giden ağırlıklar
B_hl = np.random.random((3)) # gizli katman için bias
Z_hl = np.zeros((3)) # gizli katman sonucu (inaktif)
A_hl = np.zeros((3)) # gizli katman sonucu
D_hl = np.random.random((2, 3)) # gizli katmana giden ağırlıklarının deltaları

W_op = rndarr(np.random.random((3, 2))) # çıktı katmanına giden ağırlıklar
B_op = np.random.random((2)) # çıktı katmanı için bias
Z_op = np.zeros((2)) # çıktı katmanı sonucu (inaktif)
A_op = np.zeros((2)) # çıktı katmanı sonucu
D_op = np.random.random((3, 2)) # çıktı katmanına giden ağırlıkların deltaları

####### DATA




print("Weights and biases before starting:")
prnt("W_op")
prnt("W_hl")
prnt("B_op")
prnt("B_hl")
print("\n\n")

for epoch in range(1,51):
    print("\t Training... Epoch",epoch)
    for bno in range(0,sample_count,batch_size):
        
        print("Passing samples in the range of",bno,":",bno+batch_size)
        i_low = bno
        i_high = bno+batch_size
        if i_high > sample_count:
            i_high = sample_count
        batch_x = dataset_x[i_low:i_high]
        batch_y = dataset_y[i_low:i_high]
        # print(batch_x,batch_y,"\n\n\n")
        
    
        prediction(batch_x, batch_y)

        # print("A_op",A_op)
        # error = (batch_y[i] - A_op)**2
        error = loss_function_bce(A_op, batch_y)
#         D_op = loss_function_bce_d(A_op, batch_y)
        D_op = derivative(loss_function_bce,A_op, batch_y)
        # prnt("D_op")
        # prnt("delta_by")
        # print("A_hl",A_hl)
        delta_wy = np.array([np.outer(D_op[a],A_hl[a]) for a in range(len(D_op))])
        delta_wy = np.average(delta_wy,axis=0).T
        # print("delta_wy",delta_wy)
        delta_by = np.average(D_op,axis=0).T


        D_hl = np.dot(D_op, W_op.T)
        D_hl = np.multiply(D_hl, sigmoid_d(A_hl))
        # print("D_hl",D_hl)
        delta_wh = np.array([np.outer(D_hl[a],batch_x[a]) for a in range(len(D_hl))])
        delta_wh = np.average(delta_wh,axis=0).T
        # print("delta_wh",delta_wh)
        delta_bh = np.average(D_hl,axis=0).T


        # print("W_op",W_op)
        # print("W_hl",W_hl)
        # error2 = loss_function_mse(A_op, batch_y[i])
        # delta_op = np.dot(batch_x, error * sigmoid_d(A_op))


        W_op -= learning_rate * delta_wy
        W_hl -= learning_rate * delta_wh


        B_op -= learning_rate * delta_by
        B_hl -= learning_rate * delta_bh

    
    prnt("W_op")
    prnt("W_hl")
    prnt("B_op")
    prnt("B_hl")
    prediction(dataset_x, dataset_y)
    print("Loss:",loss_function_mse(A_op,dataset_y))
    print("\n")


print("\n\nDone! Final prediction result:")
prediction(dataset_x, dataset_y)
print("Loss:",loss_function_mse(A_op,dataset_y))
# print(A_op)
# autodiff mse için çalışıyor fakat BCE'de loss değeri patlıyor

