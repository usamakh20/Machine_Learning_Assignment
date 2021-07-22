import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import torch

Xtr = np.loadtxt("TrainData.csv")
ytr = np.loadtxt("TrainLabels.csv").reshape(-1, 1)

scaler_X = MinMaxScaler().fit(Xtr)
scaler_y = MinMaxScaler().fit(ytr)

Xts = np.loadtxt("TestData.csv")

Xt_test = Variable(torch.tensor(scaler_X.fit_transform(Xts), dtype=torch.float))

model = pickle.load(open("myModel.pkl", 'rb'))

yts = model.predict(Xt_test)

np.savetxt('i191236_Predictions.csv', scaler_y.inverse_transform(yts).flatten())
