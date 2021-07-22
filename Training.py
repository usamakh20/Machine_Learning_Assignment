import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.autograd import Variable
from skorch import NeuralNetRegressor
import pickle

np.random.seed(436)

################################## Load Data ###################################

Xtr = np.loadtxt("TrainData.csv")
ytr = np.loadtxt("TrainLabels.csv").reshape(-1, 1)

##################### Normalization and Train Test split #########################

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(scaler_X.fit_transform(Xtr),
                                                    scaler_y.fit_transform(ytr),
                                                    test_size=0.15,
                                                    random_state=98)

Xt_train = Variable(torch.tensor(X_train, dtype=torch.float))
yt_train = Variable(torch.tensor(y_train, dtype=torch.float))

Xt_test = Variable(torch.tensor(X_test, dtype=torch.float))
yt_test = Variable(torch.tensor(y_test, dtype=torch.float))

############################### Creating Model ###################################

ann = nn.Sequential(nn.Linear(8, 30),
                    nn.ReLU(),
                    nn.Linear(30, 20),
                    nn.ReLU(),
                    nn.Linear(20, 10),
                    nn.ReLU(),
                    nn.Linear(10, 1),
                    nn.Sigmoid())

net = NeuralNetRegressor(
    ann,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    train_split=None,
    criterion=nn.MSELoss
)

hyper_params = {
    'lr': [0.1, 1, 2, 4],
    'max_epochs': [10, 200, 1000, 3000]
}

########### Cross validation and Hyperparameter Tuning ##########################

gs = GridSearchCV(net, hyper_params, refit=True, cv=5, scoring='neg_root_mean_squared_error', verbose=3)

gs.fit(Xt_train, yt_train)

############################### Validation and RMSE ###############################

print("Best Params: ", gs.best_params_)
print("RMSE on Train Data: ", scaler_y.inverse_transform(np.array([[gs.best_score_ * -1]])).flatten()[0])

output = gs.predict(Xt_test)

print("RMSE on Test data: ",
      mean_squared_error(scaler_y.inverse_transform(yt_test), scaler_y.inverse_transform(output), squared=False))


############################### Saving model #######################################

pickle.dump(gs.best_estimator_, open('myModel.pkl', 'wb'))
