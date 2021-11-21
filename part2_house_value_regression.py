import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn.modules import activation
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from numpy.random import default_rng

pd.options.mode.chained_assignment = None

class Regressor(nn.Module):
    
    def __init__(self, x, nb_epoch = 1000, scaler = 'Standard', batch_size = 32, loss='MSE', num_layers = 6, num_neurons = 120, activations = 'relu'
    ,num_dropout=0.2, optimizer='Adam', lr=0.001, momentum=0.9, L2=1e-5):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        super(Regressor, self).__init__()

        #Perserved variables
        self.ocean_proximity_labels = None
        if scaler == 'Standard':
            self.scaler = StandardScaler()
        elif scaler == 'MINMAX':
            self.scaler = MinMaxScaler()
        elif scaler == 'Robust':
            self.scaler = RobustScaler()


        #Get the sizes of input data and output data to build the network
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        #Select the loss function we will use
        if loss == 'MSE':
            self.loss_function = torch.nn.MSELoss()
        else:
            self.loss_function = torch.nn.CrossEntropyLoss()

        #Define the network
        activation_abbrev = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(),
                             'tanh': nn.Tanh(), 'identity':nn.Identity()}
        self.model = [nn.Linear(self.input_size, num_neurons, bias=True)]
        neurons_decay = math.floor(num_neurons/num_layers)
        for i in range(num_layers-1):
            self.model.append(nn.Dropout(p=num_dropout))
            self.model.append(activation_abbrev[activations])
            if i == num_layers-2:
                self.model.append(nn.Linear(num_neurons - neurons_decay*i,self.output_size, bias=True))
            else:
                self.model.append(nn.Linear(num_neurons - neurons_decay*i,num_neurons - neurons_decay*(i+1), bias=True ))
        self.net = nn.Sequential(*self.model)
        #self.net.apply(self.weights_init_normal)

        #Set the optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=L2)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(),lr=lr, momentum=momentum, weight_decay=L2)

        #Check if graphic card is usable, otherwise use cpu
        self.gpu = torch.device('cpu')

        #Send the network to gpu or cpu
        self.net = self.net.to(self.gpu)
        return
    
    def forward(self, x):
        #Simply forward the network defined
        x = self.net(x)
        return x
    
    def weights_init_normal(self, m):
        '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''
        # for every Linear layer in a model
        if isinstance(m, nn.Linear):
            # m.weight.data shoud be taken from a normal distribution
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data should be 0
            torch.nn.init.zeros_(m.bias)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    


    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        #Make a copy of the original data
        xc = x.copy(deep=True)
        
        #Get all the attribute index that are NOT categorical
        number_attribute = x.dtypes[x.dtypes != 'object'].index

        if training:
            #Save mean and std and categorical labels of training data
            number_attribute = x.dtypes[x.dtypes != 'object'].index
            self.ocean_proximity_labels = x.ocean_proximity.unique()
            self.scaler.fit(xc[number_attribute])

        #Fill all the NaN(empty) data
        xc[number_attribute] = xc[number_attribute].fillna(xc[number_attribute].mean())

        #Apply scaler based on training data
        xc[number_attribute] = self.scaler.transform(xc[number_attribute])

        #To ensure only all the categorical labels in training data are included in the 1-hot(Using mock up rows)
        ocean_proximity_labels = self.ocean_proximity_labels.tolist()
        count_append = 0
        #In case label showed in training data but not in testing data
        for label in ocean_proximity_labels:
            if label not in xc.ocean_proximity.unique():
                temp = xc.tail(1).copy(deep=True)
                temp.iloc[0,xc.columns.get_loc('ocean_proximity')] = label
                xc = pd.concat([xc,temp])
                count_append += 1
        #In case label showed in testing data but not in training data     
        for row in range(0,xc.shape[0]):
            if xc.iloc[row,xc.columns.get_loc('ocean_proximity')] not in ocean_proximity_labels:
                xc.iloc[row,xc.columns.get_loc('ocean_proximity')] = np.nan
        
        #Convert categorical attribute into 1-hot
        xc = pd.get_dummies(xc)
        #Drop mock up rows
        if count_append != 0:
            xc = xc.iloc[:-count_append,:]


        #Convert the data to numpy array
        xc = np.array(xc, dtype=np.float32)

        #Convert the label to log(label) in numpy array, since labels are very large values and MSELoss may overflow 
        if y is not None:
            y = y.apply(np.log)
            y = np.array(y, dtype=np.float32)
        return xc, (y if y is not None else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        #Get training data and labels
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        #Put the data into DataLoader
        train_data = DataLoader([[X[i], Y[i]] for i in range(X.shape[0])], batch_size=self.batch_size, shuffle=True)

        #Convert the network to train mode
        self.net.train()
        #train
        for epoch in range(self.nb_epoch):
            for i, [data, label] in enumerate(train_data):
                self.optimizer.zero_grad()
                data = data.type('torch.FloatTensor').to(self.gpu)
                label = label.type('torch.FloatTensor').to(self.gpu)
                pred = self.net(data)
                loss = self.loss_function(pred, label)
                #print(loss.item())
                loss.backward()
                self.optimizer.step()                      
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        #preprocess
        X, _ = self._preprocessor(x, training = False) # Do not forget
        #convert input data into tensor
        X = torch.tensor(X)
        #input to net
        pred = self.net(X)
        #convert to numpy and return
        result = pred.detach().numpy()
        return result
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        #Convert the network to eval mode
        self.net.eval()

        #List contain all the prediction results
        result = []

        total_batch = math.ceil(x.shape[0]/self.batch_size)

        for i in range(total_batch):
            #split raw data into batches
            if i == total_batch-1:
                data = x.iloc[i*self.batch_size:,:]
            else: 
                data = x.iloc[i*self.batch_size:(i+1)*self.batch_size,:]
            #make prediction on the testing data
            prediction = self.predict(data)
            #collect the prediction results
            result.append(prediction)

        #concatenate the result to compute the score
        result = np.concatenate(result)

        #Convert the log(label) back to label
        result = np.exp(result)
        Y = np.exp(Y)

        #RMSE score
        #score = math.sqrt(mean_squared_error(Y,prediction))

        #Square Root score
        try:
            score = r2_score(Y,result)
            print(str(score))
            score = mean_squared_error(Y,result,squared=False)
            print(str(score))
        except:
            score = -1.0 



        return score # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x,y): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    scaler = ['Standard','MINMAX','Robust']
    num_layers = [2,4,6,8]
    num_neurons = [40,80,120,160]
    num_dropout = [0.2,0.4]
    optimizer = ['Adam','SGD']
    learning_rate = [0.01,0.001,0.0005]
    momentum = [0.9]
    L2 = [0.0, 1e-5]
    batch_size = [32]
    loss=['MSE']
    activation=['relu','tanh']
    epoch=[100]

    scaler = ['Standard']
    num_layers = [6]
    num_neurons = [120]
    num_dropout = [0.2]
    optimizer = ['Adam']
    learning_rate = [0.001]
    momentum = [0.9]
    L2 = [1e-5]
    batch_size = [32]
    loss=['MSE']
    activation=['relu']
    epoch=[100]


    rng = default_rng(seed=1024)
    shuffle_index = rng.permutation(len(x))
    x = x.iloc[shuffle_index]
    y = y.iloc[shuffle_index]
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    train_num = round(4*x.shape[0] / 5)
    x_train = x.loc[:train_num,:]
    y_train = y.loc[:train_num,:]
    x_valid = x.loc[train_num+1:,:]
    y_valid = y.loc[train_num+1:,:]
    print("scaler,layer,neuron,activation,dropout,optimizer,learning rate,L1/L2,momentum,error")
    for s in scaler:
        for optim in optimizer:
            for layer in num_layers:
                for neuron in num_neurons:
                    for dropout in num_dropout:
                        for acti in activation:
                            for lr in learning_rate:
                                for L in L2:
                                    regressor = Regressor(x, scaler=s, nb_epoch = epoch[0], batch_size=batch_size[0], loss=loss[0], num_layers=layer, num_neurons=neuron, activations=acti, num_dropout=dropout, optimizer=optim, lr=lr,  L2=L, momentum=momentum[0])
                                    regressor.fit(x_train,y_train)
                                    error = regressor.score(x_valid, y_valid)
                                    if optim == 'SGD':
                                        print(s + "," + str(layer) + "," + str(neuron) + "," + acti + "," + str(dropout) + "," + optim + "," + str(lr) + "," + str(L) + "," + str(momentum[0])+ "," + str(error))
                                    else:
                                        print(s + "," + str(layer) + "," + str(neuron) + "," + acti + "," + str(dropout) + "," + optim + "," + str(lr) + "," + str(L) + "," + "," + str(error))
    save_regressor(regressor)
    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # # Training
    # # This example trains on the whole available dataset. 
    # # You probably want to separate some held-out data 
    # # to make sure the model isn't overfitting
    # regressor = Regressor(x_train, nb_epoch = 100, batch_size=32, loss='MSE')
    # train_num = round(4*x_train.shape[0] / 5)
    # regressor.fit(x_train.loc[:train_num,:], y_train.loc[:train_num,:])
    # save_regressor(regressor)

    #regressor = load_regressor()
    # # Error
    #error = regressor.score(x_train.iloc[train_num+1:,:], y_train.iloc[train_num+1:,:])
    #print("\nRegressor error: {}\n".format(error))

    RegressorHyperParameterSearch(x_train,y_train)


if __name__ == "__main__":
    example_main()

