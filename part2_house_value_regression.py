import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

pd.options.mode.chained_assignment = None

class Regressor(nn.Module):

    def __init__(self, x, nb_epoch = 1000, batch_size = 32, loss='MSE'):
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
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 80),
            nn.ReLU(),
            nn.Linear(80,60),
            nn.ReLU(),
            nn.Linear(60,20),
            nn.ReLU(),
            nn.Linear(20,self.output_size),
        )

        #Set the optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        #Check if graphic card is usable, otherwise use cpu
        self.gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        #Send the network to gpu or cpu
        self.net = self.net.to(self.gpu)
        return
    
    def forward(self, x):
        #Simply forward the network defined
        x = self.net(x)
        return x
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
        if training:
            #Save mean and std and categorical labels of training data
            number_attribute = x.dtypes[x.dtypes != 'object'].index
            mean = x[number_attribute].mean()
            std = x[number_attribute].std()
            ocean_proximity_labels = x.ocean_proximity.unique()
            mean.to_pickle('mean.pkl')
            std.to_pickle('std.pkl')
            np.save('ocean_proximity_labels.npy',ocean_proximity_labels, allow_pickle=True)
        else:
            #Restore mean and std and categorical labels for test data
            mean = pd.read_pickle('mean.pkl')
            std = pd.read_pickle('std.pkl')
            ocean_proximity_labels = np.load('ocean_proximity_labels.npy', allow_pickle=True)

        #Make a copy of the original data
        xc = x.copy()

        #Get all the attribute index that are NOT categorical, apply standardization based on training data
        number_attribute = xc.dtypes[xc.dtypes != 'object'].index
        xc[number_attribute] = (xc[number_attribute]  - mean) / (std)

        #To ensure all the categorical labels in training data are included in the 1-hot(Using mock up rows)
        ocean_proximity_labels = ocean_proximity_labels.tolist()
        count_append = 0
        for label in ocean_proximity_labels:
            if label not in xc.ocean_proximity.unique():
                temp = xc.tail(1).copy()
                temp.loc[0,'ocean_proximity'] = label
                xc.append(temp)
                count_append += 1
        #Convert categorical attribute into 1-hot
        xc = pd.get_dummies(xc, dummy_na=True)
        #Drop mock up rows
        xc.drop(xc.tail(count_append).index,inplace=True)

        #Fill all the NaN(empty) data
        xc = xc.fillna(xc.mean())

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
                print(loss.item())
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
        #X, _ = self._preprocessor(x, training = False) # Do not forget
        X = x
        #Put the test data into DataLoader
        validation_data = DataLoader([[X[i]] for i in range(X.shape[0])], batch_size=1, shuffle=False)
        #Use a numpy array to store prediction labels
        result = np.zeros((X.shape[0],1))
        #Convert the network to eval mode
        self.net.eval()
        for i, [data] in enumerate(validation_data):
            data = data.type('torch.FloatTensor').to(self.gpu)
            pred = self.net(data)
            #pred = np.exp(pred)
            result[i,0] = pred
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
        #make prediction on the testing data
        prediction = self.predict(X)

        #Convert the log(label) back to label
        prediction = np.exp(prediction)
        Y = np.exp(Y)

        #RMSE score
        #score = math.sqrt(mean_squared_error(Y,prediction))

        #Square Root score
        score = r2_score(Y,prediction)



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



def RegressorHyperParameterSearch(): 
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

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 20, batch_size=32, loss='MSE')
    train_num = round(4*x_train.shape[0] / 5)
    regressor.fit(x_train.loc[:train_num,:], y_train.loc[:train_num,:])
    # save_regressor(regressor)

    # Error
    error = regressor.score(x_train.loc[train_num+1:,:], y_train.loc[train_num+1:,:])
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

