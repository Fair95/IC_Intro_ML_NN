# Deep learning packages
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
# Data preprocessing packages
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV
# Other packages
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
print('Import Done')



## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''
    # for every Linear layer in a model
    if isinstance(m, nn.Linear):
        # m.weight.data shoud be taken from a normal distribution
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data should be 0
        torch.nn.init.zeros_(m.bias)

neurons = [20, 40, 10, 1]
activations = ['tanh', 'tanh', 'tanh']
dropouts = [0, 0.4, 0.2]
param_grid = {'neurons': neurons, 'activations': activations, 'dropouts': dropouts}

class LinearRegressorModel(nn.Module):
    def __init__(self, input_dim, neurons, activations, dropouts):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        super(LinearRegressorModel, self).__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations
        self.layers = [nn.Linear(self.input_dim, self.neurons[0], bias=True)]
        activation_abbrev = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(),
                             'tanh': nn.Tanh(), 'identity':nn.Identity()}
        for i in range(len(self.neurons)-1):
            self.layers.append(nn.Dropout(p=dropouts[i], inplace=True))
            self.layers.append(activation_abbrev[self.activations[i]])
            self.layers.append(nn.Linear(self.neurons[i], self.neurons[i+1], bias=True))
        self.model = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.model(x)


class Regressor():

    def __init__(self, x, nb_epoch = 1000):
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

        # Find categorical columns
        cat_col = [i for i in x.columns if x.dtypes[i]=='object']
        num_col = x.columns.difference(cat_col)

        # Find columns with missing values
        miss_col = [col for col in x.columns if x[col].isnull().any()]

        # Handle different types of data differently
        self.cat_pipe = Pipeline([
                            ('binarizer', OneHotEncoder())
                            ])
        self.num_pipe = Pipeline([
                            ('normaliser', RobustScaler()),
                            ('imputer', KNNImputer()),
                            ])

        # Combine two pipes and deal with the whole data
        self.feature_pipe = ColumnTransformer(
        transformers=[
        ('num', self.num_pipe, num_col),
        ('cat', self.cat_pipe, cat_col)
        ])

        # We only scale the label to have a max of 1, DONT SHIFT OR CENTER THE DATA
        self.label_pipe = Pipeline([
                            ('scaler', MaxAbsScaler())
                            ])

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)


        # Instantiate network
        self.net = LinearRegressorModel(input_dim=13, neurons=neurons, 
                        activations=activations, dropouts=dropouts).double()
        self.net.apply(weights_init_normal)

        # Define our optimiser and loss functions
        self.optimiser = optim.SGD(params=self.net.parameters(), lr=0.01, momentum=0.9)
        self.loss_fn = nn.MSELoss()


        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        return

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
        # train_pipe = Pipeline()

        # Only fit for training
        if training:
            x = self.feature_pipe.fit_transform(x)
            if y is not None:
                y = self.label_pipe.fit_transform(y)
        else:
            x = self.feature_pipe.transform(x)
            if y is not None:
                y = self.label_pipe.transform(y)

        # Change to torch tensor
        x = torch.tensor(x, dtype=torch.float64)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float64)
        return x, (y if torch.is_tensor(y) else None)
        # # Replace this code with your own
        # # Return preprocessed x and y, return None for y if it was None
        # nominal_column = ['ocean_proximity']
        # numeral_columns = x.columns.difference(nominal_column)
        # missing_column = 'total_bedrooms'
        # # Handle Nominal variable
        # one_hot = pd.get_dummies(x[nominal_column])
        # x = x.join(one_hot)
        # x.drop([*nominal_column], axis=1, inplace=True)

        # # Handle Missing values
        # index = x[missing_column].index[x[missing_column].apply(np.isnan)]
        # missing_idx = index.values.tolist()
        # missing_mean = x[missing_column].mean()
        # x.fillna(value=missing_mean, axis=0, inplace=True)
        # # print(missing_mean)

        # # Normalising statistics
        # if training:
        #     # Standardisation
        #     # self.mean = x[numeral_columns].mean()
        #     # self.std = x[numeral_columns].std()
        #     # x_norm = (x[numeral_columns]-self.mean)/self.std
        #     # x[numeral_columns] = x_norm

        #     # Normalisation
        #     self.max = x[numeral_columns].max()
        #     self.min = x[numeral_columns].min()
        #     x_norm = (x[numeral_columns]-self.min)/(self.max-self.min)
        #     x[numeral_columns] = x_norm
        # else:
        #     # Standardisation
        #     # x_norm = (x[numeral_columns]-self.mean)/self.std
        #     # x[numeral_columns] = x_norm

        #     # Normalisation
        #     x_norm = (x[numeral_columns]-self.min)/(self.max-self.min)
        #     x[numeral_columns] = x_norm

        # x = torch.tensor(x.values, dtype=torch.float64)
        # if y is not None:
        #     y = y/100000
        #     y = torch.tensor(y.values, dtype=torch.float64)
        # return x, (y if torch.is_tensor(y) else None)

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

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        dataset = TensorDataset(X, Y)
        batch_size = 500
        # Instantiate dataloader
        train_loader = DataLoader(dataset, batch_size, shuffle=True)

        # Set NN to train mode
        self.net.train()

        ##### Start Training #####
        pbar = tqdm(range(self.nb_epoch))
        for i in pbar :
            # Record train error for each epoch
            train_error = 0
            # Iterating all batches
            for j, train_data in enumerate(train_loader):
                # Clean gradients
                self.optimiser.zero_grad()

                # Forward pass of a batch
                x, y_true = train_data
                y_pred = self.net(x)

                # Backward pass of a batch
                loss = self.loss_fn(y_pred, y_true)
                loss.backward()

                # Record training error
                train_error += loss.item()

                # Update network parameters
                self.optimiser.step()
            # Display training error
            train_error /= len(train_loader)
            pbar.set_postfix({'training loss': train_error})

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

        X, _ = self._preprocessor(x, training = False) # Do not forget
        y_pred = self.net(X)
        return y_pred

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
        dataset = TensorDataset(X, Y)
        batch_size = 500

        # Instantiate dataloader
        test_loader = DataLoader(dataset, batch_size, shuffle=True)

        # Define our loss function
        self.loss_fn = nn.MSELoss()

        # Set NN to eval mode
        self.net.eval()

        error = 0
        pbar = tqdm(enumerate(test_loader))
        for j, train_data in pbar:
            # Predicting label
            x, y_true = train_data
            y_pred = self.net(x)
            # Calculate loss
            loss = self.loss_fn(y_pred, y_true)
            error += loss.item()
        error = error/len(test_loader)

        return error # Replace this code with your own

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



def RegressorHyperParameterSearch(regressor, x, y): 
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
    dummy_pipe = Pipeline([('preprocessor', regressor.feature_pipe),
                            ('regressor', LinearRegression())])
    param_grid = {
    "preprocessor__num__imputer__n_neighbors":[3,5],
    }
    grid_search = GridSearchCV(dummy_pipe, param_grid=param_grid, cv=2)
    x = grid_search.fit(x,y)
    print(x.best_params_)
    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():
    pd.set_option('expand_frame_repr', False)
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
    regressor = Regressor(x_train, nb_epoch = 50)
    # RegressorHyperParameterSearch(regressor, x_train, y_train)
    regressor.fit(x_train, y_train)
    # save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

