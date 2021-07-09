from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
device = torch.device('cpu')


# Define the LSTM class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.__name__ = 'LSTM'
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.__name__ = 'gru'
        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out


# Define the optimization class
class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    # plot losses function
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


def oracle_dataframe(target_dataframe, divination_dataframe, *, test_ratio=0.2, batch_size=64, scalar_type='minmax',
                     nn_class='gru'):
    # function to get particular scalar
    def get_scaler(scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

    def get_model(model, model_params):
        models = {
            "rnn": RNNModel,
            "lstm": LSTMModel,
            "gru": GRUModel,
        }
        return models.get(model.lower())(**model_params)

    def getStockPredictionDataframe(nn_class, input_dim, *, plot_loss=False, output_dim=1, hidden_dim=64, layer_dim=3,
                                    batch_size=64, dropout=0.2, n_epochs=50, learning_rate=1e-3, weight_decay=1e-6):

        def inverse_transform(scaler, df, columns):
            for col in columns:
                df[col] = scaler.inverse_transform(df[col])
            return df

        def format_predictions(predictions, values, df_test, scaler):
            vals = np.concatenate(values, axis=0).ravel()
            preds = np.concatenate(predictions, axis=0).ravel()
            df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
            df_result = df_result.sort_index()
            df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
            return df_result

        # get the model parameters
        model_params = {'input_dim': input_dim,
                        'hidden_dim': hidden_dim,
                        'layer_dim': layer_dim,
                        'output_dim': output_dim,
                        'dropout_prob': dropout}
        # create model
        model = get_model(nn_class, model_params)
        # loss function
        loss_fn = nn.MSELoss(reduction="mean")
        # init optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # get Optimizer class
        opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
        # train class
        opt.train(train_loader, validation_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
        # if user wanted to plot loss, then do it
        if plot_loss:
            opt.plot_losses()
        # unpacl predictions, values
        predictions, values = opt.evaluate(test_loader_single, batch_size=1, n_features=input_dim)
        # save
        df_result = format_predictions(predictions, values, X_test, scaler)

        return df_result

    # ensure test ratio can converted to float, and that the float is between 0 and 1
    try:
        test_ratio = float(test_ratio)
        if not (test_ratio > 0.0 and test_ratio < 1.0):
            raise Exception("'test_ratio' must be be between 0 and 1.")
    except ValueError:
        raise Exception("Unable to convert 'test_ratio' to float.")
    # ensure that batch_size could be converted to integer, and is greater than 1
    try:
        batch_size = int(batch_size)
        if not (batch_size > 1):
            raise Exception("'batch_size' must be an integer greater than one.")
    except ValueError:
        raise Exception("Unable to convert 'batch_size' to integer.")
    accepted_scalars = ['minmax', 'standard', 'maxabs', 'robust']
    scalar_type = str(scalar_type).lower()
    if scalar_type not in accepted_scalars:
        raise Exception(f"'scalar_type' not found in accepted scalar type. Accepted types are in {accepted_scalars}.")
    accepted_nn_classes = ['gru', 'rnn', 'lstm']
    nn_class = str(nn_class).lower()
    if nn_class not in accepted_nn_classes:
        raise Exception(f"'nn_class' not found in accepted nn clasess. Accepted types are in {accepted_nn_classes}.")

    # set testing/train validation ratio
    validation_ratio = test_ratio / (1 - test_ratio)
    #
    y, X = target_dataframe, divination_dataframe
    # divide up the train test data for predicting and target variables
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio,
                                                                    shuffle=False)
    # set the desired scalar
    scaler = get_scaler(scalar_type)
    # Apply scalar transform to predicting train, test, validationidate date
    X_train_arr = scaler.fit_transform(X_train)
    X_validation_arr = scaler.transform(X_validation)
    X_test_arr = scaler.transform(X_test)
    # Apply scalar transform to target train, test, validationidate date
    y_train_arr = scaler.fit_transform(y_train)
    y_validation_arr = scaler.transform(y_validation)
    y_test_arr = scaler.transform(y_test)
    # convert train, test, validationidate data for predicting (X) and target(y) data
    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    validation_features = torch.Tensor(X_validation_arr)
    validation_targets = torch.Tensor(y_validation_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)
    # Create datasets for train, validationidate, and test data
    train = TensorDataset(train_features, train_targets)
    validation = TensorDataset(validation_features, validation_targets)
    test = TensorDataset(test_features, test_targets)
    # Create DataLoader objects
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_single = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    input_dimension = len(X_train.columns)
    # return the dataframe
    df_oracle = getStockPredictionDataframe(nn_class, input_dimension)

    return df_oracle