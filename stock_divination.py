from dateutil.parser import parse
from datetime import datetime, date
import holidays
import pandas_datareader as web
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cpu')


class Stock:
    """
    Returns a dataframe of a stock/data-set retrieved using pandas_datareader.

    :param alias
        str \n
        Stock Alias. Used to alias data-set by, forced to uppercase on initialization
        (e.g. 'WALMART' or 'INFECT').

    :param ticker
        str \n
        Stock/data-set ticker. The recognized official ticker for a stock/data-set
        (e.g. 'WMT', 'INFECTDISEMVTRACKD').
    :param start_date
        str, date, datetime, Timestamp \n
        Start date. If string, will attempt to parse into datetime object
        (e.g. datetime(2000, 1, 1), '1/1/2000', '2020-01-01').
    :param end_date
        str, date, datetime, Timestamp, default date.today() \n
        End date. If string, will attempt to parse into datetime object. Defaults to the current date of user system.
        (e.g. datetime(2000, 1, 1), '1/1/2000', '2020-01-01').
    :param data_source
        str, pandas_reader data_source/api call, default None \n
        The data_source for the stock/data-set. Can be an accepted pandas_datareader API call,
        or an acceptepted pandas_datareader data_source string. If none specificed, or if data_source fails, defaults
        to 'stooq' dailyreader
        (e.g. 'yahoo', 'fred')
    :param use_stooq_reader
        bool, default False \n
        Boolean to force to use of StooqDailyReader if no data_source specified.
    """

    def __init__(self, alias, ticker, start_date, *, end_date=date.today(), data_source=None, use_stooq_reader=False):
        # get use_stooq_reader
        self.use_stooq_reader = use_stooq_reader
        # get dates
        self.start_date = self.check_date(start_date, 'start_date')
        self.end_date = self.check_date(end_date, 'end_date')
        self.check_date_interval()
        # Upper case alias
        self.alias = str(alias).upper()
        # uppercase provided ticker name
        self.ticker = str(ticker).upper()
        # lowercase provided data source (eg "yahoo")
        self.data_source = data_source
        self.features = self.reader_check()

        # if there are no features, raise an error
        if len(self.features) == 0:
            raise Exception(f'Dataset for (ticker={self.ticker}, data_source={self.data_source}) is empty')

    def reader_check(self):
        """
        Checks that a given ticker/data_source can be called. If not, attempts same ticker with StooqDailyReader.
        """
        try:
            if self.use_stooq_reader:
                features = list(web.stooq.StooqDailyReader(self.ticker, self.start_date, self.end_date).read().columns)
                features.sort()
            else:
                features = list(web.DataReader(self.ticker, self.data_source, self.start_date, self.end_date).columns)
                features.sort()
            return features
        except (web._utils.RemoteDataError, NotImplementedError):
            old_datasource = self.data_source
            try:
                self.data_source = 'stooq'
                self.use_stooq_reader = True
                warnings.warn(
                    f"Previous data_source='{old_datasource}' could not read data for ticker='{self.ticker}'. Changed to data_source='stooq'.")
            except NotImplementedError:
                raise Exception(
                    f"Neither original data_source '{old_datasource}' nor backup 'stooq' implemented for ticker '{self.ticker}'")
            except web._utils.RemoteDataError:
                raise Exception(
                    f"Data could not be read for original data_source='{old_datasource}' nor backup data_source='stooq' for ticker '{self.ticker}'")

    def check_date(self, checked_date, date_type):
        ''' Checks that given start/end data can be parsed as a date. '''
        try:
            checked_date = parse(str(checked_date), fuzzy=False).date()
        except ValueError:
            raise Exception(f"{str(date_type)} cannot be converted into datetime.")
        return checked_date

    def check_date_interval(self):
        ''' Checks that given start/end data create time-interval. '''
        # if dates are the same or end date is before start date, raise an error
        if self.start_date == self.end_date:
            raise Exception("Stock attributes 'start_date' and 'end_date' cannot be the same.")
        elif self.start_date > self.end_date:
            raise Exception("Stock attribute 'end_date' must be a later date than attribute 'start_date'.")

    def try_check_date_interval(self):
        ''' Wrapper, to catch when a date has not yet been initialized. '''
        try:
            self.check_date_interval()
        except AttributeError:
            pass

    @property
    def use_stooq_reader(self):
        return self._use_stooq_reader

    @use_stooq_reader.setter
    def use_stooq_reader(self, use_stooq_reader):
        """Setter that ensures use_stooq_reader is a bool. """
        if not isinstance(use_stooq_reader, bool):
            raise Exception("Keyword argument 'use_stooq_reader' must be a boolean.")
        self._use_stooq_reader = use_stooq_reader

    @property
    def alias(self):
        return self._alias

    @alias.setter
    def alias(self, alias):
        self._alias = str(alias).upper()

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        """Setter that ensures that feature set of stock can be found in Stock dataframe. """
        self._features = self.reader_check()
        if not isinstance(features, (list, tuple)):
            features = [features]
        if len(features) > 0 and set(features).issubset(set(self._features)):
            self._features = features
        elif len(features) > 0:
            raise Exception(
                f"Requested features not found for ticker='{self.ticker}' in data_source={self.data_source}")

    @property
    def ticker(self):
        return self._ticker

    @ticker.setter
    def ticker(self, ticker):
        """Setter that forces ticker to required uppercase, and checks that ticker exists for that datasource. """
        self._ticker = str(ticker).upper()
        try:
            self.reader_check()
            self.features = []
        except AttributeError:
            pass

    @property
    def data_source(self):
        return self._data_source

    @data_source.setter
    def data_source(self, data_source):
        """Setter that checks that data_source is callable in pandas_datareader."""
        self._data_source = str(data_source).lower()
        if self.use_stooq_reader:
            if (data_source is not None) and (data_source != 'stooq'):
                warnings.warn(
                    f"Attribute 'use_stooq_reader' is set to 'True', and does not require a passed value in keyword argument 'data_source'.")
        try:
            self.reader_check()
            self.features = []
        except AttributeError:
            pass

    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, start_date):
        """Setter that calls checker to ensure start_date can be converted to date."""
        self._start_date = self.check_date(start_date, 'start_date')
        self.try_check_date_interval()

    @property
    def end_date(self):
        return self._end_date

    @end_date.setter
    def end_date(self, end_date):
        """Setter that calls checker to ensure end_date can be converted to date."""
        self._end_date = self.check_date(end_date, 'end_date')
        self.try_check_date_interval()

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __eq__(self, other_stock):
        # ensure comparison stock is a Stock class, if not return False
        if not isinstance(other_stock, Stock):
            return False
        # a stock is "equal" to another if it's ticker, data source, and features are the same
        return (self.ticker == other_stock.ticker) and (self.data_source == other_stock.data_source)

    def __hash__(self):
        return (hash(repr(self)))

    @property
    def dataframe(self):
        """
        Provides back a pandas_reader dataframe.
        Dataframe is sorted on datetime index, and only includes columns (features) specified by user.
        Upon init, features are all columns associated for the dataframe, but user may remove undesired features.
        (e.g. stock_dataframe = Stock('walmart', 'wmt', '1/1/2000', data_source='yahoo).dataframe())

        :returns dataframe
        """
        if self.use_stooq_reader:
            df_stock = web.stooq.StooqDailyReader(self.ticker, self.start_date, self.end_date).read()
        else:
            df_stock = web.DataReader(self.ticker, self.data_source, self.start_date, self.end_date)
        df_stock = df_stock[self.features]
        return df_stock.sort_index()

    def remove_features(self, feature_list):
        """
        Function that allows user to remove features from a stock, if undesirable

        :param feature_list
            str, list \n
            A single string, or list of string features to be removed.
            (e.g. Stock.remove_features('Close'), Stock.remove_features(['Close', 'Open'])
        """
        # if item isn't a list, make it a list :)
        if not isinstance(feature_list, (list, tuple)):
            feature_list = [feature_list]
        # unique items in list
        feature_list = list(set(feature_list))
        # stringify each element of the list
        feature_list = [str(item) for item in feature_list]
        # check that each element in feature_list is in current self.features (don't remove anything yet)
        for feature in feature_list:
            # if the feature is not in self.features (and can't be removed) raise exception
            if feature not in self.features:
                raise Exception(f"Feature '{feature}' not in self.features, cannot be removed.")
        # if the number of requested feature to be removed is equal to total, raise exception
        if len(feature_list) >= len(self.features):
            raise Exception("Requested removal would remove all features. Stock requires at least one feature.")
        # remove each feature requested
        for feature in feature_list:
            self.features.remove(feature)

    def update_date(self, date_type, new_date):
        """
        Function that allows for alternative change to start/end date.

        :param date_type
            str 'start' or 'end' \n
            Specifies which date attribute to change.

        :param new_date
            str, date, datetime, Timestamp \n
            Date attribute will be updated to.
        """
        # verify date_type argument, must be either 'start' or 'end'
        date_type = str(date_type).lower()
        if date_type not in ['start', 'end']:
            raise Exception(f"Argument for first positional 'date_type' must be 'start' or 'end'.")
        if date_type == 'start':
            self.start_date = new_date
        elif date_type == 'end':
            self.end_date = new_date


class StockCollection:
    '''
    class that defines a collection of stocks, with a defined primary stock, default stocks,
    and options to add more stocks or remove default stocks
    '''
    def __init__(self, target):

        self.target = target
        # initialize default feature set
        self.divinations = [Stock('INFECT', 'INFECTDISEMVTRACKD', target[0].start_date, end_date=target[0].end_date,
                                  use_stooq_reader=False, data_source='fred'),
                            Stock('STRESS', 'STLFSI2', target[0].start_date, end_date=target[0].end_date,
                                  use_stooq_reader=False, data_source='fred'),
                            Stock('DOW_JONES_COMP', '^DJC', target[0].start_date, end_date=target[0].end_date,
                                  use_stooq_reader=True),
                            Stock('NASDAQ_COMP', '^NDQ', target[0].start_date, end_date=target[0].end_date,
                                  use_stooq_reader=True),
                            Stock('SP_500', '^SPX', target[0].start_date, end_date=target[0].end_date,
                                  use_stooq_reader=True)
                            ]
        # set target dataframe, divination dataframe to none then call function to set them
        self.target_dataframe = None
        self.divination_dataframe = None
        self.set_divinations_dataframes()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = self.check_target(target)

    def check_target(self, target):
        # check that target is a tuple
        if not isinstance(target, tuple):
            raise Exception(
                f"'target' must be a tuple of length = 2, where the first index is the target stock, and the second index is the name of the target column within the target stock.")
        # check that target (first index of tuple) is a Stock class
        if not isinstance(target[0], Stock):
            raise Exception(f"First index of 'target' argument must by a class Stock object.")
        # check that the target feature (second index of tuple) is a string that is a column name within the features of the target Stock
        if target[1] not in target[0].features:
            raise Exception(
                f"Value in second index of target tuple not in feature set of first index (the target stock).")
        try:
            if self.divination_dataframe is not None:
                # set new target_dataframe
                self.set_divinations_dataframes(set_target_only=True)
        except AttributeError:
            pass
        # target drives data start and end, and is the target for any analysis
        return target

    def add_divinations(self, divination_list):
        # check if divination_list is a list or tuple, make it a list
        if not isinstance(divination_list, (list, tuple)):
            divination_list = [divination_list]
        elif len(divination_list) == 0:
            raise Exception("'divination_list' is empty.")
        else:
            divination_list = list(set(divination_list))
        # iterate over stocks in divination_list to check if each is class Stock. Don't add anything yet
        for divination in divination_list:
            # if not a stock, raise exception. else, add to feature set
            if not isinstance(divination, Stock):
                raise Exception(f"'{divination}' in 'divination_list' not a Stock Class.")
            # if divination already in divinations, then raise error
            if divination in self.divinations:
                raise Exception(
                    f"Stock(ticker='{divination.ticker}, data_source={divination.data_source} in 'divination_list' already in self.divinations.")
        # if all items are stocks, and not in self.divinations, then add to divinations
        for divination in divination_list:
            # have stock start, end dates mimic target[0]
            divination.update_date('start', self.target[0].start_date)
            divination.update_date('end', self.target[0].end_date)
            self.divinations.append(divination)

    def set_divinations_dataframes(self, *, set_target_only=False, feature_date_cyclical=True, feature_holiday=True,
                                   holiday_set=holidays.US()):
        # get just target dataframe
        target_dataframe = self.target[0].dataframe
        # add cyclical date features if requested, and holiday feature if requested
        target_dataframe = self.dataframe_date_featuring(target_dataframe, feature_date_cyclical, feature_holiday,
                                                         holiday_set)
        # Split the dataframe between the target column and other predicting variables
        divination_dataframe, target_dataframe = target_dataframe.loc[:,
                                                 target_dataframe.columns != self.target[1]], target_dataframe.filter(
            items=[self.target[1]])
        # for the target column dataframe, shift all rows up one (want metrics to predict tomorrows target column)
        target_dataframe = target_dataframe.shift(-1)
        # for the target column dataframe, assign last row to be equal to previous row (just to make sure values are in there)
        target_dataframe.iloc[-1] = target_dataframe.iloc[-2].values
        if not set_target_only:
            # merge target predicting columns, and other divination features on index
            for divination in self.divinations:
                divination_dataframe = pd.merge(divination_dataframe, divination.dataframe, how='left',
                                                left_index=True, right_index=True)
            self.target_dataframe, self.divination_dataframe = target_dataframe.fillna(-1), divination_dataframe.fillna(
                -1)
        else:
            self.target_dataframe = target_dataframe.fillna(-1)

    def remove_divinations(self, alias_list):
        # check if alias_list is a list or tuple
        if not isinstance(alias_list, (list, tuple)):
            alias_list = [alias_list]
        elif len(alias_list) == 0:
            raise Exception("'alias_list' is empty.")
        else:
            alias_list = [str(alias) for alias in list(set(alias_list))]
        # iterate over stocks in stock remove list to check if each is class Stock. Don't remove anything yet
        for alias in alias_list:
            # check if the alias is in any of the divinations. If not, raise an error
            if alias not in [divination.alias for divination in self.divinations]:
                raise Exception(f"No alias of '{alias}' found in self.divinations.")
        # remove the aliases
        for divination, divination_alias in [(divination, divination.alias) for divination in self.divinations]:
            if divination_alias in alias_list:
                self.divinations.remove(divination)

    def show_divinations(self):
        for i, divination in enumerate(self.divinations):
            if i == 0:
                print('_' * 100)
            print(
                f'Divination {i}| alias={divination.alias}, ticker={divination.ticker}, data_source={divination.data_source}, features={divination.features}')
            print('_' * 100)

    def dataframe_date_featuring(self, dataframe, feature_date_cyclical=True, feature_holiday=True,
                                 holiday_set=holidays.US()):

        # function that determines if a given date is a holiday
        def is_holiday(date):
            date = date.replace(hour=0)
            return 1 if (date in holiday_set) else 0

        # function that adds a holiday column to a dataframe give a datetimes index
        def add_holiday_col(df_in):
            return df_in.assign(is_holiday=df_in.index.to_series().apply(is_holiday))

        # function that generates cyclical features, given a datetime column
        def generate_cyclical_features(df_in, cyclisized_name, period, start_num=0):
            kwargs = {
                f'sin_{cyclisized_name}': lambda x: np.sin(2 * np.pi * (df_in[cyclisized_name] - start_num) / period),
                f'cos_{cyclisized_name}': lambda x: np.cos(2 * np.pi * (df_in[cyclisized_name] - start_num) / period)
            }
            return df_in.assign(**kwargs).drop(columns=[cyclisized_name])

        # add date related features (day, month, day, week)
        dataframe = (
            dataframe
                .assign(day=dataframe.index.day)
                .assign(month=dataframe.index.month)
                .assign(day_of_week=dataframe.index.dayofweek)
                .assign(week_of_year=dataframe.index.isocalendar().week)
        )

        if feature_date_cyclical:
            cyclisized = {'day_of_week': [7, 0], 'month': [12, 1], 'week_of_year': [52, 0]}
            # Loop through all column feature names, add additional cyclical features
            for cyclisized_name, cyclisized_range in cyclisized.items():
                dataframe = generate_cyclical_features(
                    dataframe,
                    cyclisized_name,
                    cyclisized_range[0],
                    cyclisized_range[1]
                )

        if feature_holiday:
            dataframe = add_holiday_col(dataframe)

        return dataframe

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __hash__(self):
        return (hash(repr(self)))


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
        # TODO check folder, create if not exist, make sure no dupe
        # model_path = f'C:/Users/cruzm/Documents/models/{self.model.__name__}_{datetime.now().strftime("%Y_%m_%d %H-%M-%S")}'*/

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

        # torch.save(self.model.state_dict(), model_path)

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