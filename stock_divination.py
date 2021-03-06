import pandas as pd
from dateutil.parser import parse
from datetime import datetime, date
import holidays
import pandas_datareader as web
import numpy as np
import warnings
import stock_utils
from stock_utils import ErrorChecker


class Stock:
    """
    Returns a dataframe of a stock/data-set retrieved using pandas_datareader.

    :param alias
        str \n
        Stock Alias. Used to alias data-set by, forced to uppercase on initialization
        (e.g. 'WALMART' or 'INFECT').

    :param ticker
        str, list \n
        Stock/data-set ticker. The recognized official ticker for a stock/data-set
        (e.g. 'WMT', 'INFECTDISEMVTRACKD', ['WMT', 'INFECTDISEMVTRACKD']).
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
        or an acceptepted pandas_datareader data_source string. If none specified, or if data_source fails, defaults
        to 'stooq' dailyreader
        (e.g. 'yahoo', 'fred')
    :param dataframe_as_stock:
        pandas dataframe n\
        If, instead of wanting to use pandas_datereader, user can pass in a dataframe as the stock data.
    :param use_stooq_reader
        bool, default False \n
        Boolean to force to use of StooqDailyReader if no data_source specified.

    Attributes:
        alias:
            Equal to parameter alias
        ticker:
            Equal to parameter ticker
        start_date:
            Equal to parameter start_date
        end_date:
            Equal to parameter end_date
        data_source:
            Equal to parameter data_source
        dataframe_as_stock:
            Equal to parameter dataframe_as_stock
        use_stooq_reader:
            Equal to parameter use_stooq_reader
        features:
            A list of strings, columns names belonging to dataframe of Stock/dataset retrieved during pandas_datareader GET call

    """

    def __init__(self,
                 alias,
                 ticker,
                 start_date,
                 *,
                 end_date=date.today(),
                 data_source=None,
                 dataframe_as_stock=None,
                 save_dataframe=True,
                 use_stooq_reader=False):

        # ensure bool keyword arguments are bools, and init if yes
        self.save_dataframe = save_dataframe
        self.use_stooq_reader = use_stooq_reader
        # get dates
        self.start_date = self.check_date(start_date, 'start_date')
        self.end_date = self.check_date(end_date, 'end_date')
        self.check_date_interval()
        # Upper case alias
        self.alias = str(alias).upper()
        # uppercase provided ticker name
        self.ticker = ticker
        # if the Stock will be created from a dataframe
        if dataframe_as_stock is not None:
            self._no_reader = True
            self._unfreeze = True
            self.dataframe = dataframe_as_stock
            self._unfreeze = False
        else:
            self._no_reader = False
            self._unfreeze = False
            self.dataframe = None
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
            # use either stooq_reader or standard reader depending on use_stooq_reader
            if self.use_stooq_reader:
                df = web.stooq.StooqDailyReader(self.ticker, self.start_date, self.end_date).read()
            else:
                df = web.DataReader(self.ticker, self.data_source, self.start_date, self.end_date)
            # save features
            features = list(df.columns)
            features.sort()
            # if save_dataframe is true, save dataframe to attribute
            if self.save_dataframe:
                self._unfreeze = True
                self.dataframe = df
                self._unfreeze = False
            # return features
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
        """ Checks that given start/end data can be parsed as a date. """
        try:
            checked_date = parse(str(checked_date), fuzzy=False).date()
        except ValueError:
            raise Exception(f"{str(date_type)} cannot be converted into datetime.")
        return checked_date

    def check_date_interval(self):
        """ Checks that given start/end data create time-interval. """
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
    def alias(self):
        return self._alias

    @alias.setter
    def alias(self, alias):
        self._alias = str(alias).upper()

    @property
    def save_dataframe(self):
        return self._save_dataframe

    @save_dataframe.setter
    def save_dataframe(self, save_dataframe):
        self._save_dataframe = ErrorChecker(('save_dataframe', save_dataframe, 'Stock Class Attribute')).bool_checker()

    @property
    def use_stooq_reader(self):
        return self._use_stooq_reader

    @use_stooq_reader.setter
    def use_stooq_reader(self, use_stooq_reader):
        self._use_stooq_reader = ErrorChecker(('use_stooq_reader',
                                               use_stooq_reader,
                                               'Stock Class Attribute')).bool_checker()

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        """Setter that ensures that feature set of stock can be found in Stock dataframe. """
        if not self._no_reader:
            self._features = self.reader_check()
        else:
            self._features = features
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
        if isinstance(ticker, list):
            self._ticker = [str(ticker_item).upper() for ticker_item in ticker]
        else:
            self._ticker = str(ticker).upper()
        try:
            if not self._no_reader:
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
                    f"Attribute 'use_stooq_reader' is set to 'True', "
                    f"and does not require a passed value in keyword argument 'data_source'.")
        try:
            if not self._no_reader:
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
        # check if dataframes can be equal. if so, good
        if self.dataframe.equals(other_stock.dataframe):
            return True
        # a stock is "equal" to another if it's ticker, data source, dates, and features are the same
        return (self.ticker == other_stock.ticker) and (self.data_source == other_stock.data_source) and \
               (self.start_date == other_stock.end_date) and (self.end_date == other_stock.end_date)

    def __hash__(self):
        return (hash(repr(self)))

    @property
    def dataframe(self):
        """Setter that prevents dataframe from being set outside of initial call"""
        try:
            return self._dataframe
        except AttributeError:
            return None

    @dataframe.setter
    def dataframe(self, dataframe):
        # if no pandas reader, but rather a dataframe
        if self._no_reader:
            if self._unfreeze:
                # check to ensure it can be a dataframe
                dataframe = ErrorChecker(('dataframe_as_stock', dataframe, 'Keyword argument')).dataframe_checker()
                dataframe[dataframe.columns[0]] = pd.to_datetime(dataframe[dataframe.columns[0]])
                dataframe.set_index(dataframe.columns[0], inplace=True)
                dataframe.sort_index()
                # filter by dates
                dataframe = dataframe.loc[(dataframe.index >= pd.Timestamp(self.start_date)) &
                                          (dataframe.index <= pd.Timestamp(self.end_date))]
                # set attributes
                self._dataframe = dataframe
                self.data_source = 'local'
                self.features = list(dataframe.columns)
        try:
            if self._unfreeze:
                self._dataframe = dataframe
        except AttributeError:
            pass

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

    def get_dataframe(self):
        """
        Provides back a pandas_reader dataframe, on demand.
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

    def add_columns(self, method, apply_method_to, *args, **kwargs):
        """
        Function that will add on dataframe columns, using methods defined in stock_utils.

        :param method:
            str n\
            The method, as defined in stock_utils.
        :param apply_method_to:
            str, list n\
            The columns whose data the requested method will be applied to.
        :param args:
            Any additional required positional arguments for the method.
        :param kwargs:
            And additional keyword arguments for the method.
        :return:
            dataframe n\
            The original dataframe, with additional columns added reflecting the method.

        """
        method = str(method).lower()
        # check that requested method is actually available
        if method not in stock_utils.dataframe_additions.method_list:
            raise Exception(f"Argument '{method}' passed into positional argument 'method' not in current accepted "
                            f"method list. "
                            f"Current available methods are: {stock_utils.dataframe_additions.method_list}.")
        # add requested columns to dataframe
        self._unfreeze = True
        self.dataframe = stock_utils.dataframe_additions(self.dataframe, method, apply_method_to, *args, **kwargs)
        self._unfreeze = False

    def plot_dataframe(self):
        """
        Class method that shows plot for Stock self.dataframe attribute
        """
        # check if dataframe has been pulled. If not, pull and save it to attribute
        if self.dataframe is None:
            self._unfreeze = True
            self.dataframe = self.get_dataframe()
            self._unfreeze = False
        # call stock_utils plot_datetime_dataframe
        stock_utils.plot_datetime_dataframe(self.dataframe, plot_title='Stock Dataframe')

class StockCollection:
    """
    Allows user to "collect" stocks, with a target Stock and target Stock feature (column),
    and then a group of divination Stocks used as additional data to predict the Stock feature.

    :param target
        tuple \n
        A tuple of (Stock, Stock.feature), in that order.
        (e.g. (Stock('walmart', 'wmt', '1/1/2000', data_source='yahoo'), 'Close'))
    :param default_divinations
        bool \n
        Boolean value, controlling if default divinations - a collection of FRED and index data - is added on init.
        (e.g. default_divinations=True)

    :param go_backwards:
        int \n
        Allows user to shift a target dataframes values upwards on the same datetime index, by [go_backwards] amount.
        (e.g. go_backwards = 2)
    Attributes:
        target:
            Equal to target parameter
        go_backwards:
            Equal to go_backwards parameter
        divinations:
            list of Stock class \n
            A list of other Stock classes, whose dataframes will be used as additional predicting data
        target_dataframe:
            dataframe \n
            A dataframe constructed based off of target Stock pandas_reader dataframe,
            extracting only the target column, then shifting all values up a datetime index
            - to align all other current data with the "future" target info.
        divination_dataframe:
            dataframe \n
            A dataframe constructed based off of all non-target column info in target Stock pandas_datareader dataframe,
            as well as all other divination data.
                - If pandas_datareader only has one column and there are not additional divinations,
                then the divination_dataframe will be comprised of only day/time featuring columns

    """
    def __init__(self, target, default_divinations=False):
        # ensure start_divination is a bool.. If not, raise error
        if not isinstance(default_divinations, bool):
            raise Exception("Keyword argument 'default_divinations' must be a boolean.")
        self.target = target
        # if start_divination=True, init default divinations
        if not default_divinations:
            self.divinations = []
        elif default_divinations:
            # initialize default feature set
            self.divinations = [Stock('INFECT', 'INFECTDISEMVTRACKD', target[0].start_date, end_date=target[0].end_date,
                                      data_source='fred'),
                                Stock('STRESS', 'STLFSI2', target[0].start_date, end_date=target[0].end_date,
                                      data_source='fred'),
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
        """Back end function that checks for a given stock, and its target column, the stock and column are accessible
        through a pandas_datareader call."""
        # check that target is a tuple
        if not isinstance(target, tuple):
            raise Exception(
                f"'target' must be a tuple of length = 2, where the first index is the target stock, "
                f"and the second index is the name of the target column within the target stock.")
        # check that target (first index of tuple) is a Stock class
        if not isinstance(target[0], Stock):
            raise Exception(f"First index of 'target' argument must by a class Stock object.")
        # check that the target feature (second index of tuple) is a string that is a column name
        # within the features of the target Stock
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
        """
        Class method that allows a user to add 'divinations' - i.e. other Stock Class instances - to the StockCollection.

        :param divination_list:
            list, Stock \n
            Single, or list of initialized Stock instances to be added to the StockCollection instance.
            (e.g. my_stock_collection.add_divinations(Stock('walmart', 'wmt', '1/1/2000'))

        """
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
                    f"Stock(ticker='{divination.ticker}, data_source={divination.data_source} "
                    f"in 'divination_list' already in self.divinations.")
        # if all items are stocks, and not in self.divinations, then add to divinations
        for divination in divination_list:
            # have stock start, end dates mimic target[0]
            divination.update_date('start', self.target[0].start_date)
            divination.update_date('end', self.target[0].end_date)
            self.divinations.append(divination)

    def set_divinations_dataframes(self, *, set_target_only=False, feature_date_cyclical=True, feature_holiday=True,
                                   holiday_set=holidays.US()):
        """
        Class method that, for a StockCollection instance, sets the attributes target_dataframe and divination_dataframe.

        :param set_target_only:
            bool \n
            If True, then only the target_dataframe will be set
        :param feature_date_cyclical:
            bool \n
            If True, will add cyclical date features, where a given time is converted to a value on the unit circle,
            based on the given interval.
        :param feature_holiday:
            bool \n
            If True, then holiday feature column will be added
            - where holidays are based on argument passed to holiday_set parameter
        :param holiday_set:
            holidays, default=holidays.US() n\
            An instance from the holiday library, representing a geograpical set of holidays.

        """
        # get just target dataframe
        if self.target[0].dataframe is not None:
            target_dataframe = self.target[0].dataframe
        else:
            target_dataframe = self.target[0].get_dataframe
        # add cyclical date features if requested, and holiday feature if requested
        target_dataframe = self.dataframe_date_featuring(target_dataframe, feature_date_cyclical, feature_holiday,
                                                         holiday_set)
        # Split the dataframe between the target column and other predicting variables
        divination_dataframe, target_dataframe = target_dataframe.loc[:,
                                                target_dataframe.columns != self.target[1]], \
                                                target_dataframe.filter(items=[self.target[1]])

        if not set_target_only:
            # merge target predicting columns, and other divination features on index
            for divination in self.divinations:
                if divination.dataframe is None:
                    df = divination.get_dataframe()
                else:
                    df = divination.dataframe
                # rename each dataframe column appending _ticker
                df.columns = [f"{column}_{divination.ticker}" for column in df.columns]
                # merge new divination_dataframe into old one
                divination_dataframe = pd.merge(divination_dataframe, df, how='left',
                                                left_index=True, right_index=True)
            self.target_dataframe, self.divination_dataframe = target_dataframe.fillna(-1), divination_dataframe.fillna(
                -1)
        else:
            self.target_dataframe = target_dataframe.fillna(-1)

    def remove_divinations(self, alias_list):
        """
        Class method that, for a single alias string or list of alias strings, deletes those Stocks what have that
        alias from self.divinations

        :param alias_list:
            str, list \n
            The single stirng, or list of aliases to be removed. Method will look at each Stock instance saved in
            self.divinations attribute, and if alias is equal to the divinations Stock instance, removes that Stock.

        """
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
        """
        Class method that prints off each Stock instance in self.divinations attribute.

        :return: divination text:
            formatted text for each Stock instance present in self.divinations.
        """
        for i, divination in enumerate(self.divinations):
            if i == 0:
                print('_' * 100)
            print(
                f'Divination {i}| alias={divination.alias}, ticker={divination.ticker}, data_source={divination.data_source}, features={divination.features}')
            print('_' * 100)

    def dataframe_date_featuring(self, dataframe, feature_date_cyclical=True, feature_holiday=True,
                                 holiday_set=holidays.US()):
        """
        Class function that takes in a given dataframe, and returns the dataframe with additional date-related
        feature columns.

        :param dataframe:
            dataframe \n
            The dataframe that will have additional features added to it.
        :param feature_date_cyclical:
            bool \n
            If True, then adds cyclical feature dates. See StockCollection.set_divinations_dataframe() documentation
            for more information.
        :param feature_holiday:
            bool \n
            If True, adds a holiday feature set.  See StockCollection.set_divinations_dataframe() documentation
            for more information.
        :param holiday_set:
            holidays, default=holidays.US() n\
            An instance from the holiday library, representing a geograpical set of holidays.
        :return: featured dataframe:
            The dataframe passed in, with the additional date related feature columns added to it.
        """

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

    def plot_target_dataframe(self):
        """
        Class method that shows plot of StockCollection.target_dataframe attribute.
        """
        # call stock_utils plot_datetime_dataframe
        stock_utils.plot_datetime_dataframe(self.target_dataframe, plot_title='Target Dataframe')

    def plot_divination_dataframe(self):
        """
        Class method that shows plot of StockCollection.target_dataframe attribute.
        """
        # call stock_utils plot_datetime_dataframe
        stock_utils.plot_datetime_dataframe(self.divination_dataframe, plot_title='Divination Dataframe')

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __hash__(self):
        return hash(repr(self))
