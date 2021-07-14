import matplotlib.pyplot as plt
import pandas as pd


class ErrorChecker:
    """
    Class that allows isinstance checks, with additional flavor text for Exception messages

    :param check_tuple:
        tuple, length=3 \n
        First index is the name of the object, the second index is the object, and the third index is
        additional flavor text used to create message
        (E.g. check_tuple = ('my_bool', my_bool, 'Keyword argument') -> "Keyword argmument 'my_bool' not of type bool)
    """
    def __init__(self, check_tuple):
        self.check_tuple = check_tuple

    @property
    def check_tuple(self):
        return self._check_tuple

    @check_tuple.setter
    def check_tuple(self, check_tuple):
        # message for if format of check_tuple fails
        bad_tuple_message = "Parameter 'check_tuple' must be a tuple of length=3, " \
                            "where the first index is the string name of the argument, " \
                            "and the second index is the argument to be tested, " \
                            "and the third index is whether the argument is a the argument description."
        # ensure that bool_tuple parameter is properly formatted
        if not isinstance(check_tuple, tuple) or len(check_tuple) != 3:
            raise Exception(bad_tuple_message)
        self._check_tuple = check_tuple

    def _generic_checker(self, string_type, var_type):
        # check if the supposed object is truly it's type, if not raise exception
        if not isinstance(self.check_tuple[1], var_type):
            raise Exception(f"{self.check_tuple[2]} '{self.check_tuple[0]}' must be a {string_type}.")
        # if the supposed object is actually the object, then return it :)
        return self.check_tuple[1]

    def bool_checker(self):
        return self._generic_checker('bool', bool)

    def int_checker(self):
        return self._generic_checker('int', int)

    def dataframe_checker(self):
        if isinstance(self.check_tuple[1], pd.Series):
            self.check_tuple = (self.check_tuple[0], self.check_tuple[1].to_frame(), self.check_tuple[2])
        return self._generic_checker('dataframe', pd.DataFrame)

    def str_checker(self):
        return self._generic_checker('str', str)


def plot_datetime_dataframe(dataframe, *, plot_title='Title', style_context='dark_background'):
    """
    Function that provides a basic plot for any dataframe that has a datetime index.

    :param dataframe:
        dataframe \n
        The dataframe whose data will be plotted.
    :param plot_title:
        str n\
        The desired title for the plot
    :param style_context:
        str n\
        The desired matplotlib style.context
    """
    # set the style context
    with plt.style.context(style_context):
        # if dataframe is actually a series, make dataframe
        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()
        # add label for each column in dataframe
        for df_column in dataframe.columns:
            plt.plot(dataframe[df_column].to_list(), label=df_column.title())
        # add legend, add title, and show plot
        plt.legend()
        plt.title(str(plot_title).title())
        plt.show()


def dataframe_additions(dataframe, method, apply_method_to, *args, **kwargs):
    """
    Function that, for a given dataframe with a datetime index and a requested method,
    adds on that method to the dataframe.
    :param dataframe:
        dataframe n\
        A pandas dataframe with a datetime index.
    :param method:
        str n\
        A string that captures an available method - e.g. 'rolling_average'.
    :param apply_method_to:
        str, list n\
        A single string or list of column names, belonging to the parameter dataframe, to apply the requested method to.
    :param args:
        All additional positional arguments required for a method.
    :param kwargs:
        All additional keyword arguments belonging to a method.
    :return:
        dataframe n\
        Return the original dataframe, plus the columns that had methods applied to them as separate columns.

    Attributes:
        method_list:
        A list of available methods.
    """
    # raise error if lower of method is not in method list
    method = str(method).lower()
    if method not in dataframe_additions.method_list:
        raise Exception(f"Argument passed for second positional 'method' parameter not in accepted methods. "
                        f"Excepted methods are: {dataframe_additions.method_list}.")
    # raise error if dataframe is not a dataframe
    dataframe = ErrorChecker(('dataframe', dataframe, 'First positional argument')).dataframe_checker()
    # check that apply_method_to is list of strings
    if not isinstance(apply_method_to, list):
        apply_method_to = [apply_method_to]
    for i, list_string in enumerate(apply_method_to):
        ErrorChecker((f"apply_method_to[{i}]", list_string, "Not all elements in 'apply_method_to' are strings."))
    # ifs for different methods
    if method == 'rolling_average':
        # rolling average dataframe
        new_dataframe = dataframe.rolling(*args, **kwargs).mean()[apply_method_to]
    if method == 'rate_of_change':
        new_dataframe = dataframe[apply_method_to].diff(*args, **kwargs)
    # re-label new columns to capture method
    new_dataframe.columns = [f"{column}_{method}" for column in apply_method_to]
    # add new dataframe to existing one
    dataframe = pd.merge(dataframe, new_dataframe, how='left', left_index=True,
                         right_index=True)
    new_dataframe = None
    return dataframe


# define method list
dataframe_additions.method_list = ['rolling_average', 'rate_of_change']
