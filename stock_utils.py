
class ErrorChecker:
    """
    Class that allows isinstance checks, with additional flavor text for Exception messages

    :param check_tuple:
        tuple, length=3
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
        # check if the supposed objecy is truly it's type, if not raise exception
        if not isinstance(self.check_tuple[1], var_type):
            raise Exception(f"{self.check_tuple[2]} '{self.check_tuple[0]}' must be a {string_type}.")
        # if the supposed object is actually the object, then return it :)
        return self.check_tuple[1]

    def bool_checker(self):
        return self._generic_checker('bool', bool)

    def int_checker(self):
        return self._generic_checker('int', int)


