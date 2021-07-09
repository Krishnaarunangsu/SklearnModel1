import numpy as np


class NANRemoval:
    """

    """

    def __init__(self, df):
        """

        :param df:
        """
        self.df = df

    def remove_nan(self) -> object:
        """

        :return:
        """
        col_names = self.df.columns

        for column in col_names:
            self.df = self.df.replace("?", np.NAN)

        self.df = self.df.apply(lambda x: x.fillna(x.value_counts().index[0]))

        return self.df
