# Model Training and Test data Generation
from sklearn.model_selection import train_test_split


class ModelDataGeneration:
    """

    """

    def __init__(self):
        """

        """
        self.df: object = ""

    def generate_train_test_data(self):
        """

        :return:
        """
        X = self.df.values[:, 0:12]  # all rows for columns till 0 to 11
        Y = self.df.values[:, 12]  # all rows for column 12 - The Target Class/variable

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

        return X_train, X_test, y_train, y_test
