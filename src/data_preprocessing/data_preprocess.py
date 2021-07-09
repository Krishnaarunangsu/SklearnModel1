# preprocess data
from src.util.nan_removal import NANRemoval
from sklearn import preprocessing
from typing import Dict, Any


class DataPreprocess:
    """
    Data Preprocessing
    """

    def __init__(self):
        """
        Initialization
        """
        self.df: object = ""

    def preprocess_data(self):
        """

        :rtype: object
        :return:
        """
        # Remove Unnecessary Columns
        self.df.drop(['fnlwgt', 'educational-num'], axis=1)

        # Remove NANs
        nan_removal = NANRemoval(self.df)
        self.df = nan_removal.remove_nan()

        # Club Relationship Column
        self.df["marital-status"].replace(
            {" Divorced": "not married", " Never-married": "not married", " Separated": "not married",
             " Married-spouse-absent": "married",
             " Married-civ-spouse": "married", " Married-AF-spouse": "married", " Widowed": "married"}, inplace=True)

        return self.df

    def encode_data(self) -> object:
        """

        :rtype: object
        :return:
        """
        # data_fetch = DataFetch()
        # self.df = data_fetch.read_csv()

        # self.df = self.preprocess_data()

        category_col = self.df.select_dtypes("object")
        # print(category_col)

        label_encoder = preprocessing.LabelEncoder()

        mapping_dict: Dict[Any, Dict[Any, Any]] = {}
        for column in category_col:
            self.df[column] = label_encoder.fit_transform(self.df[column])
            label_encoder_mapping_classes = dict(
                zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            # print(label_encoder_mapping_classes)
            mapping_dict[column] = label_encoder_mapping_classes

        return self.df
