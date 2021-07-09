# Accuracy Score Generation
import pickle
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from src.data_acquisition.data_fetch import DataFetch
from src.data_preprocessing.data_preprocess import DataPreprocess
from src.model_data_generation.generate_model_data import ModelDataGeneration


class ModelPrediction:
    """

    """

    def __init__(self):
        """

        """
        self.df: object = ""

    def get_model_data(self):
        """

        :return:
        """
        data_fetch = DataFetch()
        self.df = data_fetch.read_csv()

        data_preprocess = DataPreprocess()
        self.df = data_preprocess.preprocess_data(self.df)
        self.df = data_preprocess.encode_data(self.df)

    def predict_accuracy(self):
        """

        :return:
        """
        model_data_generation = ModelDataGeneration(self.df)
        X_train, X_test, y_train, y_test = model_data_generation.generate_train_test_data()
        dt_clf_gini = DecisionTreeClassifier(criterion="gini",
                                             random_state=100,
                                             max_depth=5,
                                             min_samples_leaf=5)

        dt_clf_gini.fit(X_train, y_train)
        y_predicted_gini = dt_clf_gini.predict(X_test)

        print(f"Predicted Gini Index:{y_predicted_gini}")

        print("Decision Tree using Gini Index\nAccuracy is ",
              accuracy_score(y_test, y_predicted_gini) * 100)

        # Saving the model
        # joblib.dump(dt_clf_gini, 'adults_dtree.pkl')
        # pickle.dump(dt_clf_gini, '..//adults1_dtree.pkl')
        # save data to a file
        with open('..//myfile.pickle', 'wb') as fout:
            pickle.dump(dt_clf_gini, fout)
