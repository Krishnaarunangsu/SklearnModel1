import json
import pandas as pd
from custom_exceptions.custom_exception import FileDetailsNotExist


# Data Fetch Class
class DataFetch:
    """
    Data Fetch
    """

    def __init__(self) -> object:
        """

        """
        self.file_url = ""
        # https://www.loekvandenouweland.com/content/using-json-config-files-in-python.html
        with open("../../config/config.json") as config_file:
            data = json.load(config_file)
            print(data)
        try:
            if data is not None:
                print(data)
                self.file_url = data["file_url"]
                print(self.file_url)
            else:
                raise FileDetailsNotExist
        except FileDetailsNotExist:
            print(f"File Detail does not exist")

    def read_csv(self):
        """

        """
        print(f"File URL from config file:{self.file_url}")
        df = pd.read_csv(
            "https://raw.githubusercontent.com/hoshangk/machine_learning_model_using_flask_web_framework"
            "/master/adult.csv")
        return df


if __name__ == "__main__":
    data_fetch = DataFetch()
    data_fetch.read_csv()
