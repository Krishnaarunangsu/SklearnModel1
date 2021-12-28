import matplotlib.pyplot as plt
import pandas as pd


# Fetch the data
def fetch_data():
    """Fetch the Data"""
    data = pd.read_csv("..//simple_linear_regression//data//advertising.csv")
    print(data.head())
    # print(data['TV'].head())
    return data


def plot_regression():
    """
    Plot the regression line
    :return:
    """
    plt.figure(figsize=(14, 3))

    # get the data
    fetched_data = fetch_data()

    # TV
    print(fetched_data['TV'].head())
    plt.subplot(1, 3, 1)
    plt.title('TV-Sales')
    plt.xlabel("TV")
    plt.ylabel("Sales")
    # plt.scatter(fetched_data['TV'], fetched_data['Sales'], 'blue')
    plt.scatter(x=fetched_data['TV'], y=fetched_data['Sales'], color='blue')

    # Radio
    print(fetched_data['Radio'].head())
    plt.subplot(1, 3, 2)
    plt.title('Radio-Sales')
    plt.xlabel("Radio")
    plt.ylabel("Sales")
    plt.scatter(x=fetched_data['Radio'], y=fetched_data['Sales'], color='red')

    # Newspaper
    print(fetched_data['Newspaper'].head())
    plt.subplot(1, 3, 3)
    plt.title('Newspaper-Sales')
    plt.xlabel('Newspaper')
    plt.ylabel('Sales')
    plt.scatter(x=fetched_data['Newspaper'], y=fetched_data['Sales'], color='green')

    plt.show()


def main():
    plot_regression()


if __name__ == "__main__":
    main()
