# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MDEV']
features = data.drop('MDEV', axis=1)

# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

# Produce a matrix for client data
clients_data = [
    [[5, 17, 15]],  # Client 1
    [[4, 32, 22]],  # Client 2
    [[8, 3, 12]]    # Client 3
]

num_neighbors = 5


def nearest_neighbor_price(client_data):
    def find_nearest_neighbor_indexes(client_data, features):  # x is your vector and X is the data set.
        neigh = NearestNeighbors(num_neighbors)
        neigh.fit(features)
        distance, indexes = neigh.kneighbors(client_data)
        return indexes

    indexes = find_nearest_neighbor_indexes(client_data, features)[0]
    sum_prices = []

    print("Showing closest neighbours: ")
    for client_index in indexes:
        print(data.iloc[client_index])
        sum_prices.append(prices[client_index])

    neighbor_avg = np.mean(sum_prices)
    return neighbor_avg

index = 0

for client in clients_data:
    val = nearest_neighbor_price(client)
    index += 1
    print("The predicted {} nearest neighbors price for home {} is: ${:,.2f}".format(num_neighbors, index, val))

