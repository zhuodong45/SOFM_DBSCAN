import numpy as np
import csv
from neupy import algorithms, environment
import matplotlib.pyplot as plt

# data for each year
# used to store data
data_2015 = np.array([])
data_2016 = np.array([])
data_2017 = np.array([])

# ============================ pre-processing start ======================================
# columns: 0,3,5,6,7,8,10,9,11
# feature: [country, score, economy, family, health, freedom, generosity, trust(Government), Dystopia residual]
with open('../data/2015.csv', newline='') as csvfile:
    file_2015 = csv.reader(csvfile, delimiter=',', quotechar='|')  # read entire file
    for row in file_2015:  # get data row by row
        row = np.concatenate((row[0:1], row[3:4], row[5:9], row[10:11], row[9:10], row[11:12]))  # combine data base on corresponding index
        data_2015 = np.append(data_2015, row)  # add current row into data_2015

# columns: 0,3,6,7,8,9,11,10,12
# feature: [country, score, economy, family, health, freedom, generosity, trust(Government), Dystopia residual]
with open('../data/2016.csv', newline='') as csvfile:
    file_2016 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file_2016:
        row = np.concatenate((row[0:1], row[3:4], row[6:10], row[11:12], row[10:11], row[12:13]))
        data_2016 = np.append(data_2016, row)

# columns: 0,2,5,6,7,8,9,10,11
# feature: [country, score, economy, family, health, freedom, generosity, trust(Government), Dystopia residual]
with open('../data/2017.csv', newline='') as csvfile:
    file_2017 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file_2017:
        row = np.concatenate((row[0:1], row[2:3], row[5:12]))
        data_2017 = np.append(data_2017, row)

# extract training data and testing data
data_2015 = data_2015.reshape((-1, 9))  # [US,2,3,Canada,5,6,China,8,9] -> [ [US,2,3],
                                        #                                    [Canada,5,6],
                                        #                                    [China,8,9] ]
data_2016 = data_2016.reshape((-1, 9))
data_2017 = data_2017.reshape((-1, 9))

big_data = np.concatenate((data_2015[1:, 1:], data_2016[1:, 1:], data_2017[1:, 1:]), axis=0)  # combine all data into one dataset

big_data = big_data.astype(np.float32)

# ======================================================================================================================

sofm = algorithms.SOFM(
        n_inputs=8,
        n_outputs=5,
        learning_radius=0,
        step=0.25,
        shuffle_data=True,
        weight='sample_from_data',
        verbose=True
)
sofm.train(big_data, epochs=200)
output = sofm.predict(big_data)

# print(output[:100])

# plt.title('Clustering iris dataset with SOFM')
# plt.xlabel('Feature #3')
# plt.ylabel('Feature #4')
#
# ggplot_colors = plt.rcParams['axes.prop_cycle']
# colors = np.array([c['color'] for c in ggplot_colors])
#
#
# plt.scatter(*big_data.T, c=colors[target], s=100, alpha=1)
# cluster_centers = plt.scatter(*sofm.weight, s=300, c=colors[3])
#
# plt.legend([cluster_centers], ['Cluster center'], loc='upper left')
# plt.show()


