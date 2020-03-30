#!/usr/bin/env python
# coding: utf-8

# Import necessary packages
import pandas as pd
import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(os.path.dirname(dir_path))

database = 'GSE122649_RAW' # Specifies the local directory with the sample data from the GSE122649 database
starterFile = 'GSM3477217_ALS-21_counts' # Specifies an arbitrarily chosen file from the GSE122649 database to add the genes DataFrame column
# database = 'GSE124439_RAW' # Specifies the local directory with the sample data from the GSE124439 database
# starterFile = 'GSM3533230_CGND-HRA-00013_counts' # Specifies an arbitrarily chosen file from the GSE124439 database to add the genes DF column

path = ('data' + '/' + database + '/' + starterFile + '.txt') # Specifies the local path to the starterFile
data = pd.read_csv(path, sep = "\s+", header = 0, names = ['genes','counts']) # Reads in the starterFile
df = pd.DataFrame(data['genes']) # Adds the genes column to start a Pandas DataFrame (same for all files)

for filename in os.listdir('data' + '/' + database): # Iterates through each file in the database
    path = ('data' + '/' + database + '/' + filename)
    data = pd.read_csv(path, sep = "\s+", header = 0, names = ['genes','counts'])

    # Creates a log10-scaled version of the reads data for much easier visualization
    logData = []
    for i in data['counts']:
        if(i == 0):
            logData.append(0)
        else:
            logData.append(math.log10(i))

    truncName = filename[:-4]
    df[truncName] = pd.DataFrame(logData) # Adds the read counts for the given sample to the DataFrame

geneIndexedDF = df.set_index('genes') # Makes the genes column the index for the DataFrame

sortedDF = geneIndexedDF.sort_values(by=[starterFile],ascending=False) # Sorts reads descending for the starterFile

print(sortedDF)

# Plots a heatmap of the data
plt.subplots(figsize=(20,20))
ax = sns.heatmap(sortedDF,cmap="YlGnBu")
plt.tight_layout()
plt.savefig("figures/heatmap_db49_full.png")
plt.show()

# A truncated version of the 28,000+ gene database above to examine a small region
# truncated = df.truncate(before = 1, after = 50)
# indexedTrunc = truncated.set_index('genes')
# sortedTrunc = indexedTrunc.sort_values(by=[starterFile],ascending=False)

# plt.subplots(figsize=(10,10))
# ax = sns.heatmap(sortedTrunc,cmap="YlGnBu")
# plt.tight_layout()
# plt.savefig("figures/heatmap_db49_zoomedIn.png")
