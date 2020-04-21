# Import necessary packages
import pandas as pd
import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn import tree
from sklearn.cross_decomposition import PLSRegression
from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
import scipy as sp
import scipy.stats as stats
from statistics import mean

# Set the working directory so that code, data, and figures can be easily accessed
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(os.path.dirname(dir_path))

database = 'GSE124439_RAW' # Specifies the local directory with the sample data from the GSE124439 database
starterFile = 'GSM3533230_CGND-HRA-00013_counts' # Specifies an arbitrarily chosen file from the GSE124439 database to add the genes DF column

path = ('data' + '/' + database + '/' + starterFile + '.txt') # Specifies the local path to the starterFile
data = pd.read_csv(path, sep = "\s+", header = 0, names = ['genes','counts']) # Reads in the starterFile
df = pd.DataFrame(data['genes']) # Adds the genes column to start a Pandas DataFrame (same for all files)

for filename in os.listdir('data' + '/' + database): # Iterates through each file in the database
    path = ('data' + '/' + database + '/' + filename)
    data = pd.read_csv(path, sep = "\s+", header = 0, names = ['genes','counts'])

    regData = []
    for i in data['counts']:
        regData.append(i)

    # Creates a log10-scaled version of the reads data for much easier visualization
    #logData = []
    #for i in data['counts']:
    #    if(i == 0):
    #        logData.append(0)
    #    else:
    #        logData.append(math.log10(i))

    truncName = filename[:-4]
    df[truncName] = pd.DataFrame(regData) # Adds the read counts for the given sample to the DataFrame

geneIndexedDF = df.set_index('genes') # Makes the genes column the index for the DataFrame

sortedDF = geneIndexedDF.sort_values(by=[starterFile],ascending=False) # Sorts reads descending for the starterFile

# print(sortedDF)

# Plots a heatmap of the data
plt.subplots(figsize=(20,20))
ax = sns.heatmap(sortedDF,cmap="YlGnBu")
plt.tight_layout()
plt.savefig("figures/heatmap_db39_full.png")
# plt.show()


# Creates and normalizes a transpose of the DataFrame for use in analyses
df_T = df.T
df_T.columns = df_T.iloc[0]
df_T = df_T.drop('genes')

sc = StandardScaler()
temp = sc.fit_transform(df_T)
df_T = pd.DataFrame.from_records(temp, index = df_T.index, columns = df_T.columns)

# print(df_T)


# Load the data matrix describing what each sample is
matrixPath = ('data/patient_info_matrix.xlsx') # Specifies the local path to the matrix of information
matrixData = pd.read_excel(matrixPath, header=0) # Reads in the starterFile
# print(matrixData.head())

# print(list(dict.fromkeys(matrixData['Gender'])))
genderInfo = list(matrixData['Gender'])
# print(genderInfo)

# print(list(dict.fromkeys(matrixData['Subject Group'])))
subjectGroup = list(matrixData['Subject Group'])

# print(list(dict.fromkeys(matrixData['ALS NMF subtype**'])))
molSubtype = list(matrixData['ALS NMF subtype**'])


genderINTfo = []
for i in genderInfo:
    if (i == 'Female'):
        genderINTfo.append(0)
    else:
        genderINTfo.append(1)
# print(genderINTfo)

subGroupInt = []
for i in subjectGroup:
    if (i == 'Non-Neurological Control'):
        subGroupInt.append(0)
    elif (i == 'Other Neurological Disorders'):
        subGroupInt.append(1)
    else:
        subGroupInt.append(2)
# print(subGroupInt)

molSubtypeInt = []
for i in molSubtype:
    if (i == 'ALS-Ox'):
        molSubtypeInt.append(0)
    elif (i == 'ALS-TE'):
        molSubtypeInt.append(1)
    elif (i == 'ALS-Glia'):
        molSubtypeInt.append(2)
    else:
        molSubtypeInt.append(3)
# print(molSubtypeInt)


# Perform a PCA on the data
X = df_T

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

pca = PCA(n_components = 10)
pcaScores = pca.fit_transform(X_scaled)

# print('# of scores matrix rows = ' + str(len(pcaScores)))
# print('# of scores matrix columns = ' + str(len(pcaScores[0])))

pcaLoadings = pca.components_
# print(pcaLoadings)
# print('# of loadings matrix rows = ' + str(len(pcaLoadings)))
# print('# of loadings matrix columns = ' + str(len(pcaLoadings[0])))

explained_variance = pca.explained_variance_ratio_
# print(explained_variance)

# Calculate the cumulative explained variance as more components are included
num_com = []
com_counter = 0
cum_exvar = []
cum_counter = 0
for i in explained_variance:
    com_counter = com_counter + 1
    num_com.append(com_counter)
    cum_counter = cum_counter + i
    cum_exvar.append(cum_counter)
# print(cum_exvar)

# Plot the cumulative explained variance vs. the # of included components
plt.plot(num_com,cum_exvar,'--o')
plt.xlabel('Number of Included Components')
plt.ylabel('Total Variance Explained')
plt.xticks(num_com,num_com)
plt.yticks(np.linspace(0.4,1.0,7))
plt.tight_layout()
plt.savefig('figures/pca10_variance_explained.png', bbox='tight')

# Plot the first two PCA principal components against one another
fPC1 = []
fPC2 = []
mPC1 = []
mPC2 = []
i = 0
while(i < len(genderInfo)):
    if(genderInfo[i] == 'Female'):
        fPC1.append(pcaScores[i,0])
        fPC2.append(pcaScores[i,1])
    else:
        mPC1.append(pcaScores[i,0])
        mPC2.append(pcaScores[i,1])
    i = i + 1

fig, ax = plt.subplots()
ax.scatter(fPC1, fPC2, c='magenta', s=8, label='F')
ax.scatter(mPC1, mPC2, c='blue', s=8, label='M')
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.legend()
plt.tight_layout()
plt.savefig('figures/pc1_pc2.png', bbox='tight')
# plt.show()


# Create a copy of the transposed DataFrame that can be expanded with more features
df_T_exp = df_T
df_T_exp['Gender'] = genderINTfo
df_T_exp['SubjectGroup'] = subGroupInt
df_T_exp['MolSubtype'] = molSubtypeInt

# Dividing the dataset by gender
femaleData = df_T_exp.loc[df_T_exp['Gender'] == 0]
femaleData_SG = list(femaleData['SubjectGroup'])
femaleData_MS = list(femaleData['MolSubtype'])
femaleData = femaleData.drop(['Gender','SubjectGroup','MolSubtype'], axis=1)

maleData = df_T_exp.loc[df_T_exp['Gender'] == 1]
maleData_SG = list(maleData['SubjectGroup'])
maleData_MS = list(maleData['MolSubtype'])
maleData = maleData.drop(['Gender','SubjectGroup','MolSubtype'], axis=1)

# Dividing the dataset by the subject group (control, ALS, other neurological disorder)
controlData = df_T_exp.loc[df_T_exp['SubjectGroup'] == 0]
controlData_MF = list(controlData['Gender'])
controlData_MS = list(controlData['MolSubtype'])
controlData = controlData.drop(['Gender','SubjectGroup','MolSubtype'], axis=1)

otherNDData = df_T_exp.loc[df_T_exp['SubjectGroup'] == 1]
otherNDData_MF = list(otherNDData['Gender'])
otherNDData_MS = list(otherNDData['MolSubtype'])
otherNDData = otherNDData.drop(['Gender','SubjectGroup','MolSubtype'], axis=1)

alsData = df_T_exp.loc[df_T_exp['SubjectGroup'] == 2]
alsData_MF = list(alsData['Gender'])
alsData_MS = list(alsData['MolSubtype'])
alsData = alsData.drop(['Gender','SubjectGroup','MolSubtype'], axis=1)

# Dividing the dataset by the molecular ALS subtypes
alsOXData = df_T_exp.loc[df_T_exp['MolSubtype'] == 0]
alsOXData_MF = list(alsOXData['Gender'])
alsOXData_SG = list(alsOXData['SubjectGroup'])
alsOXData = alsOXData.drop(['Gender','SubjectGroup','MolSubtype'], axis=1)

alsTEData = df_T_exp.loc[df_T_exp['MolSubtype'] == 1]
alsTEData_MF = list(alsTEData['Gender'])
alsTEData_SG = list(alsTEData['SubjectGroup'])
alsTEData = alsTEData.drop(['Gender','SubjectGroup','MolSubtype'], axis=1)

alsGliaData = df_T_exp.loc[df_T_exp['MolSubtype'] == 2]
alsGliaData_MF = list(alsGliaData['Gender'])
alsGliaData_SG = list(alsGliaData['SubjectGroup'])
alsGliaData = alsGliaData.drop(['Gender','SubjectGroup','MolSubtype'], axis=1)


# Whole-data PLS-DA by gender
numComp = 9
plsr = PLSRegression(n_components=numComp, scale=False)
plsr.fit(df_T, genderINTfo)

# print(plsr.x_scores_) # scores
# print(plsr.x_scores_.shape) # scores shape
# print(plsr.x_weights_) # loadings
# print(plsr.x_weights_.shape) # loadings shape

scores = pd.DataFrame(plsr.x_scores_)
# print(scores)
scores.index = df_T.index

plsColumnNames = []
for i in range(numComp):
    plsColumnNames.append('LV ' + str(i + 1))
scores.columns = plsColumnNames

colormap = {
    0: '#ff0000',  # Red
    1: '#0000ff',  # Blue
}

gendLabels = ['Female', 'Male']
scores['Gender'] = genderINTfo
groups = scores.groupby('Gender')

figure(num=None, figsize=(6.6,5.25), dpi=100)
for name, group in groups:
    plt.scatter(group['LV 1'], group['LV 2'], s=50, alpha=0.7, c=colormap[name], label=gendLabels[name])
plt.xlabel('Scores on LV 1, whole dataset separated by gender')
plt.ylabel('Scores on LV 2, whole dataset separated by gender')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('figures/plsda_combined.png', bbox='tight')
# plt.show()

plt.clf()
scores.drop('Gender', axis=1)


# Specifically comparative PLS-DAs
datasets = [femaleData, maleData, controlData, otherNDData, alsData, alsOXData, alsTEData, alsGliaData]
datasetLabels = ['Female Subjects', 'Male Subjects', 'Control Subjects', 'Other Neurological Subjects',
                'ALS Subjects', 'ALS OX Subtype Subjects', 'ALS TE Subtype Subjects', 'ALS Glial Subtype Subjects']
separators = [[femaleData_SG, femaleData_MS], [maleData_SG, maleData_MS],
             [controlData_MF, controlData_MS], [otherNDData_MF, otherNDData_MS], [alsData_MF, alsData_MS],
             [alsOXData_MF, alsOXData_SG], [alsTEData_MF, alsTEData_SG], [alsGliaData_MF, alsGliaData_SG]]
separatorLabels = [['Subject Group', 'Molecular Subtype'], ['Subject Group', 'Molecular Subtype'],
                  ['Gender', 'Molecular Subtype'], ['Gender', 'Molecular Subtype'], ['Gender', 'Molecular Subtype'],
                  ['Gender', 'Subject Group'], ['Gender', 'Subject Group'], ['Gender', 'Subject Group']]
legendLabels = [['Female', 'Male'], ['Control', 'Non-ALS NeuroDisorder', 'ALS'],
                         ['ALS-OX', 'ALS-TE', 'ALS-Glia', 'Non-ALS']]

expVar = pd.DataFrame()
topGenes = pd.DataFrame()

currData = 0
for i in datasets:
    dataset = i
    datasetName = datasetLabels[currData]

    currSep = 0
    for j in separators[currData]:
        separator = j
        separatorName = separatorLabels[currData][currSep]

        if(separatorName == 'Gender'):
            legendLabel = legendLabels[0]
        elif(separatorName == 'Subject Group'):
            legendLabel = legendLabels[1]
        else:
            legendLabel = legendLabels[2]

        numComp = 9
        plsr = PLSRegression(n_components=numComp, scale=False)
        plsr.fit(dataset, separator)

        # print(plsr.x_scores_) # scores
        # print(plsr.x_scores_.shape) # scores
        # print(plsr.x_weights_) # loadings
        # print(plsr.x_weights_.shape) # loadings

        scores = pd.DataFrame(plsr.x_scores_)
        # print(scores)
        scores.index = dataset.index

        plsColumnNames = []
        for i in range(numComp):
            plsColumnNames.append('LV ' + str(i + 1))
        scores.columns = plsColumnNames

        colormap = {
            0: '#ff0000',  # Red
            1: '#0000ff',  # Blue
            2: '#40e0d0',  # Cyan
            3: '#b19cd9',  # Purple
        }

        scores[separatorName] = separator
        groups = scores.groupby(separatorName)
        figure(num=None, figsize=(6.6,5.25), dpi=100)
        for name, group in groups:
            plt.scatter(group['LV 1'], group['LV 2'], s=50, alpha=0.7, c=colormap[name], label=legendLabel[name])
        plt.xlabel('Scores on LV 1, ' + datasetName + ' separated by ' + separatorName)
        plt.ylabel('Scores on LV 2, ' + datasetName + ' separated by ' + separatorName)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('figures/plsda_' + datasetName.replace(" ", "") + '_' +
                    separatorName.replace(" ", "") + '.png', bbox='tight')
        # plt.show()
        plt.clf()
        scores.drop(separatorName, axis=1)


        # How much variance does each LV explain?
        total_variance_in_x = np.var(dataset, axis = 0)
        tv = total_variance_in_x.sum()
        variance_in_x = np.var(plsr.x_scores_, axis = 0)
        fractions_of_explained_variance = variance_in_x / tv
        expVar[(datasetName + '_' + separatorName).replace(" ", "")] = fractions_of_explained_variance


        # Extract the genes contributing the most to the model
        model_coeff = plsr.coef_ # The coefficient of contribution of each gene to the PLS-DA model
        absval_coeff = np.abs(model_coeff) # Takes the absolute value of the coefficients to compare contribution magnitudes
        sort_ind1 = absval_coeff.argsort(axis=0) # Yields the indices, sorted ascendingly, for genes contributing the most
        sorted_vals = absval_coeff[sort_ind1][::-1].squeeze() # Yields the contribution values of genes, sorted descendingly
        sort_ind2 = sort_ind1[::-1] # Yields the indices, sorted descendingly, for genes contributing the most

        df_plsda_res = pd.DataFrame(data=list(sorted_vals), columns=['Contribution Coeff. Mag.'])
        df_plsda_res['Gene Indices'] = sort_ind2
        genes = list(dataset.columns)
        modelContributors = []
        for i in sort_ind2:
            modelContributors.append(genes[i[0]])
        # print(modelContributors[:5])

        topGenes[(datasetName + '_' + separatorName).replace(" ", "")] = modelContributors[:25]
        # df_plsda_res['Genes'] = modelContributors
        # df_plsda_res = df_plsda_res.set_index('Genes')
        # print(df_plsda_res)


        # Generate a plot of model MSE vs. no. of included components for each PLS-DA model as a check on numComp
        n = len(dataset)

        # 10-fold CV, with shuffle
        kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

        mse = []

        for i in np.arange(1, 20):
            pls = PLSRegression(n_components = i)
            score = model_selection.cross_val_score(pls, dataset, separator, cv=kf_10, scoring='neg_mean_squared_error').mean()
            mse.append(-score)

        plt.plot(np.arange(1, 20), np.array(mse), '-v')
        plt.xlabel('Number of components in PLS-DA')
        plt.ylabel('MSE, ' + datasetName + ' sep by ' + separatorName)
        plt.xlim(xmin=-1)
        plt.savefig('figures/plsdaMSE_' + datasetName.replace(" ", "") + '_' +
                    separatorName.replace(" ", "") + '.png', bbox='tight')
        plt.clf()

        currSep = currSep + 1
    currData = currData + 1

expVar.index = plsColumnNames
# print(expVar)
# print(topGenes)

# SHIFT FROM PCA / PLS-DA TO MANN-WHITNEY & DEG ANALYSIS
df_scale = sortedDF.T #creates copy of the transposed dataframe

#Scales all counts to housekeeper gene Glucose-6-phosphate isomerase
#Eisenberg and Levanon, 2013
for i in df_scale:
    df_scale[i] = df_scale[i]/df_scale['GPI']

#Removes all genes which have more than 50% of samples with 0 counts
lst = []
for i in df_scale:
    count = 0
    for j in df_scale[i]:
        #print (j)
        if j == 0:
            count += 1
    if count > 0.5*len(i):
        lst.append(i)
df_scale.drop(columns = lst, inplace = True)

#Sorts DataFrame by ID number
df_scale.sort_index(inplace=True)

# Load the data matrix describing what each sample is
matrixPath = ('data/patient_info_matrix.xlsx') # Specifies the local path to the matrix of information
matrixData = pd.read_excel(matrixPath, header=0) # Reads in the starterFile
matrixData.sort_values(by=['RNA-seq ID'])
matrixData['Gender'].factorize()[0]

#Assigns gender adn Disease Group to each sample
df_scale['gender'] = matrixData['Gender'].factorize()[0] # 0 is female, 1 is male
df_scale['group'] = matrixData['Subject Group'].factorize()[0] # 0 is ALS, 1 is control

#Initializes new dataframes for each demographic group
df_T = df_scale
df_ALS_male = pd.DataFrame(index=df_T.columns)
df_ALS_female = pd.DataFrame(index=df_T.columns)
df_cont_male = pd.DataFrame(index=df_T.columns)
df_cont_female = pd.DataFrame(index=df_T.columns)

#Populates demographic based dataframes
df = df_T.T
for i in df:
    if df.loc['gender',i] == 1 and (df.loc['group',i] == 1 or df.loc['group',i] == 3):
        df_cont_male[i] = df[i]
    elif df.loc['gender',i] == 1 and (df.loc['group',i] == 0 or df.loc['group',i] == 2):
        df_ALS_male[i] = df[i]
    elif df.loc['gender',i] == 0 and (df.loc['group',i] == 1 or df.loc['group',i] == 3):
        df_cont_female[i] = df[i]
    elif df.loc['gender',i] == 0 and (df.loc['group',i] == 0 or df.loc['group',i] == 2):
        df_ALS_female[i] = df[i]

#Transposes demographic dataframes for iteration
df_maleALS = df_ALS_male.T
df_femaleALS = df_ALS_female.T
df_maleCont = df_cont_male.T
df_femaleCont = df_cont_female.T

#Creates Disease subgroups
framesALS = [df_maleALS, df_femaleALS]
df_ALS = pd.concat(framesALS)
framescont = [df_maleCont, df_femaleCont]
df_cont = pd.concat(framescont)

#Returns reates p_value and log2 fold increase for each gene
#Compares ALS patient to control patient averages for each gene
p_values = {}
for i in df_ALS.iloc[:,:-2]:
    if df_ALS[i].nunique() != 1 and df_cont[i].nunique() != 1:
        lst1 = df_ALS[i]
        lst2 = df_cont[i]
        p_values[i] = ((stats.mannwhitneyu(lst1, lst2, alternative = 'two-sided')[1]),
                           math.log2(mean(lst1)/mean(lst2)))

#Compares male ALS patient to female ALS patient averages for each gene
p_values_ALS = {}
for i in df_maleALS.iloc[:,:]:
    if df_maleALS[i].nunique() != 1 and df_femaleALS[i].nunique() != 1:
        lst1 = df_maleALS[i]
        lst2 = df_femaleALS[i]
        p_values_ALS[i] = ((stats.mannwhitneyu(lst1, lst2, alternative = 'two-sided')[1]),
                           math.log2(mean(lst1)/mean(lst2)))

#Compares male control patient to female control patient averages for each gene
p_values_cont = {}
for i in df_maleCont.iloc[:,:]:
    if df_maleCont[i].nunique() != 1 and df_femaleCont[i].nunique() != 1:
        lst1 = df_maleCont[i]
        lst2 = df_femaleCont[i]
        p_values_cont[i] = ((stats.mannwhitneyu(lst1, lst2, alternative = 'two-sided')[1]),
                           math.log2(mean(lst1)/mean(lst2)))

#Returns list of genes found to be differentially expressed using alpha value 0.05
#and employing Bonferroni correction
def DEGs(genes, alpha):
    lst = []
    for i in genes:
        if genes[i][0] < alpha/len(genes):
            lst.append(i)
    return lst

deg = DEGs(p_values, 0.05)
ALS_deg = DEGs(p_values_ALS, 0.05)
Cont_deg = DEGs(p_values_cont, 0.05)
combined = ALS_deg + Cont_deg

ALS_only = []
control_only = []
both = []
for i in combined:
    if i in ALS_deg and i not in Cont_deg:
        ALS_only.append(i) #DEGs only found in ther ALS cohort
    elif i in ALS_deg and i in Cont_deg:
        both.append(i) #DEGs found in both ALS and control cohort
    else:
        control_only.append(i) #DEGs only found in the
# print ('DEGs between ALS and Control Patients:', deg)
# print ('DEGs between men and women in ALS cohort:', ALS_only)
# print ('DEGs between men and women in Control cohort:', control_only)

#Returns sorted dataframe of log2 fold change between samples in each tested cohort for all DEGs
foldchange = pd.DataFrame(index = ALS_only, columns = ['ALS/Control', 'M/F ALS', 'M/F Control'])
cats = [p_values, p_values_ALS, p_values_cont]
for j in range(len(cats)):
    for i in range(len(ALS_only)):
        foldchange.iloc[i, j] = cats[j][ALS_only[i]][1]
foldchange.sort_values('M/F ALS')

#Returns list of x values (log2 Fold Change Expression) and y values (-log10 p-val) for every gene in sample
#and significance value cut off
def split_pval(dict):
    lstX = []
    lstY = []
    lstC = []
    for i in dict:
        lstX.append(dict[i][1])
        lstY.append(-math.log10(dict[i][0]))
        #if (dict[i][0]) < 0.05/len(dict): #This code was suppose to change colors of DEGs but it doesnt work for some reason
            #lstC.append('#ff0000')
            #print (i, -math.log(dict[i][0]))
        #else:
            #lstC.append('#0000ff')
    cutoff = -math.log(0.05/len(dict))
    return (lstX, lstY, cutoff)

vals = [p_values, p_values_ALS, p_values_cont]
titles = ['ALS vs. Control Samples', 'M vs. F ALS Patients', 'M vs. F Control Patients']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

for i in range(len(axs)):
    axs[i].scatter(split_pval(vals[i])[0], split_pval(vals[i])[1], s= 2, c = '#0000ff')
    axs[i].axhline(split_pval(vals[i])[2], linestyle = '--', c = '#ff0000')
    axs[i].set_ylim(0,30)
    axs[i].set_xlim(-4,4)
    axs[i].set_xlabel('log\N{SUBSCRIPT TWO} Fold Change')
    axs[0].set_ylabel('-log10 adjusted P value')
    axs[i].set_title(titles[i])

fig.savefig('test1.pdf')
