# Import necessary packages
import pandas as pd
import os
import math
import random
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
import scipy.stats as sciStats

dir_path = os.path.dirname(os.path.realpath(__file__)) # If <-- doesn't work, use: os.path.abspath('')
os.chdir(os.path.dirname(dir_path))

# database = 'GSE122649_RAW' # Specifies the local directory with the sample data from the GSE122649 database
# starterFile = 'GSM3477217_ALS-21_counts' # Specifies an arbitrarily chosen file from the GSE122649 database to add the genes DataFrame column
database = 'GSE124439_RAW' # Specifies the local directory with the sample data from the GSE124439 database
starterFile = 'GSM3533230_CGND-HRA-00013_counts' # Specifies an arbitrarily chosen file from the GSE124439 database to add the genes DF column

path = ('data' + '/' + database + '/' + starterFile + '.txt') # Specifies the local path to the starterFile
data = pd.read_csv(path, sep = "\s+", header = 0, names = ['genes','counts']) # Reads in the starterFile
df = pd.DataFrame(data['genes']) # Adds the genes column to start a Pandas DataFrame (same for all files)
dfLog = pd.DataFrame(data['genes']) # For creating a log10-scaled version of the above (more easily visualized)

for filename in os.listdir('data' + '/' + database): # Iterates through each file in the database
    path = ('data' + '/' + database + '/' + filename)
    data = pd.read_csv(path, sep = "\s+", header = 0, names = ['genes','counts'])

    regData = []
    logData = []
    for i in data['counts']:
        regData.append(i)
        if(i == 0):
            logData.append(0)
        else:
            logData.append(math.log10(i))

    truncName = filename[:-4]
    df[truncName] = pd.DataFrame(regData) # Adds the read counts for the given sample to the DataFrame
    dfLog[truncName] = pd.DataFrame(logData)

geneIndexedDF = df.set_index('genes') # Makes the genes column the index for the DataFrame
geneIndexedLogDF = dfLog.set_index('genes')

sortedDF = geneIndexedDF.sort_values(by=[starterFile],ascending=False) # Sorts reads descending for the starterFile
sortedLogDF = geneIndexedLogDF.sort_values(by=[starterFile],ascending=False)

print(sortedDF)

# Plots a heatmap of the data
plt.subplots(figsize=(20,20))
ax = sns.heatmap(sortedLogDF,cmap="YlGnBu")
plt.tight_layout()
plt.savefig("figures/heatmap_db39_log.png")
plt.show()


# SHIFT FROM PCA / PLS-DA TO MANN-WHITNEY & DEG ANALYSIS
df_scale = sortedDF.T #creates copy of the transposed dataframe

#Scales all counts to housekeeper gene Glucose-6-phosphate isomerase
#Eisenberg and Levanon, 2013
for i in df_scale:
    df_scale[i] = df_scale[i]/df_scale['SNRPD3']

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

#Assigns gender adn Disease Group to each sample
df_scale['gender'] = matrixData['Gender'].factorize()[0] # 0 is female, 1 is male
df_scale['group'] = matrixData['Subject Group'].factorize()[0] # 0,2 is ALS, 1 is control, 3 is other nuerological control

#Initializes new dataframes for each demographic group
df_T = df_scale
df_ALS_male = pd.DataFrame(index=df_T.columns)
df_ALS_female = pd.DataFrame(index=df_T.columns)
df_cont_male = pd.DataFrame(index=df_T.columns)
df_cont_female = pd.DataFrame(index=df_T.columns)

#Populates demographic based dataframes
df = df_T.T
for i in df:
    if df.loc['gender',i] == 1 and (df.loc['group',i] == 1):
        df_cont_male[i] = df[i]
    elif df.loc['gender',i] == 1 and (df.loc['group',i] == 0 or df.loc['group',i] == 2):
        df_ALS_male[i] = df[i]
    elif df.loc['gender',i] == 0 and (df.loc['group',i] == 1):
        df_cont_female[i] = df[i]
    elif df.loc['gender',i] == 0 and (df.loc['group',i] == 0 or df.loc['group',i] == 2):
        df_ALS_female[i] = df[i]

#Transposes demographic dataframes for iteration
df_maleALS = df_ALS_male.T
df_femaleALS = df_ALS_female.T
df_maleCont = df_cont_male.T
df_femaleCont = df_cont_female.T

#Creates Disease subgroups
frames_ALS = [df_maleALS, df_femaleALS]
df_ALS = pd.concat(frames_ALS)
frames_cont = [df_maleCont, df_femaleCont]
df_cont = pd.concat(frames_cont)

#Returns reates p_value and log2 fold increase for each gene
#Compares ALS patient to control patient averages for each gene
p_values = []
for i in df_ALS.iloc[:,:-2]:
    if df_ALS[i].nunique() != 1 and df_cont[i].nunique() != 1:
        lst1 = df_ALS[i]
        lst2 = df_cont[i]
        p_values.append((i,(stats.ttest_ind(lst1, lst2)[1]), math.log2(mean(lst1)/mean(lst2))))

#Compares male ALS patient to male control patient averages for each gene
p_values_male = []
for i in df_maleALS.iloc[:,:-2]:
    if df_maleALS[i].nunique() != 1 and df_maleCont[i].nunique() != 1:
        lst1 = df_maleALS[i]
        lst2 = df_maleCont[i]
        p_values_male.append((i,(stats.ttest_ind(lst1, lst2)[1]), math.log2(mean(lst1)/mean(lst2))))
    #else:
    #    print (df_maleALS[i], df_maleCont[i])

#Compares female ALS patient to female control patient averages for each gene
p_values_female = []
for i in df_femaleALS.iloc[:,:-2]:
    if df_femaleALS[i].nunique() != 1 and df_femaleCont[i].nunique() != 1:
        lst1 = df_femaleALS[i]
        lst2 = df_femaleCont[i]
        p_values_female.append((i,(stats.ttest_ind(lst1, lst2)[1]), math.log2(mean(lst1)/mean(lst2))))

#sorts DEG analyses by p_value
p_sort = sorted(p_values, key=lambda pval: pval[1])
pmale_sort = sorted(p_values_male, key=lambda pval: pval[1])
pfemale_sort = sorted(p_values_female, key=lambda pval: pval[1])

#Determine differentially expressed genes using Bonferroni Correction (Not used in report)
#takes lst of genes with associated p value and returns genes under threshold determined by bonferrroni correction
#alpha = 0.1
def DEG(lst):
    degs = []
    for i in lst:
        if i[1] < 0.10/len(lst):
            degs.append(i[0])
    return degs

print ('ALS:', DEG(p_sort), len(DEG(p_sort)))
print ('Male:', DEG(pmale_sort), len(DEG(pmale_sort)))
print ('Female:', DEG(pfemale_sort), len(DEG(pfemale_sort)))


#may throw error. To resolve try (import statsmodels as sm)
import statsmodels.api as sm

#Takes list of genes (gene, p_val, 2foldchange) and returns adjusted p_value in place of old p_value
#Used Bonferroni_Hachberg correction
def DEF(genes, alpha):
    pvals = []
    for i in genes:
        pvals.append(i[1])
    results = sm.stats.multitest.multipletests(pvals, alpha=alpha,
                                               method='fdr_bh', is_sorted=False, returnsorted=False)
    adj_pval = []
    for i in range(len(genes)):
        adj_pval.append((genes[i][0], results[1][i], genes[i][2], results[0][i]))
    return adj_pval

#Takes list of genes with adjusted p_value and resuturns list of genes found to be differentially expressed
def diff_exp(lst):
    degs = []
    for i in lst:
        if i[3] == True:
            degs.append(i[0])
    return degs

print('DEGs between ALS and control patients:', diff_exp(DEF(p_sort, .1)), len(diff_exp(DEF(p_sort, .1))))
print('DEGs between ALS and control patients in male cohort:', diff_exp(DEF(pmale_sort, .1)), len(diff_exp(DEF(pmale_sort, .1))))
print('DEGs between ALS and control patients in female cohort:', diff_exp(DEF(pfemale_sort, .1)), len(diff_exp(DEF(pfemale_sort, .1))))



#Creates Values for Venn Diagram
ALS = diff_exp(DEF(p_sort, .1))
male = diff_exp(DEF(pmale_sort, .1))
female = diff_exp(DEF(pfemale_sort, .1))

combined = male + female + ALS
combined = list(dict.fromkeys(combined))

group1 = 0 #male, female, ALS
group2 = 0 #male only
group3 = 0 #female only
group4 = 0 #ALS only
group5 = 0 #male, female
group6 = 0 #male, ALS
group7 = 0 #female ALS


for i in combined:
    if i in male and i in female and i in ALS:
        group1 += 1
        print (i) #prints list of genes at intersection of DEG analysis
    elif i in male and i not in female and i not in ALS:
        group2 += 1
    elif i in female and i not in male and i not in ALS:
        group3 += 1
    elif i in ALS and i not in male and i not in female:
        group4 += 1
    elif i in male and i in female and i not in ALS:
        group5 += 1
    elif i in male and i in ALS and i not in female:
        group6 += 1
    elif i in female and i in ALS and i not in male:
        group7 += 1

print (group1, group2, group3, group4, group5, group6, group7)

#Genes of interest identified in (Lederer, 2007; Mougeot, 2011; Scoles, 2020)
gois = ["SOD1", "TARDBP", "FUS", "C9orf72", "ATXN2", "TAF15", "UBQLN2", "OPTN", "KIF5A", "hnRNPA1",
        "hnRNPA2 B1","MATR3", "CHCHD10", "EWSR1", "TIA1", "SETX", "ANG", "CCNF", "NEK1", "TBK1", "VCP",
        "SQSTM1", "PFN1", "TUBB4A", "CHMP2B", "SPG11", "ALS2", "TDP43", "SOD1", "Ctss", "Mmp12", "Gpnmb",
        "Trem2", "Cd300c2","Mpeg1", "Hvcn1", "C4b", "Gm23969", "Tyrobp", "Steap1", "Lgals3", "Atp6v0d2", "C3",
        "Gm6166", "Clic6", "Rgs1", "Ifi202b", "Cd68 ", "Clec7a", "Sspo", "Il1rrn", "Adgrg5",]

#Creates list of GOI in DEG analysis groups for plotting
gois_ALS = []
for i in DEF(p_sort, .1):
    if i[0] in gois:
        gois_ALS.append(i)

gois_male = []
for i in DEF(pmale_sort, .1):
    if i[0] in gois:
        gois_male.append(i)

gois_female = []
for i in DEF(pfemale_sort, .1):
    if i[0] in gois:
        gois_female.append(i)


#Plots DEGs from combined group onto plot using log2 Fold Change Expression in Male and Female Cohorts
ALS_pval = DEF(p_sort, .05)
male_pval = DEF(pmale_sort, .1)
female_pval = DEF(pfemale_sort, .1)

dict_male = {}
dict_female = {}
for i in male_pval:
    if i[0] in ALS:
        dict_male[i[0]] = i[2]

for i in female_pval:
    if i[0] in ALS:
        dict_female[i[0]] = i[2]

goi_male = {}
goi_female = {}
for i in DEF(pmale_sort, .1):
    if i[0] in gois:
        goi_male[i[0]] = i[2]

for i in DEF(pfemale_sort, .1):
    if i[0] in gois:
        goi_female[i[0]] = i[2]

lstx = []
lsty = []
lstC = []
for i in ALS:
    lstx.append(dict_male[i])
    lsty.append(dict_female[i])

plt.figure(figsize=(5,5))
plt.plot([-10, 10], [-10, 10], '--', alpha = 0.5, c = 'black')
plt.scatter(lstx, lsty, s = 2)
plt.ylim([-3, 2])
plt.xlim([-3, 2])
plt.xlabel('Male Cohort log\N{SUBSCRIPT TWO} Fold Change')
plt.ylabel('Female Cohort log\N{SUBSCRIPT TWO} Fold Change')
plt.title('Fold Change Expression ALS/Control')

plt.savefig('figures/DEG_foldchange.pdf')


#Returns list of x values (log2 Fold Change Expression) and y values (-log10 p-val) for every gene in sample
#and significance value cut off
def split_pval(lst):
    lstX = []
    lstY = []
    lstC = []
    for i in range(len(lst)):
        lstX.append(lst[i][2])
        lstY.append(-math.log10(lst[i][1]))
        if (lst[i][3]) == True : #This code was suppose to change colors of DEGs but it doesnt work for some reason
            lstC.append('#ff0000')
        else:
            lstC.append('#0000ff')
    return (lstX, lstY, lstC)

#Volcano Plot for DEGs for 3 analyses, requires lst of genes with associated adjusted p value and fodl change expression
vals = [DEF(p_sort, .1), DEF(pmale_sort, .1), DEF(pfemale_sort, .1), gois_ALS, gois_male, gois_female]
titles = ['ALS vs. Control Patients', 'ALS vs Control Male Cohort', 'ALS vs Control Female Cohort']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

for i in range(len(axs)):
    axs[i].scatter(split_pval(vals[i])[0], split_pval(vals[i])[1], s = 2, c = split_pval(vals[i])[2])
    axs[i].plot([-10, 10],[-math.log10(0.101), -math.log10(0.101)], '--', c = 'black')
    axs[i].set_ylim(0,5)
    axs[i].set_xlim(-5,5)
    axs[i].set_xlabel('log\N{SUBSCRIPT TWO} Fold Change')
    axs[0].set_ylabel('-log10 FDR Adjusted P value')
    axs[i].set_title(titles[i])

fig.savefig('figures/VolcanoPlots.pdf')

#Plots all genes on volcano plot, but only highlights 26 GOIs

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
for i in range(len(axs)):
    axs[i].scatter(split_pval(vals[i])[0], split_pval(vals[i])[1], s = 2, alpha = 0.5, c = 'lightblue')
    axs[i].scatter(split_pval(vals[i+3])[0], split_pval(vals[i+3])[1], s = 10, c = 'orange')
    axs[i].plot([-10, 10],[-math.log10(0.1), -math.log10(0.1)], '--', c = 'black')
    axs[i].set_ylim(0,5)
    axs[i].set_xlim(-5,5)
    axs[i].set_xlabel('log\N{SUBSCRIPT TWO} Fold Change')
    axs[0].set_ylabel('-log10 FDR Adjusted P value')
    axs[i].set_title(titles[i])

fig.savefig('figures/VolcanoPlots_grayed.pdf')


### TRANSITION TO PCA & PLS-DA ANALYSES

df_T = df.T
df_T.columns = df_T.iloc[0]
df_T = df_T.drop('genes')

sc = StandardScaler()
temp = sc.fit_transform(df_T)
df_T = pd.DataFrame.from_records(temp, index = df_T.index, columns = df_T.columns)

print(df_T)


# Load the data matrix describing what each sample is
matrixPath = ('data/patient_info_matrix.xlsx') # Specifies the local path to the matrix of information
matrixData = pd.read_excel(matrixPath, header=0) # Reads in the starterFile
print(matrixData.head())

# print(list(dict.fromkeys(matrixData['Sex'])))
sexInfo = list(matrixData['Gender'])
# print(sexInfo)

# print(list(dict.fromkeys(matrixData['Subject Group'])))
subjectGroup = list(matrixData['Subject Group'])

# print(list(dict.fromkeys(matrixData['ALS NMF subtype**'])))
molSubtype = list(matrixData['ALS NMF subtype**'])


sexINTfo = []
for i in sexInfo:
    if (i == 'Female'):
        sexINTfo.append(0)
    else:
        sexINTfo.append(1)
print(sexINTfo)

subGroupInt = []
for i in subjectGroup:
    if (i == 'Non-Neurological Control'):
        subGroupInt.append(0)
    elif (i == 'Other Neurological Disorders'):
        subGroupInt.append(1)
    else:
        subGroupInt.append(2)
print(subGroupInt)

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
print(molSubtypeInt)


df_T_exp = df_T
df_T_exp['Sex'] = sexINTfo
df_T_exp['SubjectGroup'] = subGroupInt
df_T_exp['MolSubtype'] = molSubtypeInt

# Dividing the dataset by sex
femaleData = df_T_exp.loc[df_T_exp['Sex'] == 0]
femaleData_SG = list(femaleData['SubjectGroup'])
femaleData_MS = list(femaleData['MolSubtype'])
femaleData = femaleData.drop(['Sex','SubjectGroup','MolSubtype'], axis=1)

maleData = df_T_exp.loc[df_T_exp['Sex'] == 1]
maleData_SG = list(maleData['SubjectGroup'])
maleData_MS = list(maleData['MolSubtype'])
maleData = maleData.drop(['Sex','SubjectGroup','MolSubtype'], axis=1)

# Dividing the dataset by the subject group (control, ALS, other neurological disorder)
controlData = df_T_exp.loc[df_T_exp['SubjectGroup'] == 0]
controlData_MF = list(controlData['Sex'])
controlData_MS = list(controlData['MolSubtype'])
controlData = controlData.drop(['Sex','SubjectGroup','MolSubtype'], axis=1)

otherNDData = df_T_exp.loc[df_T_exp['SubjectGroup'] == 1]
otherNDData_MF = list(otherNDData['Sex'])
otherNDData_MS = list(otherNDData['MolSubtype'])
otherNDData = otherNDData.drop(['Sex','SubjectGroup','MolSubtype'], axis=1)

alsData = df_T_exp.loc[df_T_exp['SubjectGroup'] == 2]
alsData_MF = list(alsData['Sex'])
alsData_MS = list(alsData['MolSubtype'])
alsData = alsData.drop(['Sex','SubjectGroup','MolSubtype'], axis=1)

# Dividing the dataset by the molecular ALS subtypes
alsOXData = df_T_exp.loc[df_T_exp['MolSubtype'] == 0]
alsOXData_MF = list(alsOXData['Sex'])
alsOXData_SG = list(alsOXData['SubjectGroup'])
alsOXData = alsOXData.drop(['Sex','SubjectGroup','MolSubtype'], axis=1)

alsTEData = df_T_exp.loc[df_T_exp['MolSubtype'] == 1]
alsTEData_MF = list(alsTEData['Sex'])
alsTEData_SG = list(alsTEData['SubjectGroup'])
alsTEData = alsTEData.drop(['Sex','SubjectGroup','MolSubtype'], axis=1)

alsGliaData = df_T_exp.loc[df_T_exp['MolSubtype'] == 2]
alsGliaData_MF = list(alsGliaData['Sex'])
alsGliaData_SG = list(alsGliaData['SubjectGroup'])
alsGliaData = alsGliaData.drop(['Sex','SubjectGroup','MolSubtype'], axis=1)


# Perform a PCA on the data
PCAs = [df_T, controlData, alsData]
labels = ['All','Control','ALS']
legends = [sexINTfo,controlData_MF,alsData_MF]

for i in range(len(PCAs)):
    X = PCAs[i]
    label = labels[i]
    legend = legends[i]

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
    print(explained_variance)

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
    plt.figure = figure(num=None, figsize=(5.2,3.8), dpi=100)
    plt.plot(num_com,cum_exvar,'--o')
    plt.xlabel('Number of Included PCA Components')
    plt.ylabel('Total Variance Explained')
    plt.xticks(num_com,num_com)
    plt.yticks(np.linspace(0.4,1.0,7))
    plt.tight_layout()
    plt.savefig('figures/pcaScree_' + label + '_varexplained.png', bbox='tight')

    # Plot the first two PCs
    fPC1 = []
    fPC2 = []
    mPC1 = []
    mPC2 = []
    i = 0
    while(i < len(legend)):
        if(legend[i] == 0):
            fPC1.append(pcaScores[i,0])
            fPC2.append(pcaScores[i,1])
        else:
            mPC1.append(pcaScores[i,0])
            mPC2.append(pcaScores[i,1])
        i = i + 1

    plt.figure = figure(num=None, figsize=(3.9,2.9), dpi=100)
    plt.scatter(fPC1, fPC2, c='magenta', s=8, label='F')
    plt.scatter(mPC1, mPC2, c='blue', s=8, label='M')
    plt.xlabel(label + ' Data PC1 (' + str(int(explained_variance[0]*100000)/1000) + '%)')
    plt.ylabel(label + ' Data PC2 (' + str(int(explained_variance[1]*100000)/1000) + '%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/pc1_pc2_' + label + '.png', bbox='tight')
    plt.show()



# Whole-data PLS-DA by sex
numComp = 9
plsr = PLSRegression(n_components=numComp, scale=False)
plsr.fit(df_T, sexINTfo)

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
scores['Sex'] = sexINTfo
groups = scores.groupby('Sex')

figure(num=None, figsize=(6.6,5.25), dpi=100)
for name, group in groups:
    plt.scatter(group['LV 1'], group['LV 2'], s=50, alpha=0.7, c=colormap[name], label=gendLabels[name])
plt.xlabel('Scores on LV 1, whole dataset separated by sex')
plt.ylabel('Scores on LV 2, whole dataset separated by sex')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('figures/plsda_combined.png', bbox='tight')
plt.show()

#plt.clf()
scores.drop('Sex', axis=1)


# 10-fold Cross Validation Function, courtesy of Erin Tevonian

#Input:values of X and Y used during PLSDA
#Output: accuracy score array for each of the 10 validations (score = fraction of correctly classified samples)

def cross_val_plsda(X, y):
    kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

    accuracy = []

    for train_index, test_index in kf_10.split(X,y):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        plsr.fit(X_train, y_train)

        scores = pd.DataFrame(data = plsr.x_scores_)

        #Need to compare integers of Y pred and Y test to use accuracy score
        y_pred = plsr.predict(X_test)
        y_pred = y_pred.round()
        y_pred = y_pred.astype(int)

        test_y = y_test.astype(int)

        accuracy.append(accuracy_score(test_y, y_pred))


    return accuracy



# Specifically comparative PLS-DAs
datasets = [femaleData, maleData, controlData, otherNDData, alsData, alsOXData, alsTEData, alsGliaData]
datasetLabels = ['Female Subjects', 'Male Subjects', 'Control Subjects', 'Other Neurological Subjects',
                'ALS Subjects', 'ALS OX Subjects', 'ALS TE Subjects', 'ALS Glial Subjects']
separators = [[femaleData_SG, femaleData_MS], [maleData_SG, maleData_MS],
             [controlData_MF], [otherNDData_MF], [alsData_MF, alsData_MS],
             [alsOXData_MF], [alsTEData_MF], [alsGliaData_MF]]
separatorLabels = [['Subject Group', 'Molecular Subtype'], ['Subject Group', 'Molecular Subtype'],
                  ['Sex'], ['Sex'], ['Sex', 'Molecular Subtype'],
                  ['Sex'], ['Sex'], ['Sex']]
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

        if(separatorName == 'Sex'):
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
        figure(num=None, figsize=(4.95,3.94), dpi=100)
        for name, group in groups:
            plt.scatter(group['LV 1'], group['LV 2'], s=30, alpha=0.7, c=colormap[name], label=legendLabel[name])
        plt.xlabel('LV 1 Scores, ' + datasetName + ' sep by ' + separatorName)
        plt.ylabel('LV 2 Scores, ' + datasetName + ' sep by ' + separatorName)
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

        # 10-fold CV of PLS-DA
        X = dataset.values
        y = separator
        Y_rand = pd.DataFrame(random.sample(separator, len(separator)))

        accuracy = cross_val_plsda(X,y)
        accuracy_rand = cross_val_plsda(X, Y_rand.values)

        cross_vals = pd.DataFrame([accuracy, accuracy_rand]).T
        cross_vals.columns = ['PLSDA','Random']

        cross_vals = cross_vals.melt(var_name='Model', value_name='Accuracy Score')

        figure(num=None, figsize=(4.4,3.5), dpi=100)
        ax = sns.violinplot(x = 'Model', y = 'Accuracy Score', data=cross_vals, palette = "binary")
        ax.set_title(datasetName + ' sep by ' + separatorName)
        plt.savefig('figures/plsdaVP_' + datasetName.replace(" ", "") + '_' +
            separatorName.replace(" ", "") + '.png', bbox='tight')
        plt.show()

        #T-test
        print('T-test for ' + datasetName + ' sep by ' + separatorName + ' yields: ' +
              str(sciStats.ttest_ind(accuracy,accuracy_rand, equal_var = False)))

        currSep = currSep + 1
    currData = currData + 1

expVar.index = plsColumnNames
topGenes.to_excel("figures/plsda_topgenes.xlsx")
# print(expVar)
# print(topGenes)
