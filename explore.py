# imports
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as  stats

import warnings
warnings.simplefilter('ignore')


def get_stroke_visual(train):
    '''
    Get a count plot for stroke in the train dataset
    '''
    
    sns.set()
    sns.countplot(train['stroke'].replace([1, 0], ['Stroke', 'No Stroke']), palette=['b', 'r'], alpha=.5, ec=['b', 'r'], linewidth=1)
    plt.ylabel('Count')
    plt.xlabel('Stroke')
    plt.title('Imbalanced Stroke Values')
    plt.show()
    
    
    return

def get_age_visual(train):
    '''
    Actions: gets plot with the age density of those who had a stroke and those who did not have a stroke
    Modules:
        1. import seaborn as sns
        2. import matplotlib.pyplot as plt
    '''
    
    # getting two dataframes, one with only people who had a stroke a 
    no_stroke = train[train['stroke'] != 1]
    
    stroke = train[train['stroke'] == 1]
    

    # plotting both distibutions on the same figure
    fig = sns.kdeplot(stroke['age'], shade=True, color = "r", label = 'Stroke')
    fig = sns.kdeplot(no_stroke['age'], shade=True, color="b", label = 'No Stroke')
    plt.xlabel('Age')
    plt.title('Stroke Risk Higher with Age')
    plt.legend(loc='upper left')
    plt.show()
    
    return


def get_blood_sugar_visual(train):
    '''
    Get a density graph with stroke risk of people based on blood sugar levels
    '''
    
    # getting two dataframes, one with only people who had a stroke a 
    no_stroke = train[train['stroke'] != 1]

    stroke = train[train['stroke'] == 1]
    
    # setting he style

    # plotting both distibutions on the same figure
    fig = sns.kdeplot(no_stroke['avg_glucose_level'], shade=True, color="b", label = 'No Stroke')
    fig = sns.kdeplot(stroke['avg_glucose_level'], shade=True, color = "r", label = 'Stroke')
    plt.xlabel('Average Blood Sugar Levels')
    plt.title('Two Blood Sugar Peaks: Average & High')
    plt.legend(loc='upper left')
    plt.xticks()
    plt.show()
    
    return

def get_gender_visual(train):
    '''
    Gets barchart with proportion of each gender that have a stroke or dont have a stroke
    '''
    (train.groupby('gender')['stroke'].value_counts(normalize=True)
       .unstack('stroke').plot.bar(stacked=True, color=['b', 'r'], alpha=.5))
    plt.ylabel('Proportion')
    plt.xlabel('Gender')
    plt.legend(['No Stroke', 'Stroke'], loc = 'lower right')
    plt.xticks(rotation=0)
    plt.title('No Association between Gender & Stroke')
    plt.show()
    
    return


def eval_results(train, col):
    '''
    this function will take in the p-value, alpha, and a name for the 2 variables 
    you are comparing (group 1 and group 2)
    '''
    # set alpha
    alpha = 0.05
    
    # set target
    target = 'stroke'
    
    # get observed
    observed = pd.crosstab(train[col], train[target])
    
    # get stats
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    col = col.replace('_', ' ')
    
    # print results
    if p < alpha:
        print(f'There exists some relationship between {target} and {col}. (p-value: {p})')
    else:
        print(f'There is not a significant relationship between {target} and {col}. (p-value: {p})')
        
    return

