# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns




def bivariate_visulization(df, target):
    cat_cols, num_cols = [], []
    for col in df.columns:
        if df[col].dtype == "o":
            cat_cols.append(col)
        else:
            if df[col].nunique() < 10:
                cat_cols.append(col)
            else:
                num_cols.append(col)
    print(f'Numeric Columns: {num_cols}')
    print(f'Categorical Columns: {cat_cols}')
    explore_cols = cat_cols + num_cols
    for col in explore_cols:
        if col in cat_cols:
            if col != target:
                print(f'Bivariate assessment of feature {col}:')
                sns.countplot(data = df, x = df[col], hue = df[target])
                plt.show()
        if col in num_cols:
            if col != target:
                print(f'Bivariate feature analysis of feature {col}: ')
                sns.boxplot(x = df[target], y = df[col], palette='crest')
                plt.show()
    print('_____________________________________________________')
    print('_____________________________________________________')
    print()


def univariate_visulization(df):
    cat_cols, num_cols = [], []
    for col in df.columns:
        if df[col].dtype == "o":
            cat_cols.append(col)
        else:
            if df[col].nunique() < 10:
                cat_cols.append(col)
            else:
                num_cols.append(col)
    explore_cols = cat_cols + num_cols
    print(f'cat_cols: {cat_cols}')
    print(f'num_cols: {num_cols}')
    for col in explore_cols:
        if col in cat_cols:
            print(f'Univariate assessment of feature {col}:')
            sns.countplot(data=df, x=col, color='turquoise', edgecolor='black')
            plt.show()
        if col in num_cols:
            print(f'Univariate feature analysis of feature {col}: ')
            plt.hist(df[col], color='turquoise', edgecolor='black')
            plt.show()
            df[col].describe()
    print('_____________________________________________________')
    print('_____________________________________________________')
    print()


def viz_explore(train, target):
    univariate_visulization(train)
    bivariate_visulization(train, target)
    plt.figure(figsize=(20,15))
    plt.rc('font', size=14)
    plt.show()


def boxplot(train):
    target = 'quality'
    explore_cols = train.columns.to_list()
    for col in explore_cols:
        if col != target:
            print(f'Bivariate assessment of feature {col}:')
            sns.boxplot(data = train, x = train[target], y = train[col], palette='crest')
            plt.show()
    print('_____________________________________________________')
    print('_____________________________________________________')
    print()
