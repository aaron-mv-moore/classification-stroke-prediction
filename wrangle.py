# imports
import pandas as pd
import os
import opendatasets as od
from sklearn.model_selection import train_test_split

# acquire data
# function
def get_data_csv():
    '''
    Argument: No arguments required
    Note: kaggle api username and key are required to download the data
    Actions: 
        1. Checks for the existence of the csv
            a. if present:
                i. reads the csv from the current working directory
            b. if not present:
                i. downloads csv from kaggle api
    Return: dataframe
    Modules:
        1. import pandas as pd
        2. import os
        3. import opendatasets as od
    '''
    # a variable to hold the xpected or future file name
    filename = 'stroke-prediction-dataset/healthcare-dataset-stroke-data.csv'
    
    # if the file is present in the directory 
    if os.path.isfile(filename):
      
        # read the csv and assign it to the variable df
        df = pd.read_csv(filename)
        
        # return the dataframe and exit the funtion
        return df
    
    # if the file is not in the current working directory,
    else:
        
        # url needed to read from a csv
        url = 'https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?select=healthcare-dataset-stroke-data.csv'
        
        # downloads the csv from kaggle after api key is entered
        od.download(url)
        
        # reads csv from url using pandas function
        df = pd.read_csv(filename)
        
        # returns the dataframe
        return df

# clean
def clean_data():
    '''
    Arguments: none
    Action:
        1. Gets data
        2. Changes colmn names
        3. Drops 2 columns
        4. Drops nulls
    Returns: Clean df
    Modules:
        1. import pandas as pd
        2. from wrangle import get_data_csv
    '''
    # get data
    df = get_data_csv()
    
    # changing column names
    df.columns = df.columns.str.lower().str.strip()
    
    # dropping unneeded columns
    df.drop(['smoking_status', 'id'], axis=1, inplace=True)
    
    # dropping null values
    df.dropna(inplace=True)
    
    # exit df with returned values
    return df

# prep
def prepare_data(df, base_explore=True):
    '''
    Arguments: cleaned df, base_explore retains variables in a non-encoded format, useful for visualizations and exploration
    Actions:
        1. Creates a dataframe with only dummy variables, numerical variables, and the target
        2. Formats all the column titles for python usability
        3. Splits data into train validate, and test with straitification on target
    Return: train, validate, test
    Modules: pandas as pd
    '''
    
    # creating age bins based on decade
    df['age_bins'] = pd.cut(df['age'], bins= [0, 9, 19, 29, 39, 49, 59, 69, 79, 89], 
      labels = ['under 10', 'teens', '20s', '30s', '40s', '50s', '60s', '70s', '80s']
      )
    
    # creating glucose bins based on medical reccommendation
    df['glucose_bins'] = pd.cut(df['avg_glucose_level'], bins = [0, 70, 125, 300], labels=['low', 'average', 'high'])
    
    # creating bmi bins 
    df['bmi_bins'] = pd.cut(df['bmi'], bins= 5)
    
    # assigning a target
    target = 'stroke'
    
    # default argument fo base_explore is True
    if base_explore == True:

        # skip the encoding of the variables
        pass
    
    else:
        # Create list of object type/categorical columns
        df_objects = [col for col in df if df[col].dtype == 'O' and col != target]
        
        # Create dummy variables and add them to the df
        df = pd.concat([df, pd.get_dummies(df[df_objects], drop_first=True)], axis=1)
    
        # Create a list of all non-object variables and including the target
        num_cols = [col for col in df if df[col].dtype != 'O' or col == target]

        # creating a df with only the variables needed for exploring and modeling
        df = df[num_cols]
   
    return df

# split
def split_data(df, stratify_on='stroke'):
    '''
    Arguments: prepared dataframe, optional target - must be a string literal that is a column title
    Actions: 
        1. Splits the dataframe with 80% of the data assigned to tv and 20% assigned to test
        2. Splits the tv dataset with 70% of tv assigned to train and 30% assigned to validate
    Returns: 3 variables, each containing a portion
    Modules: 
        1. from sklearn.model_selection import train_test_split
        2. pandas as pd
    Note: Order matters with variable assignment
    '''
    
    # when the target is a string that is a column title
    if stratify_on in df.columns.to_list():
        # the data is split 80/20 with the target used for stratification
        train_validate, test = train_test_split(df, train_size=.8, random_state = 1017,
                stratify = df[stratify_on])
        
         # splitting train_validate 70/30 with the target used for stratification
        train, validate = train_test_split(train_validate, train_size=.7, stratify=train_validate[stratify_on])
    # for all other targets
    else:
        # inform user that there is no stratification
        print('No stratification applied during the split')
        
        # split that data 80/20
        train_validate, test = train_test_split(df, train_size=.8, random_state = 1017)
        
        # splitting train_validate 70/30
        train, validate = train_test_split(train_validate, train_size=.7)
    
    return train, validate, test

# wrangle
def wrangle_data():
    '''
    Arguments: none
    Actions: uses all other modules created to wrangle data in one function
    Returns: train, validate, test
    '''
    # split the data and assign it to variables
    train, validate, test = split_data(
        
        # prep the data
        prepare_data(
            
            # get clean data
            clean_data()))
    
    # exit function and return train, validate, and test ds
    return train, validate, test
