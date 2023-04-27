# import datasets
import pandas as pd
import os
import opendatasets as od

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
