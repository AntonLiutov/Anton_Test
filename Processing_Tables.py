import numpy as np
import pandas as pd

def getColumnsLists(data):
    '''
    Purpose:
        extracting a column with ids and others
    Input:
        data - pd.DataFrame()
    Output:
        id_job_column - a list of id_job column name
        other_columns - a list of other columns names
    '''
    # extracting a column with ids
    id_job_column = list(data.columns[(data.columns.isin(['id_job']))])
    
    # extracting name of the rest of columns
    other_columns = list(data.columns[~(data.columns.isin(['id_job']))])
    
    return id_job_column, other_columns


def Standard_Scaler(data):
    '''
    Purpose:
        Z-scoring a table
    Input:
        data - pd.DataFrame(), a table of features of integer types
    Output:
        standardized data   
    '''
    
    return ((data.iloc[:,1:] - data.iloc[:,1:].mean()) / data.iloc[:,1:].std()).values

def mean_absolute_error(numpyarray):
    '''
    Purpose:
        calculating mean absolute error
    Input:
        numpyarray - np.array of a feature
    Output:
        mae - mean absolute error of numpyarray
    '''
    
    mae = np.mean(np.abs(numpyarray - np.mean(numpyarray)))
    
    return mae


def createTable(data, columns, standardization_func):
    
    '''
    Purpose:
        converting all features except for id in a standardized 
        representation and indexes of max values and abs mean diff
    Input:
        data - pd.DataFrame(), data with ids and features
        columns - a list of features except for ids
        standardization_func - a function to convert raw features into 
            other representation
    Output:
        data_all - pd.DataFrame(), data with converted features into a
            specific representation and some statistics as well for a 
            specific code name
    '''
    
    # initializing an empty table
    data_all = pd.DataFrame()
    
    # looping over all code name features
    for column in columns:

        # splitting the feature into 256 features and a code name
        data = data[column].str.split(",", expand=True).astype(int)

        # extracting a code name
        code_number = str(data.iloc[0,0])

        # creating columns
        columns = list(map(lambda x: 'features_' + code_number + '_stand_' + str(x), data.columns[1:]))

        # standardizing features
        # a function standardization_func should return a matrix, numpy array
        data_st = pd.DataFrame(data=standardization_func(data),
                               columns=columns)

        # extracting indexes of a max value of each feature
        data_max = data.iloc[:,1:].idxmax(axis=1).rename('max_feature_' + code_number + '_index').reset_index(drop=True)

        # taking an mean absolute error
        data_max_abs_mean_diff = pd.DataFrame(data = [
            mean_absolute_error(data.iloc[:,i].values) for i in data_max],
                                              columns = ['max_feature_' + code_number + 'abs_mean_diff'])

        # concatenating all stats into one table and concatenating to the previous state of the table
        data_all = pd.concat([data_all, data_st, data_max, data_max_abs_mean_diff], axis=1)
        
    return data_all


def load_data(file_name):
    '''
    Purpose:
        loading data
    Input:
        file_name - string, a file name
    '''
        
    return pd.read_csv(file_name, sep='\t')


def main(file_name):
    '''
    Purpose:
        converting a file to a specific file with statistics
    Input:
        file_name - string, a file name
    Output:
        saved a preprocessed data
    '''
    
    # loading data
    data = load_data(file_name)
    
    # extracting columns names
    id_job_column, other_columns = getColumnsLists(data)
    
    # calculating statistics
    data_all = createTable(data, other_columns, Standard_Scaler)
    
    # concatenating ids and stats
    data_proc = pd.concat([data[id_job_column], data_all], axis=1)
    
    # recording data to a file
    data_proc.to_csv(file_name.replace('.tsv', '') + '_proc.tsv', sep='\t', index=False)





