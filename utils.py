import pandas as pd
import random

'''
Util function that can concat csv files to produce one
with mixed labels
'''
def concat_csv(file_1, file_2, save_file_path):
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)
    df = pd.concat([df1, df2], sort=True)
    df.to_csv(save_file_path)

'''
Used to generate random file/directory names
'''
def name_gen(name, ext=''):
    return name + str(random.randint(1111, 9999)) + ext