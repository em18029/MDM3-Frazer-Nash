import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def imputation(df):

    df = df.replace([9999.99],np.nan)
    print(df)
    imp = SimpleImputer(strategy='mean')
    df_mat = imp.fit_transform(df.values)
    df = pd.DataFrame(df_mat, index=df.index, columns=df.columns)
    
    return df

if __name__ == '__main__':
    
    #########   df = import csv here   ########
    df_new = imputation(df)