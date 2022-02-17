"""MCP python script
   For this script to work, the data in the excel sheets must be slightly cleaned:
       all values of 9999.99 must be converted to nan"""

import copy
import math
from scipy.stats import linregress
import pandas as pd
import numpy as np
from cmath import rect, phase
from math import radians, degrees
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error




def make_datetime_index(df):
    """Smooth together the date and time entries to return a neat 'datetime' column"""
    df['datetime'] = df['Day'] + '/' + df['Month'] + '/' + df['Year'] + ' ' + df['Hour'] + ':' + df['Minute'] + ':00'
    df.index = pd.to_datetime(df['datetime'], errors='coerce', yearfirst=True)
    return df[['Wind_Speed', 'Wind_Direction']]



def clean_data(df):
    """remove any rows with anomalous results, i.e. nan - when the mast is broken"""
    df.dropna(subset = ['Wind_Speed'], inplace=True)
    df.dropna(subset = ['Wind_Direction'], inplace=True)
    return df





def hourly_measured_windspeed(df):
    """Gets the hourly average of the measured data.
       Do not use for wind direction"""
    df = df.drop('Wind_Direction', 1)
    df['datetime'] = df.index
    df = df[['datetime', 'Wind_Speed']]
    df = df.rename(columns={"Wind_Speed" : "Wind_Speed_meas"})
    dfh = df.groupby(pd.Grouper(key='datetime', axis=0, 
                      freq='h', sort=True)).mean('Wind_Speed')
    return dfh




def combine_the_short_term_and_long_term_speed_measurements(long_term_df, short_term_df):
    # Match the short and long periods, then do the correlation. Uses the direction from the short-term
    # as the true direction.
    # Abbreviate
    ltdf = long_term_df
    stdf = short_term_df
    # Get inputs to combined dataframe
    ltdf = ltdf['Wind_Speed'].dropna()
    # Create a combined dataframe
    comb_df = pd.DataFrame()
    comb_df['lt_speed']=ltdf.loc[ltdf.index.intersection(stdf.index)]
    # Get the short-term speed
    st_speed = stdf['Wind_Speed'].groupby(pd.Grouper(freq='H')).mean()
    comb_df['st_speed'] = st_speed
    # Add the direction from the short-term measurements
    direction = stdf['Wind_Direction']
    comb_df['direction']=direction.loc[direction.index.intersection(stdf.index)]
    comb_df = comb_df.dropna()
    return comb_df




def get_sector_dict(sector_size=30, debug=False, dirstring='m_dir'):
    """Returns dict with queries defining sectors based on compass point wind directions"""
    sector_dict = {}
    sectors_start = 0
    num_sectors = 360 / sector_size
    if 360 % num_sectors != 0:
        raise ValueError('360 should divide exactly into sector size')
    sector_centres = [x*sector_size for x in range(int(num_sectors))]
    for i_centre, centre in enumerate(sector_centres):
        #sector_name = wa.WIND_DIRECTIONS[i_centre]
        sector_right = centre + sector_size / 2
        sector_left = centre - sector_size/2
        if sector_left < 0:
            sector_left = 360 + sector_left
        if sector_right > 360:
            sector_right = 360 - sector_right
        if sector_right > sector_left:
            sector_string = '{dirstring} > {} and {dirstring} <= {}'.format(sector_left, sector_right, dirstring=dirstring)
        elif sector_left > sector_right:
            sector_string = '{dirstring} > {} or {dirstring} <= {}'.format(sector_left, sector_right, dirstring=dirstring)
        else:
            sector_string = 'Invalid sector'
        if debug:
            print(sector_string)
        sector_dict[str(sector_centres[i_centre])] = sector_string
    return(sector_dict)




def get_dict_with_dfs_for_each_sector(df, **kwargs):
    """ Split the data into sectors, for a selected vane"""
    sector_dict = get_sector_dict(**kwargs)
    sectored_dfs = dict()
    for sector, query in sector_dict.items():
        sectored_dfs[sector] = copy.deepcopy(df.query(query))
    # from pprint import pprint
    # pprint(sector_dict)
    return sectored_dfs




def train_a_linear_model(X, y, plot=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

    if plot:
        plt.subplots()
        plt.scatter(X_test, y_test, color='black', alpha=0.3, marker='+')
        plt.plot(X_test, y_pred, color='red', linewidth=0.5)

    return regressor, df





def find_linear_model_between_two_speeds(series1, series2, plot=True):
    """ Plot the correlation between the shared heights (use the
     linear model from LNG terminal lifing work)"""
    X = series1.values.reshape(-1,1)
    y = series2.values.reshape(-1,1)
    regressor1, df = train_a_linear_model(X, y, plot=plot)
    #print(f' The correlation coefficient between s1 and s2 is {s1.corr(s2)}')
    return regressor1, df





def find_sectorial_correlations(comb_df, name=''):
    regressor_dict = dict()
    sect_dict = get_dict_with_dfs_for_each_sector(comb_df, sector_size=30, dirstring='direction')
    corr_coef_df = pd.DataFrame(columns=sect_dict.keys(), index=[0])
    mean_corr = 0
    for sector, df in sect_dict.items():
        u1 = df['lt_speed'] # The long-term is x; we know it
        u2 = df['st_speed'] # the 'short term'; e.g. speed at the mast is what we want, so it's Y.
        corr_coeff = u1.corr(u2)
        print(f'Correlation in sector {sector} is {corr_coeff}')
        mean_corr += corr_coeff
        regressor, obs_v_predicted = find_linear_model_between_two_speeds(u1, u2, plot=True)
        regressor_dict[sector] = regressor
        corr_coef_df[sector] = corr_coeff
        plt.grid()
        plt.title(f'{name} Sector: {sector}')
        plt.xlabel('Long-term wind speed (m/s)')
        plt.ylabel('Short-term wind speed (m/s)')
    mean_corr = mean_corr/12
    print(f'Mean sector correlation is {mean_corr}')
    return regressor_dict, sect_dict, corr_coef_df




def make_a_long_term_prediction(long_term_df, regressor_dict):
    """The P part of MCP. Long-term df is typically a 20-year set of reanalysis data.
    regressor dict is a dict produced by the function `find_sectorial_correlations`.
    The function returns a timeseries with the predicted speed for all sectors combined."""
    lt_dict = get_dict_with_dfs_for_each_sector(long_term_df, dirstring='Wind_Direction')
    predicted_sect_speed = dict()
    for sector, df in lt_dict.items():
        print(sector)
        df = df.dropna()
        regressor = regressor_dict[sector]
        prediction = regressor.predict(df['Wind_Speed'].values.reshape(-1, 1)).flatten()
        predicted_sect_speed[sector] = pd.Series(index=df.index,
                                                 data=prediction)
    lt_df_prediction = pd.concat(predicted_sect_speed.values()).sort_index()
    return lt_df_prediction, predicted_sect_speed





def predicted_speeds(df_meas, df_re):
    """Returns a dataframe with the hourly wind speeds:
        - measured at the reference site,
        - extrapolated to 60m
        - predicted at the mast site
       """
    slope, intercept, r_value, p_value, std_err = linregress(df_meas().mean_mast_speed.tolist(), df_meas().df_re['Wind_Speed'].tolist())
    df = extrapolated_reference_data()
    df['site_speed'] = pd.Series((df['Wind_Speed'] - intercept) / slope)
    df = df.drop('Wind_Direction', 1)
    return df




def mean_speeds_from_predicted(predicted_speeds):
    """Returns a dataframe with yearly mean windspeeds for:
        - measured at the reference site,
        - predicted at the mast site
    """
    return predicted_speeds().groupby(predicted_speeds().index.year).mean()





def combine_meas_prediction_2015(predicted, df_hourly):
    """Returns dataframe which consist of the
       2015 predicted and measured windspeeds
    """
    df = pd.merge(predicted, df_hourly, how='outer', on='datetime')
    dfh = df.dropna('index')
    return dfh





def rms_error(dfh):
    """Returns the Root Mean Square Error between
       predicted and measured speeds
    """
    actu_list = dfh['Wind_Speed'].values.tolist()
    pred_list = dfh['Wind_Speed_meas'].values.tolist()
    mse = mean_squared_error(actu_list, pred_list)
    rmse = math.sqrt(mse)
    return rmse





#def bias_error_degrees():
#    """The problem is getting the average windspeed for the hour from the mast.
#       This is where wrap around is an issue.
#       The hourly data we have collected in joined_2012_data may contain
#       incorrect means wind directions per hour
#    """
#    mast_direction = hourly_wind_direction().as_matrix()
#    site_direction = joined_2012_data().wdir_deg.as_matrix()
#    bias = (mast_direction - site_direction).mean()
#    return bias






def long_term_uncertainty(dfh):
    """returns the standard deviation of the long term
       predicted yearly means
    """   
    dfh = lt_pred.to_frame()
    speeds = dfh['Wind_Speed'].values
    stdev = np.std(speeds, axis=0)
    return stdev





if __name__ == '__main__':

    # Load in measured data from 2015.
    measurement_data_2015 = pd.read_excel('measurement_data_2015.xlsx',
                       header=0,
                       usecols='B:F, L:M',
                       dtype={"Day":str, 'Month':str, 'Year':str, 'Hour':str, 'Minute':str})
    df_meas = make_datetime_index(measurement_data_2015)
    df_meas = clean_data(df_meas)   #   df_meas = dataframe of the measured data
    df_hourly = hourly_measured_windspeed(df_meas)
    
    # Load in reanalysis data
    reanalysis_data = pd.read_excel('reanalysis_data_2002-2016.xlsx',
                       header=0,
                       usecols='B:F, L:M',
                       dtype={"Day":str, 'Month':str, 'Year':str, 'Hour':str, 'Minute':str})
    df_re = make_datetime_index(reanalysis_data)
    df_re = clean_data(df_re)   #   df_re = dataframe of the reanalysis data

    # Combine short term and long term data sets and perform linear regression.
    df = combine_the_short_term_and_long_term_speed_measurements(df_re, df_meas)
    regressor_dict, sd, ccd = find_sectorial_correlations(df, 'Wind_Direction')
    lt_df_prediction, predicted_sect_speed = make_a_long_term_prediction(df_re, regressor_dict)
    #plt.subplots()
    #lt_df_prediction.plot()
    #plt.title('Predicted windspeed for 20 years')
    #plt.show()
    #lt_df_prediction.info()
    """Clean predicted series into a dataframe"""
    lt_df_prediction = pd.DataFrame({'datetime':lt_df_prediction.index, '':lt_df_prediction.values})
    lt_df_prediction = lt_df_prediction.rename({'':'Wind_Speed'}, axis = 1)

    """Get yearly mean windspeed from predicted data"""
    lt_pred = lt_df_prediction.groupby(
    [lt_df_prediction["datetime"].dt.year])["Wind_Speed"].mean()
    print(lt_pred)

    
    """Write file with the predicted windspeed for the time period"""
    #lt_df_prediction.to_csv("mcp_file.csv")


    """Get combined dataframe for the 2015 predicted and measured windspeeds"""
    df_2015 = combine_meas_prediction_2015(lt_df_prediction, df_hourly)


    """root mean squared is still bugging, I have more work to do on that"""
    rmse = rms_error(df_2015)
    print('root mean squared error =   {}'.format(rmse))


    """Standard deviation of the predicted windspeed"""
    stdev = long_term_uncertainty(lt_pred)
    print('standard deviation =   {}'.format(stdev))

