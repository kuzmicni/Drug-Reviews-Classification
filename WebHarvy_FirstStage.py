import numpy as np
import pandas as pd
import matplotlib.pyplot
import scipy as sp
import tensorflow as TF
import keras as K
import seaborn as sns
import scipy as sp
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from __future__ import division

# import tables from the database
import MySQLdb
db = MySQLdb.connect(host = "mytestinstance.cylp6sglqzny.ca-central-1.rds.amazonaws.com",
                     user = "meysamtestdb",
                     passwd = "liebenswert",
                     db = "dbname")
df_Region01_new = pd.read_sql('SELECT * FROM new_complete_region_01', con=db)
df_Region02_new = pd.read_sql('SELECT * FROM new_complete_region_02', con=db)
df_Region03_new = pd.read_sql('SELECT * FROM new_complete_region_03', con=db)
df_Region04_new = pd.read_sql('SELECT * FROM new_complete_region_04', con=db)
df_Region05_new = pd.read_sql('SELECT * FROM new_complete_region_05', con=db)
df_Region06_new = pd.read_sql('SELECT * FROM new_complete_region_06', con=db)
df_Region07_new = pd.read_sql('SELECT * FROM new_complete_region_07', con=db)
df_Region08_new = pd.read_sql('SELECT * FROM new_complete_region_08', con=db)
df_Region09_new = pd.read_sql('SELECT * FROM new_complete_region_09', con=db)
df_Region10_new = pd.read_sql('SELECT * FROM new_complete_region_10', con=db)
df_Region11_new = pd.read_sql('SELECT * FROM new_complete_region_11', con=db)

# import address and MLS# from realtor.ca
df_Region01_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_complete_region_01.xlsx")
df_Region02_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_comlpete_region_02.xlsx")
df_Region03_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_comlpete_region_03.xlsx")
df_Region04_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_complete_region_04.xlsx")
df_Region05_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_comlpete_region_05.xlsx")
df_Region06_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_complete_region_06.xlsx")
df_Region07_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_comlpete_region_07.xlsx")
df_Region08_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_comlpete_region_08.xlsx")
df_Region09_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_complete_region_09.xlsx")
df_Region10_new = pd.read_excel("/Users/meysam/Documents/Windows_Instance_WebHarvy/new_comlpete_region_10.xlsx")

# from the second time on, use the saved realtor file
df_realtor = pd.read_excel("/Users/meysam/Documents/realtor.xlsx")

# import Enbridge customers dataset
df_Enbridge_Customers = pd.read_csv("/Users/meysam/Documents/partner_customers.csv")

# columns importing from realtor.ca:
# MLS/address/desc/storeys/land_size/built_in/beds_above/beds_below/bath/appliances/basement/basement_type/style/arch_type/floor_area/cooling/heating

# DetailPageURL
# address
# MLS_number
# description
# storeys
# land_size
# built_in
# beds_above
# beds_below
# baths
# appliances
# basement
# basement_type
# style
# architecture_type
# floor_area
# heating system type and fuel
# cooling system type

# drop unnecessary columns
df_Enbridge_Customers = df_Enbridge_Customers.drop(['secondary_id','mail_address_line_2','mail_address_line_3','mail_address_line_4','first_name','phone_1',
                                                    'phone_2','language_preference','business_name','other_business','other_business_segment','programs',
                                                    'termination_reason','voltage_class','latitude','longitude','photovoltaic','heat_type','assess_value',
                                                    'total_rooms','dwelling_type','bldg_sq_foot','parcel_sq_foot','year_built','bedrooms','user_id',
                                                    'partner_id','meter_type','pool'],axis=1)

# drop null values
df_Enbridge_Customers = df_Enbridge_Customers.dropna(subset=['mail_address_line_1'], axis = 0, inplace=False)

# concat the five regions files to one DataFrame
frames_new = [df_Region01_new, df_Region02_new, df_Region03_new, df_Region04_new, df_Region05_new,
              df_Region06_new, df_Region07_new, df_Region08_new, df_Region09_new, df_Region10_new,
              df_Region11_new]
df_realtor_new = pd.concat(frames_new)

# add a date_in column to the new data in the first time
now = datetime.datetime.now()
df_realtor['date_in'] = now.strftime("%Y-%m-%d")

# drop duplicates if any
df_realtor_new.drop_duplicates(keep=False, inplace=True)

# drop DetailPageURL
df_realtor_new = df_realtor_new.drop(['DetailPageURL'],axis=1)

# split the strings (address to street_number, street_name, city, and province)
df_realtor_new['street'],df_realtor_new['city'],df_realtor_new['province'] = df_realtor_new.address.str.split(', ',2).str
df_realtor_new['street_number'],df_realtor_new['street_name'] = df_realtor_new.street.str.split(' ',1).str

# drop null values in address
df_realtor_new.dropna(subset=['city'],axis='rows',inplace=True) # address not available

# change the street number to numeric, if necessary
# df_realtor['street_number'] = df_realtor.to_numeric()

# capitalize all strings (address)
# df_realtor_object = df_realtor.select_dtypes(include='object')
df_realtor_new['street'] = df_realtor_new['street'].apply(lambda x: " ".join(x.upper() for x in x.split()))
df_realtor_new['city'] = df_realtor_new['city'].apply(lambda x: " ".join(x.upper() for x in x.split()))
df_realtor_new['province'] = df_realtor_new['province'].apply(lambda x: " ".join(x.upper() for x in x.split()))
df_realtor_new['street_name'] = df_realtor_new['street_name'].apply(lambda x: " ".join(x.upper() for x in x.split()))
df_realtor_new['address'] = df_realtor_new['address'].apply(lambda x: " ".join(x.upper() for x in x.split()))

# changing Enbridge attributes according to MEX data dictionary

# change column names in Enbridge dataset
df_Enbridge_Customers.rename(columns={'mail_address_line_1': 'street'}, inplace=True) # this includes street_number and street_name

df_MLS_new = df_realtor_new[['MLS']]
df_MLS = df_realtor[['MLS','date_in']]

# merge old and new MLS datasets in three ways
df_MLS['ones'] = 1
df_MLS_new['ones'] = 1
# method_1: left (new) new in the market
df = pd.merge(df_MLS_new, df_MLS, on = 'MLS', how = 'left')
df_new_market = df[df.ones_y.isnull()]
# add a date_in column
now = datetime.datetime.now()
df_new_market['date_in'] = now.strftime("%Y-%m-%d")
# method_2: inner (similar) still in the market
df_still_market = pd.merge(df_MLS_new, df_MLS, on='MLS', how='inner')
# drop the similar columns from the left dataset (realtor_new)

# method_3: right (old) out of the market
df = pd.merge(df_MLS_new, df_MLS, on = 'MLS', how = 'right')
df_sold_market = df[df.ones_x.isnull()]
# add a date_out column
now = datetime.datetime.now()
df_sold_market['date_out'] = now.strftime("%Y-%m-%d")

# delete unnecessary columns
df_new_market = df_new_market.drop(['ones_x','ones_y'],axis=1)
df_still_market = df_still_market.drop(['ones_x','ones_y'],axis=1)
df_sold_market = df_sold_market.drop(['ones_x','ones_y'],axis=1)

# merge all three dataframes to df_realtor
df = pd.concat([df_new_market, df_sold_market])
df = pd.concat([df, df_still_market])

# from the second day on, the updated dataset is saved at this point for the following day (sold records are eliminated)
df_realtor = df[df.date_out.isnull()]
df_realtor = df_realtor.drop(['date_out'],axis=1)

# from the second day on, the updated dataset is saved at this point for the following day (sold records are eliminated)
df_realtor_out = df[df.date_out.notnull()]
df_realtor_out.to_excel("/Users/meysam/Documents/realtor_old.xlsx")

df_reset_new = df_realtor_new.reset_index(drop=True)
df_reset = df_realtor.reset_index(drop=True)

# merge the updated MLS data with the features dataframe
df_merge = pd.merge(df_reset_new, df_reset, on = 'MLS', how='inner')

df_merge.to_excel("/Users/meysam/Documents/realtor.xlsx") # this should be saved with a new name everyday

# make sure the addresses are in the same format before merging
# street suffix in realtor.ca is both complete and abbreviated, in Enbridge abbreviated only
read_dictionary = np.load('street_suffix_dictionary.npy').item()
# regions in GTA in realtor.ca is all Toronto (North York, York, Etobicoke, Scarborough), but in Enbridge separated
# merge MLS and Enbridge datasets on 'street'
df = df_merge.merge(df_Enbridge_Customers, on = 'street', how='inner')

# convert the heating system to MEX data dictionary

# heating systems in MEX data dictionary:
# central_air_furnace/hot_water_boiler/air_source_heat_pump/electric_space_heater/electric_baseboard_heater/
# mini_split_heat_pump/other/radiator_hot_water/ground_source_heat_pump/rooftop_unit/fireplace

# heating systems in realtor.ca:
# forced air
# radiant heat
# baseboard heater
# hot water radiator

df_realtor['heating_type'],df_realtor['heating_fuel'] = df_realtor.heating.str.split('(',1).str
df_realtor['heating_fuel'] = df_realtor.heating.apply(lambda st: st[st.find("(")+1:st.find(")")])

# change the empty fields with 'not_available'
df_realtor['heating_type'] = df_realtor['heating_type'].replace({' ': 'Not_Available'})
# change the empty fields with 'not_available'
df_realtor['heating_type'] = df_realtor['heating_type'].replace({'': 'Not_Available'})
# change the 'Forced air' fields with 'Central Air Furnace'
df_realtor['heating_type'] = df_realtor['heating_type'].replace({'Forced air': 'Central Air Furnace'})
# change the 'Forced air ' fields with 'Central Air Furnace'
df_realtor['heating_type'] = df_realtor['heating_type'].replace({'Forced air ': 'Central Air Furnace'})
# change the 'Baseboard heaters' fields with 'Electric Baseboard Heater'
df_realtor['heating_type'] = df_realtor['heating_type'].replace({'Baseboard heaters': 'Electric Baseboard Heater'})
# change the 'Baseboard heaters' fields with 'Electric Baseboard Heater'
df_realtor['heating_type'] = df_realtor['heating_type'].replace({'Baseboard heaters ': 'Electric Baseboard Heater'})
# change the 'Radiant heat' fields with 'Radiator Hot Water'
df_realtor['heating_type'] = df_realtor['heating_type'].replace({'Radiant heat': 'Radiator Hot Water'})
# change the 'Hot water radiator heat' fields with 'Radiator Hot Water'
df_realtor['heating_type'] = df_realtor['heating_type'].replace({'Hot water radiator heat': 'Radiator Hot Water'})
# change the 'Boiler' fields with 'Radiator Hot Water'
df_realtor['heating_type'] = df_realtor['heating_type'].replace({'Boiler': 'Radiator Hot Water'})

# search for any info in the description column


# Feature Transformation:   INCLUDE ARCH_STYLE, E.G. BUNGALOW ONE STOREY
# check the architectural style
df_realtor['arc_style'] = df_realtor['arc_style'].replace({' ': 'not_specified'})
df_realtor['arc_style'],df_realtor['arc_style_secondary'] = df_realtor.arc_style.str.split('(',1).str
df_realtor = df_realtor.drop(['arc_style_secondary'],axis=1)
df_realtor.loc[df_realtor.arc_style == 'Bungalow', 'floors_above'] = 1
df_realtor.loc[df_realtor.arc_style == 'Raised bungalow', 'floors_above'] = 1
df_realtor.loc[df_realtor.arc_style == '2 Level', 'floors_above'] = 2
df_realtor.loc[df_realtor.arc_style == '3 Level', 'floors_above'] = 3
df_realtor.loc[df_realtor.arc_style == '4 Level', 'floors_above'] = 4

# interpret the values for floors above and below grade from 'storeys'
# floors_above: floors above ground including main floor
df_realtor['storeys'] = df_realtor['storeys'].replace({' ': '0'})
df_realtor['storeys'] = pd.to_numeric(df_realtor.storeys)
df_realtor.loc[df_realtor.storeys == 0, 'floors_above'] = 1
df_realtor.loc[df_realtor.storeys == 1, 'floors_above'] = 1
df_realtor.loc[df_realtor.storeys == 1.5, 'floors_above'] = 2
df_realtor.loc[df_realtor.storeys == 2, 'floors_above'] = 2
df_realtor.loc[df_realtor.storeys == 2.5, 'floors_above'] = 3
df_realtor.loc[df_realtor.storeys == 3, 'floors_above'] = 3
df_realtor.loc[df_realtor.storeys == 4, 'floors_above'] = 4


df_realtor['floors_below'] = 1

# change the building_type according to MEX data dictionary
df_realtor.rename(columns={'style':'building_type'}, inplace=True)
df_realtor['building_type'] = df_realtor['building_type'].replace({'Semi-detached': 'Semi-Detached'})

# eliminate the empty values in built_in column
# replace space with value 0
df_realtor['built_in'] = df_realtor['built_in'].replace({' ': '0'})
# change the construction year to numeric, if necessary
df_realtor['built_in'] = pd.to_numeric(df_realtor.built_in)

# eliminate sq.ft from floor area and change the type to numeric, if necessary
df_realtor['floor_space'] = df_realtor.floor_space.apply(lambda x: x.replace('sqft',''))
# or
# df_realtor.floor_area.apply(lambda x: x.strip('sqft'))
df_realtor['floor_space'] = pd.to_numeric(df_realtor.floor_space)

# interpret cooling system type to MEX data dictionary
# cooling systems in MEX data dictionary:
#
# cooling systems in realtor.ca:

# divide the cooling system into primary cooling by selecting the first one if more than is available

# divide the cooling column into three primary, secondary, and tertiary
df_realtor['cooling_type'],df_realtor['cooling_type_secondary'], df_realtor['cooling_type_tertiary'] = df_realtor.cooling.str.split(', ',2).str

# change the empty fields with 'Portable'
df_realtor['cooling'] = df_realtor['cooling'].replace({' ': 'Portable'})
# change the 'None' fields with 'Portable'
df_realtor['cooling'] = df_realtor['cooling'].replace({'None': 'Portable'})
# change the 'Wall unit' fields with 'Window Unit'
df_realtor['cooling'] = df_realtor['cooling'].replace({'Wall unit': 'Window Unit'})
# change the 'Window air conditioner' fields with 'Window Unit'
df_realtor['cooling'] = df_realtor['cooling'].replace({'Window air conditioner': 'Window Unit'})
# change the 'Air exchanger' fields with 'Air Source Heat Pump'
df_realtor['cooling'] = df_realtor['cooling'].replace({'Air exchanger': 'Air Source Heat Pump'})
# change the 'Air Conditioned' fields with 'Central Air System'
df_realtor['cooling'] = df_realtor['cooling'].replace({'Air Conditioned': 'Central Air System'})
# change the 'Central air conditioning' fields with 'Central Air System'
df_realtor['cooling'] = df_realtor['cooling'].replace({'Central air conditioning': 'Central Air System'})


# LabeblEncoder, all strings to numeric (import the saved fitted model)
# from sklearn import preprocessing
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# class MultiColumnLabelEncoder:
#     def __init__(self,columns = None):
#         self.columns = columns # array of column names to encode
#     def fit(self,X,y=None):
#         return self # not relevant here
#     def transform(self,X):
#         '''
#         Transforms columns of X specified in self.columns using
#         LabelEncoder(). If no columns specified, transforms all
#         columns in X.
#         '''
#         output = X.copy()
#         if self.columns is not None:
#             for col in self.columns:
#                 output[col] = LabelEncoder().fit_transform(output[col])
#         else:
#             for colname,col in output.iteritems():
#                 output[colname] = LabelEncoder().fit_transform(col)
#         return output
#     def fit_transform(self,X,y=None):
#         return self.fit(X,y).transform(X)
#
# df_object = df_incentives_input.select_dtypes(include='object')
# df_bool = df_incentives_input.select_dtypes(include='bool')
# MultiColumnLabelEncoder(columns = df_object.columns).fit(df_incentives_input)
# MultiColumnLabelEncoder(columns = df_bool.columns).fit(df_incentives_input)
# df_energy_plus_input = MultiColumnLabelEncoder(columns = df_object.columns).transform(df_energy_plus_input)
# df_energy_plus_input = MultiColumnLabelEncoder(columns = df_bool.columns).transform(df_energy_plus_input)
# rescaled_realtor = scaler.transform(df_realtor_input)

# loading the saved LabelEncoder dictionaries
le_dict_cooling_systems_fuel = np.load('/Users/meysam/Documents/Label_Encoder_Dictionaries/le_dict_cooling_systems_fuel.npy').item()
le_dict_cooling_systems_type = np.load('/Users/meysam/Documents/Label_Encoder_Dictionaries/le_dict_cooling_systems_type.npy').item()
le_dict_heatingfuel = np.load('/Users/meysam/Documents/Label_Encoder_Dictionaries/le_dict_heatingfuel.npy').item()
le_dict_heating_systems_type_primary = np.load('/Users/meysam/Documents/Label_Encoder_Dictionaries/le_dict_heating_systems_type_primary.npy').item()
le_dict_related_systems_fuel_DHW = np.load('/Users/meysam/Documents/Label_Encoder_Dictionaries/e_dict_related_systems_fuel_DHW.npy').item()
le_dict_related_systems_type_DHW = np.load('/Users/meysam/Documents/Label_Encoder_Dictionaries/le_dict_related_systems_type_DHW.npy').item()
le_dict_building_type = np.load('/Users/meysam/Documents/Label_Encoder_Dictionaries/le_dict_building_type.npy').item()
le_dict_zip = np.load('/Users/meysam/Documents/Label_Encoder_Dictionaries/le_dict_zip.npy').item()
le_dict_city = np.load('/Users/meysam/Documents/Label_Encoder_Dictionaries/le_dict_city.npy').item()

# print(le_dict_city['Toronto'])

# apply the LabelEncoder dictionaries to dataset
df_realtor['city'] = df_realtor['city'].apply(lambda x: le_dict_city.get(x, '<unknown_value>'))
df_realtor['building_type'] = df_realtor['building_type'].apply(lambda x: le_dict_building_type.get(x, '<unknown_value>'))
df_realtor['heating_systems_type_primary'] = df_realtor['heating_systems_type_primary'].apply(lambda x: le_dict_heating_systems_type_primary.get(x, '<unknown_value>'))
df_realtor['heatingfuel'] = df_realtor['heatingfuel'].apply(lambda x: le_dict_heatingfuel.get(x, '<unknown_value>'))
df_realtor['cooling_systems_type'] = df_realtor['cooling_systems_type'].apply(lambda x: le_dict_cooling_systems_type.get(x, '<unknown_value>'))

# reading the asigned value to a new item
# le_dict.get(new_item, '<Unknown>')

# ordering the features similar to training datset
df_realtor_input = df_realtor[['built_in','floor_space','floors_above','floors_below','building_type','heating_type','heating_fuel','cooling_type']]

# rescaling based on training dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_energy_inputs)
df_realtor_input_rescaled = scaler.transform(df_realtor_input)

# rescale the output
rescaledY = np.log1p(df_energy_output)

# transform the input using dnn model (import the fitted model)
model_inc = load_model('model_inc.h5')
model_total_savings = load_model('model_total_savings.h5')

results_inc = model_inc.predict(df_Cambridge_input_rescaled)
results_total_savings = model_building.predict(df_Cambridge_input_rescaled)

results_savings = np.expm1(results_savings)

dataset_inc = pd.DataFrame({'Column1':results_inc[:,0],'Column2':results_inc[:,1],'Column3':results_inc[:,2],
                            'Column4':results_inc[:,3],'Column5':results_inc[:,4],'Column6':results_inc[:,5],
                            'Column7':results_inc[:,6],'Column8':results_inc[:,7],'Column9':results_inc[:,8],
                            'Column10':results_inc[:,9],'Column11':results_inc[:,10],'Column12':results_inc[:,11],
                            'Column13':results_inc[:,12],'Column14':results_inc[:,13],'Column15':results_inc[:,14],
                            'Column16':results_inc[:,15],'Column17':results_inc[:,16],'Column18':results_inc[:,17],
                            'Column19':results_inc[:,18],'Column20':results_inc[:,19],'Column21':results_inc[:,20],
                            'Column22':results_inc[:,21],'Column23':results_inc[:,22],'Column24':results_inc[:,23],
                            'Column25':results_inc[:,24],'Column26': results_inc[:,25],'Column27':results_inc[:,26]})

dataset_savings = pd.DataFrame({'Column1':results_savings[:,0],'Column2':results_savings[:,1],
                                'Column3':results_savings[:,2],'Column4':results_savings[:,3],
                                'Column5':results_savings[:,4],'Column6':results_savings[:,5],
                                'Column7':results_savings[:,6]})

dataset_inc.to_csv('results_inc.csv')
dataset_savings.to_csv('results_savings.csv')