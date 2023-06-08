import pandas as pd
import numpy as np
import urllib.parse
import re
from sklearn.impute import KNNImputer
from fastapi.encoders import jsonable_encoder

def handle_garden_terrace(df):
    for feature in ['Garden', 'Terrace']:
        conditions = [
            df[feature]== True,
            df[feature]== "Yes",
            (df[feature].isna()) & (df[feature + " surface"].isna()),
            df[feature + " surface"].notna()
        ]
        values = [1, 1, 0, 1]
        df[feature] = np.select(conditions, values)

        df.loc[(df[feature] == 0 ) & (df[feature + " surface"].isna()), feature + ' surface'] = 0
    return df

def get_province(zip_code):
    if 1000 <= zip_code <= 1299:
        return 1
    elif 1300 <= zip_code <= 1499:
        return 2
    elif 1500 <= zip_code <= 1999 or 3000 <= zip_code <= 3499:
        return 3
    elif 2000 <= zip_code <= 2999:
        return 4
    elif 3500 <= zip_code <= 3999:
        return 5
    elif 4000 <= zip_code <= 4999:
        return 6
    elif 5000 <= zip_code <= 5999:
        return 7
    elif 6000 <= zip_code <= 6599 or 7000 <= zip_code <= 7999:
        return 8
    elif 6600 <= zip_code <= 6999:
        return 9
    elif 8000 <= zip_code <= 8999:
        return 10
    elif 9000 <= zip_code <= 9999:
        return 11
    else:
        return 'Unknown'

def set_urbain(df):
    df_urbain = pd.read_csv('./utils/urbain.csv')
    postcode_set = set(df_urbain['Postcode'])
    df['Urban_value'] = df['Zip'].apply(lambda x: 1 if x in postcode_set else 0)
    return df

def boolean_to_num(df, cols):
    for col in cols:
        df[col] = df[col].replace(True, 1).replace(False, 0).replace('', np.nan).fillna(0)
    return df


# def provice_column(df):
#     df['Province'] = df['Zip'].apply(get_province)

#     provinces = ['Province_Antwerp', 'Province_Brussels Capital Region',
#        'Province_East Flanders', 'Province_Flemish Brabant',
#        'Province_Hainaut', 'Province_Limburg', 'Province_Liège',
#        'Province_Luxembourg', 'Province_Namur', 'Province_Walloon Brabant',
#        'Province_West Flanders']
#     df.insert(column=provinces, value=0)
#     # province_df = pd.get_dummies(df[['Province']])
#     df[f"Province_{df['Province']}"] = df['Province'].apply(lambda x: 1)
#     return df
    

def preprocess(json_data):
    df = pd.DataFrame(json_data,index=[0])
    df = df.rename(columns={
        'area' : 'Living area',
        'property_type' : 'Type',
        'rooms_number': 'Number of rooms',
        'zip_code' : 'Zip',
        'land_area' : 'Surface of the land',
        'garden' : 'Garden',
        'garden_area' : 'Garden surface',
        'equipped_kitchen' : 'Kitchen values',
        'swimming_pool' : 'Swimming pool',
        'furnished' : 'Furnished',
        'open_fire' : 'Open fire',
        'terrace' : 'Terrace',
        'terrace_area' : 'Terrace surface',
        'facades_number': 'Number of facades',
        'building_state' : 'Building Cond. values'
    })


    df = handle_garden_terrace(df)
    df['Province'] = df['Zip'].apply(get_province)
 
    building_cond_mapping = {"NEW": 4, "GOOD": 3, "TO RENOVATE": 1, "JUST RENOVATED": 3, "TO REBUILD": 0}
    df['Building Cond. values'] = df['Building Cond. values'].map(building_cond_mapping)


    df = set_urbain(df)

    df = boolean_to_num(df, ['Garden','Kitchen values','Swimming pool','Furnished','Open fire','Terrace'])
    
    # df = provice_column(df)
    # province = pd.get_dummies(df[['Province']])

    df = df[['Type','Living area', 
       'Surface of the land',
       'Number of rooms', 
       'Number of facades', 
       'Swimming pool', 'Furnished', 'Open fire',
       'Terrace', 'Terrace surface', 'Garden', 'Garden surface',
       'Kitchen values', 'Building Cond. values',
       'Urban_value','Province']]

    return df