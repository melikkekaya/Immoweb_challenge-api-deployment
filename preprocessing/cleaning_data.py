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
        return 'Brussels Capital Region'
    elif 1300 <= zip_code <= 1499:
        return 'Walloon Brabant'
    elif 1500 <= zip_code <= 1999 or 3000 <= zip_code <= 3499:
        return 'Flemish Brabant'
    elif 2000 <= zip_code <= 2999:
        return 'Antwerp'
    elif 3500 <= zip_code <= 3999:
        return 'Limburg'
    elif 4000 <= zip_code <= 4999:
        return 'Liège'
    elif 5000 <= zip_code <= 5999:
        return 'Namur'
    elif 6000 <= zip_code <= 6599 or 7000 <= zip_code <= 7999:
        return 'Hainaut'
    elif 6600 <= zip_code <= 6999:
        return 'Luxembourg'
    elif 8000 <= zip_code <= 8999:
        return 'West Flanders'
    elif 9000 <= zip_code <= 9999:
        return 'East Flanders'
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
    # print("preprocess json:", json_data)

    df = pd.DataFrame.from_dict(json_data,orient='index')
    df.info()
    df = df.T
    df = df.set_axis(['Living area', 'Type', 'Number of rooms', 'Zip', 'Surface of the land', 'Garden', 'Garden surface', 'Kitchen values', 'Swimming pool', 'Furnished', 'Open fire', 'Terrace', 'Terrace surface', 'Number of facades', 'Building Cond. values'], axis=1)
    print(df.columns)
    print(df.info())

    # df = df.astype({
    #             "Living area":"float",
    #             "Number of rooms":"float",
    #             "Zip":"int",
    #             "Surface of the land":"float",
    #             "Garden surface":"float",
    #             "Terrace surface":"float",
    #             "Number of facades":"float",
    #             'Number of rooms':"float"
                

     

    #             'garden' : 'Garden',
    #             'garden_area' : 'Garden surface',
    #             'equipped_kitchen' : 'Kitchen values',
    #             'swimming_pool' : 'Swimming pool',
    #             'furnished' : 'Furnished',
    #             'open_fire' : 'Open fire',
    #             'terrace' : 'Terrace',
    #             'terrace_area' : 'Terrace surface',
    #             'facades_number': 'Number of facades',
    #             'building_state' : 'Building Cond. values'
    #             })
    
    print(df.info())
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
       'Urban_value']]
    
    # print(df.loc[:, ~df.columns.duplicated()])
    # final_json = df.to_json(df)
    return df.loc[:, ~df.columns.duplicated()]
    # return df

# json_data = {
# "area": 80,
# "property_type": "HOUSE",
# "rooms_number" : 4,
# "zip_code": 1000,
# "land_area": 180,
# "garden": 1,
# "garden_area": 50,
# "equipped_kitchen": True,
# "full_address": "dfkgfg",
# "swimming_pool": 0,
# "furnished": False,
# "open_fire": 0,
# "terrace": 0,
# "terrace_area": 0,
# "facades_number": 4,
# "building_state": "NEW"
# }

# print(preprocess(json_data))


# df= old_df.copy()
    
    # df = df.rename(columns={
    #     'area' : 'Living area',
    #     'property_type' : 'Type',
    #     'rooms_number': 'Number of rooms',
    #     'zip_code' : 'Zip',
    #     'land_area' : 'Surface of the land',
    #     'garden' : 'Garden',
    #     'garden_area' : 'Garden surface',
    #     'equipped_kitchen' : 'Kitchen values',
    #     'swimming_pool' : 'Swimming pool',
    #     'furnished' : 'Furnished',
    #     'open_fire' : 'Open fire',
    #     'terrace' : 'Terrace',
    #     'terrace_area' : 'Terrace surface',
    #     'facades_number': 'Number of facades',
    #     'building_state' : 'Building Cond. values'
    # })