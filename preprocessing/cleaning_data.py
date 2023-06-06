import pandas as pd
import numpy as np
import urllib.parse
import re
from sklearn.impute import KNNImputer


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df = df.drop(df[df["Type"]=="house group"].index)
    df = df.drop(df[df["Type"]=="apartment group"].index)
    return df

def select_and_rename_columns(df):
    df = df[['id','Price','Zip','Type','Subtype','location',
       'Surroundings type',
       'Living area',
       'Bedrooms','Kitchen type','Bathrooms',
       'Building condition',
       'Construction year',
       'Number of frontages',
       'Covered parking spaces', 'Outdoor parking spaces',
       'Swimming pool',
       'Furnished',
       'How many fireplaces?','Surface of the plot',
       'Terrace','Terrace surface',
       'Garden','Garden surface',
       'Primary energy consumption','Energy class','Heating type'
    ]]

    df = df.rename(columns={
        'location' :'Locality',
        'Transaction Type' : 'Type of sale',
        'Number of frontages': 'Number of facades',
        'Bedrooms':'Number of rooms',
        'Kitchen type' : 'Fully equipped kitchen',
        'How many fireplaces?' : 'Open fire',
        'Surface of the plot' :'Surface of the land',
    })
    return df

def convert_and_clean(df, common_cols):
    def clean_and_convert(column):
        column = column.apply(lambda x: re.sub('\D+', '', str(x)))
        column = column.replace('', np.nan)
        return column

    for col in common_cols:
        df[col] = clean_and_convert(df[col])

    df = df.astype({"Price": "float", "Number of rooms": "float", "Living area": "float",
                    "Surface of the land": "float", "Terrace surface": "float", "Garden surface": "float",
                    "Number of facades": "float", "Primary energy consumption": "float"})
    return df

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

def nan_replacement(df, cols):
    for col in cols:
        df[col] = df[col].replace("Yes", 1).replace("No", 0).replace('', np.nan).fillna(0)
    return df

def handle_categorical_columns(df): 
    kitchen_mapping = {'Not installed': 0, 'Installed': 1, 'Semi equipped': 1, 'Hyper equipped': 1, 'USA uninstalled': 0,
                       'USA installed': 1, 'USA semi equipped': 1, 'USA hyper equipped': 1}
    building_cond_mapping = {'To restore': 0, 'To be done up': 2, 'Just renovated': 3, 'To renovate': 1, 'Good': 3, 'As new': 4}

    df['Kitchen values'] = df['Fully equipped kitchen'].map(kitchen_mapping).fillna(df['Fully equipped kitchen'])
    df['Building Cond. values'] = df['Building condition'].map(building_cond_mapping).fillna(df['Building condition'])

    df = df.drop(columns=['Fully equipped kitchen', 'Building condition'])
    return df

def handle_parking(df):
    conditions = [
        (df["Covered parking spaces"].notna()) & (df["Outdoor parking spaces"].notna()),
        (df["Covered parking spaces"].isna()) & (df["Outdoor parking spaces"].isna()),
        (df["Covered parking spaces"].isna()) & (df["Outdoor parking spaces"].notna()),
        (df["Covered parking spaces"].notna()) & (df["Outdoor parking spaces"].isna())
    ]
    values = [(df["Covered parking spaces"]+df["Outdoor parking spaces"]), 0, df["Outdoor parking spaces"],df["Covered parking spaces"]]
    df['Parking'] = np.select(conditions, values)

    df = df.drop(columns=["Covered parking spaces","Outdoor parking spaces"])

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

def remove_outliers(df, columns, n_std):
    for col in columns:
        mean = df[col].mean()
        sd = df[col].std()
        df = df[(df[col] <= mean+(n_std*sd))]
    return df

def one_convert_to_nan(column):
    column = column.replace(1.0, np.nan)
    return column

def knn_imputer(df, exclude_cols):
    other_cols = [col for col in df.columns if col not in exclude_cols]
    impute_knn = KNNImputer(n_neighbors=5)
    df[other_cols] = impute_knn.fit_transform(df[other_cols]).astype(float)
    return df

def set_urbain(df):
    df_urbain = pd.read_csv('./utils/urbain.csv')
    postcode_set = set(df_urbain['Postcode'])
    df['Urban_value'] = df['Zip'].apply(lambda x: 1 if x in postcode_set else 0)
    return df

def house_type(df):
    mansion = ["manor house","mansion","castle","exceptional property"]
    house = ["house","villa","bungalow","chalet","country cottage","farmhouse","mixed use building","town house"]
    other = ["apartment block","other property"]

    df["Mansion"] = df["Subtype"].apply(lambda x: 1 if x in mansion else 0)
    df["House_villa"] = df["Subtype"].apply(lambda x: 1 if x in house else 0)
    df["Other_house"] = df["Subtype"].apply(lambda x: 1 if x in other else 0)
    return df

def apt_type(df):
    df["Subtype"] = df["Subtype"].replace(np.nan,"apartment")

    apartment = ["apartment","ground floor","loft","service flat","flat studio","kot"]
    big_apt = ["penthouse","triplex","duplex"]

    df["Normal_apt"] = df["Subtype"].apply(lambda x: 1 if x in apartment else 0)
    df["Big_apt"] = df["Subtype"].apply(lambda x: 1 if x in big_apt else 0)
    return df


def main():
    df = load_and_preprocess_data("./utils/raw_data.csv")
    
    common_cols = ['Living area', 'Surface of the land', 'Terrace surface', 'Garden surface', 'Primary energy consumption']
    df = select_and_rename_columns(df)
    df = convert_and_clean(df, common_cols)
    df = handle_garden_terrace(df)
    df = handle_categorical_columns(df)
    df = nan_replacement(df, ['Furnished', 'Swimming pool', 'Open fire'])
    df = handle_parking(df)
    df = df.drop(df[df["Living area"].isna()].index)
    df['Locality'] = df['Locality'].apply(urllib.parse.unquote)
    df['Province'] = df['Zip'].apply(get_province)
    df = set_urbain(df)


    exclude_cols = ["Price","Type","Subtype","Locality","Surroundings type","Energy class","Heating type","Province"]

    apt_df = df[df["Type"] == "apartment"]
    apt_df = apt_df.drop(columns=['Surface of the land'])
    apt_df = remove_outliers(apt_df, ['Price'], 4)
    apt_df = remove_outliers(apt_df, ['Living area'], 3)
    apt_df = apt_df.drop(apt_df[apt_df["Number of rooms"]>6].index)
    apt_df = knn_imputer(apt_df, exclude_cols)
    apt_df = apt_type(apt_df)

    apt_df.to_csv("./utils/final_apartment.csv")


    house_df = df[df["Type"] == "house"]
    house_df = house_df.drop(house_df[house_df["Surface of the land"].isna()].index)
    house_df['Surface of the land'] = one_convert_to_nan(house_df['Surface of the land'])
    house_df = remove_outliers(house_df, ['Price'], 4)
    house_df = remove_outliers(house_df, ['Living area', 'Surface of the land'], 3)
    house_df = knn_imputer(house_df, exclude_cols)
    house_df = house_type(house_df)

    house_df.to_csv("./utils/final_house.csv")





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
    df = pd.DataFrame(json_data)

    df = df.rename(columns={
        'area' : 'Living area',
        'property-type' : 'Type',
        'rooms-number': 'Number of rooms',
        'zip-code' : 'Zip',
        'land-area' : 'Surface of the land',
        'garden' : 'Garden',
        'garden-area' : 'Garden surface',
        'terrace' : 'Terrace',
        'terrace-area' : 'Terrace surface',
        'facades-number': 'Number of facades',
        'equipped-kitchen' : 'Kitchen values',
        'open-fire' : 'Open fire',
        'full-address' : 'Address',
        'swimming-pool' : 'Swimming pool',
        'furnished' : 'Furnished',
        'building-state' : 'Building Cond. values'
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
       'Urban_value']]


    return df

    # common_cols = ['Living area', 'Surface of the land', 'Terrace surface', 'Garden surface', 'Primary energy consumption']
    # df = select_and_rename_columns(df)
    # df = convert_and_clean(df, common_cols)
    # df = handle_garden_terrace(df)
    # df = handle_categorical_columns(df)
    # df = nan_replacement(df, ['Furnished', 'Swimming pool', 'Open fire'])
    # df = handle_parking(df)
    # df = df.drop(df[df["Living area"].isna()].index)
    # df['Locality'] = df['Locality'].apply(urllib.parse.unquote)
    
    

    # df['zip_code'] = df['zip_code'].astype(str).str[:-2].astype(np.int64)
    # mapping = {73: 1, 64: 2, 60: 3, 76: 4, 71: 5, 56: 6, 79: 7, 55: 8, 41: 9, 61: 10, 44: 11, 62: 12, 70: 13, 65: 14, 48: 15, 40: 16, 45: 17, 68: 18, 77: 19, 43: 20, 50: 21, 95: 22,
    #            75: 23, 66: 24, 69: 25, 53: 26, 42: 27, 46: 28, 89: 29, 88: 30, 78: 31, 21: 32, 49: 33, 84: 34, 51: 35, 38: 36, 34: 37, 37: 38, 96: 39, 93: 40, 36: 41, 91: 42, 67: 43,
    #            97: 44, 86: 45, 92: 46, 99: 47, 87: 48, 94: 49, 33: 50, 85: 51, 39: 52, 24: 53, 28: 54, 22: 55, 23: 56, 26: 57, 47: 58, 32: 59, 80: 60, 35: 61, 15: 62, 98: 63, 25: 64,
    #            17: 65, 14: 66, 82: 67, 10: 68, 18: 69, 90: 70, 20: 71, 13: 72, 31: 73, 16: 74, 12: 75, 29: 76, 30: 77, 11: 78, 83: 79, 19: 80}
    # df['zip_code'].replace(mapping, inplace=True)

    # mapping_type = {'HOUSE': 1, 'APARTMENT': 0}
    # df['property_type'].replace(mapping_type, inplace=True)

    # mapping_kitchen = {False: 0, True: 1}
    # df['equipped_kitchen'].replace(mapping_kitchen, inplace=True)

    # mapping_pool = {False: 0, True: 1}
    # df['swimming_pool'].replace(mapping_pool, inplace=True)

    # mapping_state = {'NEW': 5, 'JUST RENOVATED': 4,
    #                  'GOOD': 3, 'TO RENOVATE': 2, 'TO REBUILD': 1}
    # df['building_state'].replace(mapping_state, inplace=True)

    # df['equipped_kitchen'].fillna(0, inplace=True)

    # df['land_area'].fillna(0, inplace=True)

    # df['terrace_area'].fillna(0, inplace=True)

    # df['facades_number'].fillna(1, inplace=True)

    # dictionary = df.to_dict(orient='records')

    # return dict1



# if __name__ == "__main__":
#     preprocess()


#  json_data = {
#     "area": int,
#     "property-type": "APARTMENT" | "HOUSE" ,
#     "rooms-number": int,
#     "zip-code": int,
#     "land-area": Optional[int],
#     "garden": Optional[bool],
#     "garden-area": Optional[int],
#     "equipped-kitchen": Optional[bool],
#     "full-address": Optional[str],
#     "swimming-pool": Optional[bool],
#     "furnished": Optional[bool],
#     "open-fire": Optional[bool],
#     "terrace": Optional[bool],
#     "terrace-area": Optional[int],
#     "facades-number": Optional[int],
#     "building-state": Optional[
#       "NEW" | "GOOD" | "TO RENOVATE" | "JUST RENOVATED" | "TO REBUILD"
#     ]}