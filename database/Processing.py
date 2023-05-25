import pandas as pd
import numpy as np
from Calculator import AutoDescriptor
import joblib

calculator = AutoDescriptor()

# Set read and write paths as well as path to the organizing file
read_path = "/home/taylo773/Quantum/GitHub/qregress/database/bse49-main/"
write_path = "/home/taylo773/Quantum/GitHub/qregress/database/processed/"
org_file = "/home/taylo773/Quantum/GitHub/qregress/database/bse49-main/BSE49_Existing.org"

# unpack the xyz file names and energies into corresponding lists
df = pd.read_csv(org_file, delimiter='|', usecols=[3, 5, 7, 8], names=['A', 'B', 'AB', 'BSE'])

nameA, nameB, nameAB = list(df['A']), list(df['B']), list(df['AB'])

# Remove excess white space at the ends of names and create path to file
for i in range(len(nameA)):
    nameA[i] = nameA[i][1:-1]
    nameA[i] = read_path + "Geometries/Existing/" + nameA[i] + ".xyz"
    nameB[i] = nameB[i][1:-1]
    nameB[i] = read_path + "Geometries/Existing/" + nameB[i] + ".xyz"
    print(nameAB[i])
    nameAB[i] = nameAB[i][1:-1]
    print(nameAB[i])
    nameAB[i] = read_path + "Geometries/Existing/" + nameAB[i] + ".xyz"
    print(nameAB[i])
print("Starting to process data...\n")

# Calculate features for each set of molecules
dfA = calculator(nameA)
dfA.to_csv(write_path + "molsA.csv")
print("Dataframe A created")

dfB = calculator(nameB)
dfB.to_csv(write_path + "molsB.csv")
print("Dataframe B created")

dfAB = calculator(nameAB)
dfAB.to_csv(write_path + "molsAB.csv")
print("Dataframe AB created")

# ----------- A + B - AB Calculation ----------- #
print("Combining dataframes...\n")
df_result = dfA.loc[:, dfA.columns != 'XYZ'].add(dfB.loc[:, dfB.columns != 'XYZ'], fill_value=0)
df_result = df_result.subtract(dfAB.loc[:, dfAB.columns != 'XYZ'], fill_value=0)


# Now rewrite redundancies
for column in df_result.columns:
    temp_df = df_result[column]
    AB_temp_df = dfAB[column]
    score = 0
    for value in temp_df:
        if np.abs(value) > 10 ** -10:  # differences below 10^-10 are likely round errors
            score = 1
            break
        elif value == 0:
            pass
    if score == 0:
        for value in AB_temp_df:
            if value != 0:
                score = 1
                break
            else:
                pass
        if score == 0:
            A_temp_df = dfA[column]
            new_col = A_temp_df
        else:
            new_col = AB_temp_df
    else:
        new_col = temp_df
    df_result[column] = new_col


# Drop the features we don't want based on drop_cols
drop_cols = pd.read_csv("/home/taylo773/Quantum/GitHub/qregress/database/comparisons/drop_cols.csv")
for col in drop_cols['0']:
    df_result.drop('RDKit_desc_'+col, axis=1, inplace=True)

# Now add the BSEs to the df
data_col = df['BSE']
df_result = df_result.join(data_col)

df_result.to_csv(write_path + "BSE49_existing.csv")

joblib.dump(df_result, write_path + "BSE49_existing.bin")
print("Processing complete")
