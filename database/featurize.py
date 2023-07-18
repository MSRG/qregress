import pandas as pd
import numpy as np
from Calculator import AutoDescriptor
import joblib
import os
import click

calculator = AutoDescriptor()


@click.command()
@click.option('--write', type=click.Path(), default="/home/taylo773/Quantum/GitHub/qregress/database/processed/",
              help='Write path for processed files. ')
@click.option('--read', required=True, type=click.Path(), help='Read path for xyz files. ')
@click.option('--org', required=True, type=click.Path(exists=True), help="Path to org file specifying xyz file "
                                                                         "organization. ")
def main(write, read, org):
    """
    Takes a dataset of xyz files and translates it into a set rdkit generated features. Uses org file to read the
    organization of the xyz dataset. Reads finds xyz files within read path directory and writes processed files within
    the write directory.
    """
    # Set read and write paths as well as path to the organizing file
    # read_path = "/home/taylo773/Quantum/GitHub/qregress/database/bse49-main/Geometries/Existing/"
    # write_path = "/home/taylo773/Quantum/GitHub/qregress/database/processed/"
    # org_file = "/home/taylo773/Quantum/GitHub/qregress/database/bse49-main/BSE49_Existing.org"
    read_path = read
    write_path = write
    org_file = org
    file_name, ext = os.path.splitext(os.path.basename(org_file))

    # unpack the xyz file names and energies into corresponding lists
    df = pd.read_csv(org_file, delimiter='|', usecols=[3, 5, 7, 8], names=['A', 'B', 'AB', 'BSE'])

    nameA, nameB, nameAB = list(df['A']), list(df['B']), list(df['AB'])

    # Remove excess white space at the ends of names and create path to file
    for i in range(len(nameA)):
        if nameA[i][-1] == ' ':
            nameA[i] = nameA[i][1:-1]
        nameA[i] = read_path + nameA[i] + ".xyz"
        if nameB[i][-1] == ' ':
            nameB[i] = nameB[i][1:-1]
        nameB[i] = read_path + nameB[i] + ".xyz"
        if nameAB[i][-1] == ' ':
            nameAB[i] = nameAB[i][1:-1]
        nameAB[i] = read_path + nameAB[i] + ".xyz"
    print("Starting to process data...\n")

    # Calculate features for each set of molecules
    dfA = calculator(nameA)
    while True:
        save = input("Dataframe for molA created. Would you like to save? ")
        if save != 'Yes' and save != 'No':
            print("Please enter either 'Yes' or 'No' ")
        else:
            break
    if save == 'Yes':
        name = input('Enter file name... ')
        dfA.to_csv(write_path + name + ".csv")
        print("Dataframe A created as: " + name + ".csv")

    dfB = calculator(nameB)
    while True:
        save = input("Dataframe for molB created. Would you like to save? ")
        if save != 'Yes' and save != 'No':
            print("Please enter either 'Yes' or 'No' ")
        else:
            break
    if save == 'Yes':
        name = input('Enter file name... ')
        dfB.to_csv(write_path + name + ".csv")
        print("Dataframe B created as: " + name + ".csv")

    dfAB = calculator(nameAB)
    while True:
        save = input("Dataframe for molAB created. Would you like to save? ")
        if save != 'Yes' and save != 'No':
            print("Please enter either 'Yes' or 'No' ")
        else:
            break
    if save == 'Yes':
        name = input('Enter file name... ')
        dfAB.to_csv(write_path + name + ".csv")
        print("Dataframe AB created as: " + name + ".csv")

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
        df_result.drop('RDKit_desc_' + col, axis=1, inplace=True)

    # Now add the BSEs to the df
    data_col = df['BSE']
    df_result = df_result.join(data_col)

    print('Job done, now saving files... ')
    df_result.to_csv(write_path + file_name + ".csv")

    joblib.dump(df_result, write_path + file_name + ".bin")
    print("Processing complete. Files saved as: \n" + write_path + file_name + ".csv" +
          '\n' + write_path + file_name + ".bin")


if __name__ == '__main__':
    main()
