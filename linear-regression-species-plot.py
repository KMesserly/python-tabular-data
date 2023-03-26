import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy
from scipy import stats

def linear_regression(df, species):
    """
    Perform linear regression for a given species and create a scatter plot.

    Args:
        df (pd.DataFrame): The DataFrame containing the Iris data.
        species (str): The species to perform linear regression on.

    Returns:
        slope (float): The slope of the regression line.
        intercept (float): The y-intercept of the regression line.
    """
    # Extract data for the given species
    df_species = df[df['species'] == species]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_species['petal_length_cm'], df_species['sepal_length_cm'])

    # Create scatter plot
    plt.scatter(df_species['petal_length_cm'], df_species['sepal_length_cm'])
    plt.plot(df_species['petal_length_cm'], slope * df_species['petal_length_cm'] + intercept, color='red')
    plt.xlabel('Petal length (cm)')
    plt.ylabel('Sepal length (cm)')
    plt.title('Linear regression for ' + species)

    # Save plot to a file
    plt.savefig(species + '_linear_regression.png')

    return slope, intercept

if __name__ == '__main__':
    # Load Iris data
    df = pd.read_csv('iris.csv')

    # Perform linear regression and create plot for each species
    for species in df['species'].unique():
        slope, intercept = linear_regression(df, species)
        print('Linear regression for', species, ': slope =', slope, ', intercept =', intercept)

