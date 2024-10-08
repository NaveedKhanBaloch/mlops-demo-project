# Import necessary libraries
# SHIFT + ENTER to run the lines of code in an interactive way.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Display the first few rows of the dataset
print(titanic_data.head())
