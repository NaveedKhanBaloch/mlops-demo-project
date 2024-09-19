# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Display the first few rows of the dataset
print(titanic_data.head())


# %% [markdown]
# Step 3: Perform Basic EDA
# 3.1: Summary of the Dataset

# %%
# Display basic information about the dataset
print("\nDataset Information:")
print(titanic_data.info())

# Check for missing values
print("\nMissing values in each column:")
print(titanic_data.isnull().sum())

# Display statistical summary of the dataset
print("\nStatistical Summary:")
print(titanic_data.describe())


# %% [markdown]
# Step 4: Visualizing Data Distributions Using matplotlib and seaborn
# 4.1: Histogram for Age and Fare

# %%
# Plot histogram for 'Age' and 'Fare' columns
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(titanic_data['Age'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(titanic_data['Fare'], bins=20, color='salmon', edgecolor='black')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# %% [markdown]
# 4.2: Scatter Plot for Age vs. Fare

# %%
# Scatter plot to visualize Age vs. Fare
plt.figure(figsize=(8, 6))
plt.scatter(titanic_data['Age'], titanic_data['Fare'], alpha=0.5, color='green')
plt.title('Age vs. Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.grid(True)
plt.show()


# %% [markdown]
# 4.3: Boxplot for Fare by Passenger Class

# %%
# Boxplot to compare Fare distribution by Passenger Class
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Fare', data=titanic_data)
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()


# %% [markdown]
# Step 5: Feature Engineering
# 5.1: Create a New Feature - FamilySize

# %%
# Create a new feature 'FamilySize'
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1

# Display the first few rows to confirm the new feature
print(titanic_data[['SibSp', 'Parch', 'FamilySize']].head())


# %% [markdown]
# 5.2: Create a New Feature - IsAlone

# %%
# Create a new feature 'IsAlone'
titanic_data['IsAlone'] = 1  # Initialize to 1 (is alone)
titanic_data['IsAlone'].loc[titanic_data['FamilySize'] > 1] = 0  # Update to 0 if FamilySize > 1

# Display the first few rows to confirm the new feature
print(titanic_data[['FamilySize', 'IsAlone']].head())


# %% [markdown]
# Step 6: Data Cleaning
# 6.1: Handling Missing Values
# Fill Missing Values in Age with Median:

# %%
# Fill missing values in 'Age' with the median age
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)


# %% [markdown]
# Fill Missing Values in Embarked with Mode:

# %%
# Fill missing values in 'Embarked' with the most common port
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# %% [markdown]
# Drop Cabin Column Due to Excessive Missing Values:

# %%
# Drop the 'Cabin' column as it has too many missing values
titanic_data.drop(columns=['Cabin'], inplace=True)


# %% [markdown]
# Confirm Missing Values:

# %%
# Confirm that missing values have been addressed
print("\nMissing values after cleaning:")
print(titanic_data.isnull().sum())


# %% [markdown]
# 6.2: Handling Outliers in Fare

# %%
# Remove outliers in 'Fare' (e.g., fare > 300 considered as an outlier)
titanic_data = titanic_data[titanic_data['Fare'] < 300]

# Boxplot for Fare by Passenger Class after removing outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Fare', data=titanic_data)
plt.title('Fare Distribution by Passenger Class (Outliers Removed)')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()


# %% [markdown]
# Step 7: Save the Cleaned Dataset

# %%
# Save the cleaned dataset to a new CSV file
titanic_data.to_csv('titanic_cleaned.csv', index=False)
print("Cleaned dataset saved as 'titanic_cleaned.csv'")
