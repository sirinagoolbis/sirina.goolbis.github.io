import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set seaborn style
sns.set(style="whitegrid")

# Load the dataset
file_path = '/Users/sirinagoolbis/predictive modeling proj/HRdata.csv'
df = pd.read_csv(file_path)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Column names and dataset info
print("\nColumn Names:\n", df.columns)
print("\nDataset Info:")
print(df.info())

# Filter to valid Attrition values
df = df[df['Attrition'].isin(['Yes', 'No'])]

# Convert numeric columns to proper types
numeric_cols = ['Age', 'DailyRate', 'MonthlyIncome']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for missing values
print("\nMissing Values Summary:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("HRdata_cleaned.csv", index=False)

# === Visualizations === #

# Attrition count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Attrition')
plt.title('Employee Attrition Count')
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Monthly Income vs Attrition
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome')
plt.title('Monthly Income by Attrition Status')
plt.tight_layout()
plt.show()

# Attrition by Job Role (count with labels)
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df, x='JobRole', hue='Attrition')
plt.title('Attrition by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Add count labels on bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Normalized Attrition by Job Role (%)
job_role_attrition = pd.crosstab(df['JobRole'], df['Attrition'], normalize='index') * 100
job_role_attrition = job_role_attrition[['Yes', 'No']]  # correct order

job_role_attrition.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
plt.title('Attrition Rate (%) by Job Role')
plt.ylabel('Percentage')
plt.xlabel('Job Role')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attrition')
plt.tight_layout()
plt.show()

# Sorted Attrition by Job Role
sorted_order = df[df['Attrition'] == 'Yes']['JobRole'].value_counts().index
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='JobRole', hue='Attrition', order=sorted_order)
plt.title('Attrition by Job Role (Sorted by Yes Count)')
plt.xlabel('Job Role')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calculate percentage of Attrition='Yes' per JobRole
attrition_yes_pct = df[df['Attrition'] == 'Yes']['JobRole'].value_counts(normalize=False)
job_role_counts = df['JobRole'].value_counts()
attrition_rate = (attrition_yes_pct / job_role_counts * 100).sort_values(ascending=False)

# Plot it
plt.figure(figsize=(12, 6))
sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette="coolwarm")
plt.title("Attrition Rate (%) by Job Role (Descending)")
plt.xlabel("Job Role")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
