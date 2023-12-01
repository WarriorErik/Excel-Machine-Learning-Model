import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#1. EDA (Exploratory Data Analysis) 


# Load the dataset
data = pd.read_csv('titanic_passengers.csv')

# Get an overview of the data
print(data.head())
print(data.info())


#Visualizations


# Histogram for age distribution
sns.histplot(data['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()

# Bar plot for the survival rate with respect to Pclass (Ticket class)
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival by Ticket Class')
plt.show()


#Interesting Find: Age seems to be slightly right skewed, and the higher class passengers (Pclass = 1) have a higher survival rate.



#2. Data Processing




#Handling Missing Data

# Check for missing values
print(data.isnull().sum())

# Fill Age missing values with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Drop Cabin column due to excessive missing values
data.drop(columns='Cabin', inplace=True)

# Fill Embarked with mode (most common value)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)




#Feature Engineering


# Create a family size feature
data['FamilySize'] = data['SibSp'] + data['Parch']

# Drop unnecessary columns
data.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)






#3. Train Model

# Convert categorical columns to one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop(columns='Survived')
y = data_encoded['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#A Random Forest is essentially a collection of decision trees. Thought it was very interesting.
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)



#4. Model Validation

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
