# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load Data Frame

df = pd.read_csv('Iris.csv')

# EDA

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

print(df['Species'].value_counts())

# Drop Unecessary Column

df.drop(['Id'], axis= 1, inplace= True)

print(df.shape)

# Data Visualization

plt.figure(figsize= (10,6))
sns.scatterplot(data= df, x= 'PetalLengthCm', y= 'PetalWidthCm', hue = 'Species')
plt.title("Visualizing different classes")
plt.show()

sns.pairplot(data=df, hue= 'Species')
plt.show()

plt.figure(figsize= (10,6))
sns.boxplot(data = df, x= 'Species', y= 'SepalWidthCm', hue= 'Species')
plt.show()

# Splitting Data

X = df.drop(['Species'], axis= 1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, stratify= y, random_state= 42)

# Scaling Features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training

model = KNeighborsClassifier(n_neighbors= 5)

model.fit(X_train_scaled, y_train)

# Model Prediction

y_pred = model.predict(X_test_scaled)

# Model Evaluation

cr = classification_report(y_test, y_pred)
print("Classification Report: ", cr)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

