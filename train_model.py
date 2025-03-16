import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
df = pd.read_csv('Fish.csv')

# Feature selection (using Length1, Length2, Length3, Height, Width to predict Weight)
X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Weight']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model as a .pkl file
with open('fish_weight_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as fish_weight_model.pkl")