import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\Data Science\\Projects & Assignmets\\Files\\data\\data_weather.csv")
print("Columns: \n",data.columns)
print("Data: ",data)
print("Null: \n",data[data.isnull().any(axis=1)])
#Data Cleaning Steps
#We will not need the "number" column for each row so we can clean it.
del data['number']
#Let us drop null values using the pandas dropna function.
before_rows = data.shape[0]
print(before_rows)
data = data.dropna()
after_rows = data.shape[0]
print(after_rows)
#How many rows dropped due to cleaning?
print("Total rows dropped: ",before_rows - after_rows)
#Convert to a Classification Task
#Binarize the relative_humidity_3pm to 0 or 1.
clean_data = data.copy()
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99)*1
print(clean_data['high_humidity_label'])

#Target is stored in 'y'.
y = clean_data[['high_humidity_label']].copy()
clean_data['relative_humidity_3pm'].head()
print("Y Data: \n",y.head())

#Use 9am Sensor Signals as Features to Predict Humidity at 3pm
morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
                    'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
                    'rain_duration_9am']
print(len(morning_features))
X = clean_data[morning_features].copy()
print("Columns in X",X.columns)
print("Columns in y",y.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
print("X_train is as under:")
print(X_train.head())
print("X_test is as under:")
print(X_test.head())
print("y_train is as under:")
print(y_train.head())
print("y_test is as under:")
print(y_test.head())
print("Let us describe y_train")
print(y_train.describe)
# fit on train set
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10,random_state=0)
humidity_classifier.fit(X_train, y_train)
type(humidity_classifier)
#predict on Test set
predictions = humidity_classifier.predict(X_test)
print("Sample Predictions: \n",predictions[:10])
print("Sample Y Test(Actual Data): \n",y_test['high_humidity_label'][:10])
#Measure accuracy of the classifier
print("Accuracy: ",accuracy_score(y_true=y_test,y_pred=predictions))
