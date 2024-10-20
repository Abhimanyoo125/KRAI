# K means clustering 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data 
labels = iris.target 

K = 4 
iterations = 10 
np.random.seed(42)
initial_indices = np.random.choice(data.shape[0], K, replace=False)
cluster_means = data[initial_indices]

for _ in range(iterations):
    distances = np.linalg.norm(data[:, np.newaxis] - cluster_means, axis=2) 
    cluster_assignments = np.argmin(distances, axis=1)
    new_cluster_means = np.array([data[cluster_assignments == k].mean(axis=0) for k in range(K)])

    if np.all(cluster_means == new_cluster_means):
        break

    cluster_means = new_cluster_means  

print("Final Cluster Means:")
for i, mean in enumerate(cluster_means):
    print(f"Cluster {i + 1}: {mean}")

plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap='viridis', marker='o', alpha=0.5)
plt.scatter(cluster_means[:, 0], cluster_means[:, 1], c='red', marker='X', s=200, label='Cluster Means')
plt.title('K-means Clustering on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()





# Linear regression 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

years_worked = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
cakes_made = np.array([6500, 7805, 10835, 11230, 15870, 16387])

model = LinearRegression()
model.fit(years_worked, cakes_made)

m = model.coef_[0]
b = model.intercept_

line_of_best_fit = m * years_worked + b

correlation_coef, _ = pearsonr(years_worked.flatten(), cakes_made)

years_future = np.array([10]).reshape(-1, 1)
cakes_predicted_10_years = model.predict(years_future)[0]

plt.scatter(years_worked, cakes_made, color='blue', label='Data Points')
plt.plot(years_worked, line_of_best_fit, color='red', label=f'Line of Best Fit: y = {m:.2f}x + {b:.2f}')
plt.title('Scatter Plot and Line of Best Fit')
plt.xlabel('Years Worked')
plt.ylabel('Cakes Made')
plt.legend()
plt.show()

(m, b, correlation_coef, cakes_predicted_10_years)






# Multiple linear regression 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = {
    'Interest_Rate': [2.75, 2.50, 2.75, 3.00, 3.25, 3.50, 3.50, 3.75, 3.75, 4.00],
    'Unemployment_Rate': [5.3, 5.0, 5.0, 5.5, 5.5, 5.5, 5.0, 5.0, 4.9, 4.9],
    'Stock_Index_Price': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1199, 1167, 1130]
}

df = pd.DataFrame(data)
X = df[['Interest_Rate', 'Unemployment_Rate']]
y = df['Stock_Index_Price']

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
coefficients = model.coef_

interest_rate = 3
unemployment_rate = 5.7
predicted_stock_price = model.predict([[interest_rate, unemployment_rate]])[0]

(intercept, coefficients, predicted_stock_price)




# Multiple linear regression Co2
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('car_data.csv')
X = df[['Volume', 'Weight']]
y = df['CO2']

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
coefficients = model.coef_

volume = 1300
weight = 3300
predicted_CO2 = model.predict([[volume, weight]])[0]

(intercept, coefficients, predicted_CO2)





# Logistic regression 1 and 2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('bank_data.csv')
X = df[['age', 'job', 'balance', 'duration', 'campaign']] 
y = df['y'] 
X = pd.get_dummies(X, drop_first=True)
y = y.map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy of Logistic Regression model: {accuracy * 100:.2f}%')

new_customer_df = pd.DataFrame(new_customer, columns=['age', 'job', 'balance', 'duration', 'campaign'])
new_customer_encoded = pd.get_dummies(new_customer_df, drop_first=True)

predicted_outcome = model.predict(new_customer_encoded)







# K-means clustering Longitude 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('countries_data.csv')
X = df[['Longitude', 'Latitude']]
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

print("Cluster Centers (Longitude, Latitude):")
print(kmeans.cluster_centers_)

plt.figure(figsize=(10, 6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering of Countries Based on Longitude and Latitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()





# Random Forest method 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names']) 
y = pd.Series(iris['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris['target_names'])

print(f"Accuracy of Random Forest model: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)






# Bay's classification method 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('Social_Network_Ads.csv')
X = df[['Age', 'EstimatedSalary']] 
y = df['Purchased'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy of Naive Bayes model: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)






# SVM model 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
X = df[['Age', 'EstimatedSalary']]  
y = df['Purchased'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy of SVM model: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)

def plot_decision_boundary(X, y, model):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=plt.cm.coolwarm)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=['red', 'green'][i], label=j)
    plt.title('SVM Decision Boundary')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

plot_decision_boundary(X_train, y_train, svm_classifier)
