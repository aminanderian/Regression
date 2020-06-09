import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

data=pd.read_csv(r'C:\Users\pc\Desktop\داده کاوی\Mega Project 2\test.csv')
data=data.loc[:,['lstat','medv']]
print(data.head(5))
data.plot(x='lstat',y='medv',style='o')
plt.xlabel('lstat')
plt.ylabel('medv')
plt.show()
X=pd.DataFrame(data['lstat'])
y=pd.DataFrame(data['medv'])
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
regressor=LinearRegression()
regressor.fit(X_train, y_train)
print(regressior.intercept_)
print(regressor.coef_)
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared  Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
