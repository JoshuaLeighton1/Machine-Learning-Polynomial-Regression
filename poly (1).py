#the relationship between the height of a projectile and time


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#On my side I don't get any error, I have uploaded a screenshot as proof
#create a function that generates data based on a mathematical formula with velocity = 100m/s and the acceleration approximated to 5
def data(x):
    #create empty arrays for height and time
    h_val = []
    t_val = []
    for t in range(x):
        height = 0 + 100*t - 5*(t**2)
        h_val.append([height])
        t_val.append([t])
        
    return h_val, t_val
        
#training set

values = data(21)
x = values[1]
y = values[0]

x_train = x[:15]
y_train = y[:15]

#testing set

x_test = x[-15:]
y_test = y[-15:]

#create regression object
regressor = LinearRegression()
#train it
regressor.fit(x_train, y_train)
#choose 21 equally spaced values
xx = np.linspace(0,50, 21)
#create a prediction function
yy = regressor.predict(xx.reshape(xx.shape[0],1))
#plot the linear regression model
plt.plot(xx, yy)

#create a quadratic function
quadratic = PolynomialFeatures(degree =2)

X_train_quadratic = quadratic.fit_transform(x_train)
X_test_quatratic = quadratic.transform(x_test)
#create a linear regression object
regressor_quad = LinearRegression()
#fit it
regressor_quad.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic.transform(xx.reshape(xx.shape[0],1))

#plot the quadratic prediction 
plt.plot(xx, regressor_quad.predict(xx_quadratic), c = 'r', linestyle ='--')
plt.title('Graph of a projectiles height and time relationship with v = 100m/s')
plt.xlabel('time /s')
plt.ylabel('Height /m')
plt.axis([0,21,0,600])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()

#we can see that our predictive function models the data pretty well and that the linear regression doesn't model it well at all.
