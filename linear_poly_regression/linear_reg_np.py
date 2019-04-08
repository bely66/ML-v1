import numpy as np 
data = np.loadtxt("data_np.csv",delimiter=',')
X = data[:,:-1]
Y = data [:,-1]
points = X.shape[0]
W = np.zeros(X.shape[1]) # coefficients
b = 0 # intercept
regression_coef = [np.hstack((W,b))]
def MSEStep(X,y,W,B,learn_rate=0.05):
    y_pred = np.matmul(X, W) + b
    error = y - y_pred
    
    # compute steps
    W_new = W + learn_rate * np.matmul(error, X)
    b_new = b + learn_rate * error.sum()
    return W_new, b_new    


for _ in range (25):
        batch = np.random.choice(range(points), 20)
        X_batch = X[batch,:]
        y_batch = Y[batch]
        W, b = MSEStep(X_batch, y_batch, W, b, 0.05)
        regression_coef.append(np.hstack((W,b)))
reg_arr = regression_coef[25] 
y_pred = X[23,0]*reg_arr[0]+reg_arr[1]