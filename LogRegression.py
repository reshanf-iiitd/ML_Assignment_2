import matplotlib.pyplot as plt
import numpy as np

class LogRegression(object):

  """docstring for LogRegression."""
  multi_class=''
  def __init__(self, multi_class):

    super(LogRegression, self).__init__()
    LogRegression.multi_class = multi_class
  def OVR_cost_function(h,theta, y):
    m = len(y)
    cost = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
    return cost

  def OVR_gradient_descent(X,h,theta,y,m):
    # This function calculates the theta value by gradient descent
    gradient_value = np.dot(X.T, (h - y)) / m
    theta -= .0001 * gradient_value
    return theta
  def OVR_score(self,X, y):
    #This function compares the predictd label with the actual label to find the model performance
    score = sum(LogRegression.predict(X) == y) / len(y)
    return score

  def OVR_plot_cost(self,costh,costh1):
    # This function plot the Cost function value
    for cost,c in costh :
      plt.plot(range(len(cost)),cost)
      # plt.plot(range(len(cost1)),cost1)
      plt.title("Convergence Graph of Cost Function of type-" + str(c) +" vs All")
      plt.xlabel("Number of Iterations")
      plt.ylabel("Cost")
      plt.show()  
    for cost1,c1 in costh1   :
      # plt.plot(range(len(cost)),cost)
      plt.plot(range(len(cost1)),cost1)
      plt.title("Convergence Graph of Cost Function of type-" + str(c) +" vs All")
      plt.xlabel("Number of Iterations")
      plt.ylabel("Cost")
      plt.show()  

    ################# Sigmoid value for a given input score ####################
  def sigmoid(score):
     return (1 / (1 + np.exp(-score)))
  def predict_probability(features, weights):
    score = np.dot(features, weights)
    return LogRegression.sigmoid(score)

  def feature_derivative(errors, feature):
    derivative = np.dot(np.transpose(errors), feature)
    return derivative
###################### Log Likelihod #######################################
  def compute_log_likelihood(features, label, weights):
    indicator = (label==+1)
    scores    = np.dot(features, weights)
    ll        = np.sum((np.transpose(np.array([indicator]))-1)*scores - np.log(1. + np.exp(-scores)))

  # ll        = np.sum((np.transpose(np.array([indicator]))-1)*scores - np.log(1. + np.exp(-scores))) - (l2_penalty * np.sum(weights[1:]**2))
    return ll

#sum over all the training examples, and the derivative rteun number #################
  def l2_feature_derivative(errors, feature,weight,l2_penalty, feature_is_constant):
    derivative = np.dot(np.transpose(errors), feature)
    if not feature_is_constant:
      derivative -= 2 * l2_penalty * weight
    return derivative

  def l2_compute_log_likelihood(features, labels, weights, l2_penalty):

    indicator = (labels==+1)
    scores    = np.dot(features, weights)
    ll        = np.sum((np.transpose(np.array([indicator]))-1)*scores - np.log(1. + np.exp(-scores))) - (l2_penalty * np.sum(weights[1:]**2))
    return ll


  def h(X, theta):
    return LogRegression.sigmoid(X.dot(theta))
  
    






    """You can give any required inputs to the fit()"""
  def fit(self,features, labels, lr, epochs,val_features, val_labels,l2_penalty):

    bias      = np.ones((features.shape[0], 1))
    features  = np.hstack((bias, features))
    bias1      = np.ones((val_features.shape[0], 1))
    val_features  = np.hstack((bias1, val_features))
    LogRegression.weights = np.zeros((features.shape[1], 1))
    logs = []
    logs1=[]
    err_1=[]
    l2err_2=[]
    acc_1=[]
    l2acc_2=[]

      # loop over epochs times
    if(l2_penalty==0):

      if(LogRegression.multi_class=='binary'):
        for epoch in range(epochs):
          predictions = LogRegression.predict_probability(features, LogRegression.weights)
        # predict probability for each row in the datas
        # calculate the indicator value
          indicators = (labels==+1)

        # calculate the errors
          errors = np.transpose(np.array([indicators])) - predictions
          err_1.append(predictions)
        # loop over each weight coefficient
          for j in range(len(LogRegression.weights)):

          # calculate the derivative of jth weight cofficient
            derivative = LogRegression.feature_derivative(errors, features[:,j])
            LogRegression.weights[j] += lr * derivative

          ll = LogRegression.compute_log_likelihood(features, labels, LogRegression.weights)
          logs.append(ll)
          ll1 = LogRegression.compute_log_likelihood(val_features, val_labels, LogRegression.weights)
          logs1.append(ll1)

          # compute the log-likelihood Validation
        

          
        
      if(LogRegression.multi_class=='OVR'):
        LogRegression.theta = []
        LogRegression.cost = []

        features = np.insert(features, 0, 1, axis=1)
        m = len(labels)
        for i in np.unique(labels):

                #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
          y_onevsall = np.where(labels == i, 1, 0)
          theta = np.zeros(features.shape[1])
          cost = []
          for _ in range(epochs):
            z = features.dot(theta)
            h = LogRegression.sigmoid(z)
            theta = LogRegression.OVR_gradient_descent(features,h,theta,y_onevsall,m)
            cost.append(LogRegression.OVR_cost_function(h,theta,y_onevsall)) 
          LogRegression.theta.append((theta, i))
          LogRegression.cost.append((cost,i))

        LogRegression.theta1 = []
        LogRegression.cost1 = []

        val_features = np.insert(val_features, 0, 1, axis=1)
        m = len(val_labels)
        for i in np.unique(val_labels):

                #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
          y_onevsall = np.where(val_labels == i, 1, 0)
          theta = np.zeros(val_features.shape[1])
          cost = []
          for _ in range(epochs):
            z = val_features.dot(theta)
            h = LogRegression.sigmoid(z)
            theta = LogRegression.OVR_gradient_descent(val_features,h,theta,y_onevsall,m)
            cost.append(LogRegression.OVR_cost_function(h,theta,y_onevsall)) 
          LogRegression.theta1.append((theta, i))
          LogRegression.cost1.append((cost,i))

        return self
      
      if(LogRegression.multi_class=='OVO'):
        LogRegression.theta = []
        LogRegression.cost = []
        # print(":OVOV")

        features = np.insert(features, 0, 1, axis=1)
        m = len(labels)
        for i in np.unique(labels):

                #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
          y_onevsall = np.where(labels == i, 1, 0)
          theta = np.zeros(features.shape[1])
          cost = []
          for _ in range(epochs):
            z = features.dot(theta)
            h = LogRegression.sigmoid(z)
            theta = LogRegression.OVR_gradient_descent(features,h,theta,y_onevsall,m)
            cost.append(LogRegression.OVR_cost_function(h,theta,y_onevsall)) 
          LogRegression.theta.append((theta, i))
          LogRegression.cost.append((cost,i))

        LogRegression.theta1 = []
        LogRegression.cost1 = []

        val_features = np.insert(val_features, 0, 1, axis=1)
        m = len(val_labels)
        for i in np.unique(val_labels):

                #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
          y_onevsall = np.where(val_labels == i, 1, 0)
          theta = np.zeros(val_features.shape[1])
          cost = []
          for _ in range(epochs):
            z = val_features.dot(theta)
            h = LogRegression.sigmoid(z)
            theta = LogRegression.OVR_gradient_descent(val_features,h,theta,y_onevsall,m)
            cost.append(LogRegression.OVR_cost_function(h,theta,y_onevsall)) 
          LogRegression.theta1.append((theta, i))
          LogRegression.cost1.append((cost,i))

        return self



      
    else:

        ############################################## BINARY #########################################
      if(LogRegression.multi_class=='binary'):
        for epoch in range(epochs):
      # predict probability for each row in the dataset
          predictions = LogRegression.predict_probability(features, LogRegression.weights)
        # calculate the indicator value
          indicators = (labels==+1)
       # calculate the errors
          errors = np.transpose(np.array([indicators])) - predictions
          l2err_2.append(errors)
      # loop over each weight coefficient
          for j in range(len(LogRegression.weights)):

            isIntercept = (j==0)
          # calculate the derivative of jth weight cofficient
            derivative = LogRegression.l2_feature_derivative(errors, features[:,j], LogRegression.weights[j], l2_penalty, isIntercept)
            LogRegression.weights[j] += lr * derivative
          ll = LogRegression.compute_log_likelihood(features, labels, LogRegression.weights)
          logs.append(ll)
          
          ll1 = LogRegression.l2_compute_log_likelihood(val_features, val_labels, LogRegression.weights,l2_penalty)
          logs1.append(ll1)
        ##################################OVR###############################################
      if LogRegression.multi_class=='OVR':
        LogRegression.theta = []
        LogRegression.cost = []
        features = np.insert(features, 0, 1, axis=1)
        m = len(labels)
        for i in np.unique(labels):
                #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
          y_onevsall = np.where(labels == i, 1, 0)
          theta = np.zeros(features.shape[1])
          cost = []
          for _ in range(epochs):
            z = features.dot(theta)
            h = LogRegression.sigmoid(z)
            theta = LogRegression.OVR_gradient_descent(features,h,theta,y_onevsall,m)
            cost.append(LogRegression.OVR_cost_function(h,theta,y_onevsall)) 
          LogRegression.theta.append((theta, i))
          LogRegression.cost.append((cost,i))

        LogRegression.theta1 = []
        LogRegression.cost1 = []

        val_features = np.insert(val_features, 0, 1, axis=1)
        m = len(val_labels)
        for i in np.unique(val_labels):

                #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
          y_onevsall = np.where(val_labels == i, 1, 0)
          theta = np.zeros(val_features.shape[1])
          cost = []
          for _ in range(epochs):
            z = val_features.dot(theta)
            h = LogRegression.sigmoid(z)
            theta = LogRegression.OVR_gradient_descent(val_features,h,theta,y_onevsall,m)
            cost.append(LogRegression.OVR_cost_function(h,theta,y_onevsall)) 
          LogRegression.theta1.append((theta, i))
          LogRegression.cost1.append((cost,i))
        return self
      
      if(LogRegression.multi_class=='OVO'):
        LogRegression.theta = []
        LogRegression.cost = []

        features = np.insert(features, 0, 1, axis=1)
        m = len(labels)
        for i in np.unique(labels):

                #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
          y_onevsall = np.where(labels == i, 1, 0)
          theta = np.zeros(features.shape[1])
          cost = []
          for _ in range(epochs):
            z = features.dot(theta)
            h = LogRegression.sigmoid(z)
            theta = LogRegression.OVR_gradient_descent(features,h,theta,y_onevsall,m)
            cost.append(LogRegression.OVR_cost_function(h,theta,y_onevsall)) 
          LogRegression.theta.append((theta, i))
          LogRegression.cost.append((cost,i))

        LogRegression.theta1 = []
        LogRegression.cost1 = []

        val_features = np.insert(val_features, 0, 1, axis=1)
        m = len(val_labels)
        for i in np.unique(val_labels):

                #print('Descending the gradient for label type ' + str(i) + 'vs Rest')
          y_onevsall = np.where(val_labels == i, 1, 0)
          theta = np.zeros(val_features.shape[1])
          cost = []
          for _ in range(epochs):
            z = val_features.dot(theta)
            h = LogRegression.sigmoid(z)
            theta = LogRegression.OVR_gradient_descent(val_features,h,theta,y_onevsall,m)
            cost.append(LogRegression.OVR_cost_function(h,theta,y_onevsall)) 
          LogRegression.theta1.append((theta, i))
          LogRegression.cost1.append((cost,i))

        return self





        

        
    if(LogRegression.multi_class=='binary') :
      import matplotlib.pyplot as plt
      x = np.linspace(0, len(logs), len(logs))
      x1 = np.linspace(0, len(logs1), len(logs1))
      fig = plt.figure()
      plt.plot(x, logs,label="Training")
      plt.plot(x1,logs1,label="Validation")
      fig.suptitle('Training the classifier (without L2)')
      plt.xlabel('Epoch')
      plt.ylabel('Log-likelihood')
      plt.legend()
      # fig.savefig('train_without_l2.jpg')
      plt.show()
    # print(err_1[0])
      err_1=np.array(err_1)
      # print(err_1.shape)
      return LogRegression.weights
  def predict(self,X_test):

    if(LogRegression.multi_class=='binary'):
      bias_train     = np.ones((X_test.shape[0], 1))
  
      features_train = np.hstack((bias_train, X_test))
    

      test_predictions  = (LogRegression.predict_probability(features_train, LogRegression.weights).flatten()>0.5)
      return test_predictions

    if(LogRegression.multi_class=='OVR'):
      m=len(X_test)
      X_test = np.insert(X_test, 0, 1, axis=1)

      X_test = np.hstack((np.ones((m, 1)), X_test))

      X_predicted = [max((LogRegression.sigmoid(i.dot(theta)), c) for theta, c in LogRegression.theta)[1] for i in X_test ]

      return X_predicted


    if(LogRegression.multi_class=='OVO'):
      m=len(X_test)
      X_test = np.insert(X_test, 0, 1, axis=1)

      X_test = np.hstack((np.ones((m, 1)), X_test))

      X_predicted = [max((LogRegression.sigmoid(i.dot(theta)), c) for theta, c in LogRegression.theta)[1] for i in X_test ]

      return X_predicted
      # m = len(X_test)
      # X_test = np.hstack((np.ones((m, 1)), X_test))
      # LogRegression.theta=np.array(LogRegression.theta)
      # return np.argmax(LogRegression.h(X_test, LogRegression.theta.T), axis=1) + 1

    #   bias_train     = np.ones((X_test.shape[0], 1))
    # # bias_test      = np.ones((X_test.shape[0], 1))
    #   features_train = np.hstack((bias_train, X_test))
    #   # features_test  = np.hstack((bias_test, X_test))

    #   test_predictions  = (LogRegression.predict_probability(features_train, LogRegression.weights).flatten()>0.75)
    #   return test_predictions


   


        # return y_predicted
