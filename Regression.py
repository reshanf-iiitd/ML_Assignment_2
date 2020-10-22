from sklearn.linear_model import LinearRegression
class Regression(object):
  def __init__(self,arg):
    super(Regression,self).__init__()
    self.arg=arg
  def fit(self,X_train,y_train):
    regr=LinearRegression()
    reg=regr.fit(X_train,y_train)	
    return reg
  def predict(self,X_test):
    inter=reg.intercept_
    coef=regr.coef_
    y_predicted=np.empty(len(X_test))
    for i in range(len(test_sample)):

      y_predicted[i] = inter + coef[0] * test_sample[i][0] + coef[1] * test_sample[i][1] + coef[2] * test_sample[i][2] + coef[3] * test_sample[i][3] + coef[4] * test_sample[i][4]+ coef[5] * test_sample[i][5] + coef[6] * test_sample[i][6] + coef[7] * test_sample[i][7]
      return y_predicted


  	# 	y_predicted[i] = inter + coef[0] * test_sample[i][0] + coef[1] * test_sample[i][1] + coef[2] * test_sample[i][2] + coef[3] * test_sample[i][3] + coef[4] * test_sample[i][4]+ coef[5] * test_sample[i][5] + coef[6] * test_sample[i][6] + coef[7] * test_sample[i][7]
    # return y_predicted

