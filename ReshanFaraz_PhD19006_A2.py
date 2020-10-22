import pandas as pd
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
import pandas as pdo
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import tree
import plotly.express as px
from sklearn.metrics import zero_one_loss
from tabulate import tabulate
from sklearn.manifold import TSNE 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

import scipy.io as sio

from Regression import Regression
from LogRegression import LogRegression


def U_mean(modell,givenn):
  c=0
  # print("inside  MEANSQUARE")
  for i in range(len(givenn)):
    z=(modell[i]-givenn[i])
    # print(z)
    z=z**2
    c=c+z
  # print(c)
  # print(c/len(givenn))
  return (c/len(givenn))
###########################################################################################################
def U_kfold(indexx,samplee,labell):
  n=len(samplee)
  if indexx==0:
    trainn_sample=samplee[:int(n*0.8)]
    validd_sample=samplee[int(n*0.8):]
    trainn_label=labell[:int(n*0.8)]
    validd_label=labell[int(n*0.8):]
  elif indexx==1:
    trainn1_sample=samplee[:int(n*0.2)]
    trainn2_sample=samplee[int(n*0.4):]
    trainn1_label=labell[:int(n*0.2)]
    trainn2_label=labell[int(n*0.4):]
    trainn_sample=np.concatenate((trainn1_sample,trainn2_sample))
    trainn_label=np.concatenate((trainn1_label,trainn2_label))
    validd_sample=samplee[int(n*0.2):int(n*0.4)]
    validd_label=labell[int(n*0.2):int(n*0.4)]
  elif indexx==2:
    trainn1_sample=samplee[:int(n*0.4)]
    trainn2_sample=samplee[int(n*0.6):]
    trainn1_label=labell[:int(n*0.4)]
    trainn2_label=labell[int(n*0.6):]
    trainn_sample=np.concatenate((trainn1_sample,trainn2_sample))
    trainn_label=np.concatenate((trainn1_label,trainn2_label))
    validd_sample=samplee[int(n*0.4):int(n*0.6)]
    validd_label=labell[int(n*0.4):int(n*0.6)]
  elif indexx==3:
    trainn1_sample=samplee[:int(n*0.6)]
    trainn2_sample=samplee[int(n*0.8):]
    trainn1_label=labell[:int(n*0.6)]
    trainn2_label=labell[int(n*0.8):]
    trainn_sample=np.concatenate((trainn1_sample,trainn2_sample))
    trainn_label=np.concatenate((trainn1_label,trainn2_label))
    validd_sample=samplee[int(n*0.6):int(n*0.8)]
    validd_label=labell[int(n*0.6):int(n*0.8)]
  else:
    trainn_sample=samplee[int(n*0.2):]
    validd_sample=samplee[:int(n*0.2)]
    trainn_label=labell[int(n*0.2):]
    validd_label=labell[:int(n*0.2)]

  return trainn_sample,trainn_label,validd_sample,validd_label
  
def U_acc(y_predicted_test,test_label):

  z=0
  for x1,x2 in zip(y_predicted_test,test_label):
    if(x1==x2):
      z+=1
  return (z/len(y_predicted_test))


def Q1_2():
	############################      ANSWER 1_1 & 1_2     #########################################################
	colnames=['Sex', 'Length', 'Diameter', 'Height','H_Weight','S_weight','V_weight','Sh_weight','Rings']  

	mat = pd.read_csv('Dataset.data',header=None,names=colnames,delimiter=' ')
	sample=np.array(mat[['Sex','Length', 'Diameter', 'Height','H_Weight','S_weight','V_weight','Sh_weight']])
	label=np.array(mat['Rings'])
	# print(label)

	for i in range(len(label)):
	  if sample[i][0]=='M':
	    sample[i][0]=1
	  elif sample[i][0]=='F':
	    sample[i][0]=0
	  else:
	    sample[i][0]=-1


	train_e=[]
	test_e=[]
	Utrain_e=[]
	Utest_e=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,sample,label)
	  # print(test_label)
	  regr = Regression("Custom") 
	  # print(len(test_sample))
	  # print(len(train_sample))
	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  # print(len(y_predicted_train))
	  reg=regr.fit(train_sample, train_label)
	  y_predicted_test=reg.predict(test_sample)
	  y_predicted_train=reg.predict(train_sample)
	  ans1=U_mean(y_predicted_test,test_label)
	  ans2=U_mean(y_predicted_train,train_label)
	  # print("Validation Error : ", ans1)
	  # print("Training Error :",ans2)
	  # y_predicted_train = regr.predict(train_sample)
	  # y_predicted_test = regr.predict(test_sample)
	  
	  train_e.append(mean_squared_error(y_predicted_train, train_label))
	  test_e.append(mean_squared_error(y_predicted_test, test_label))
	  Utest_e.append(ans1)
	  Utrain_e.append(ans2)

	df=pd.DataFrame({'Training MSE(sklearn)':train_e ,
	                 "Validation MSE(sklearn)":test_e,
	                  'Training MSE(Custom)':Utrain_e,
	                 'Validation MSE(Custom)' : Utest_e   
	})
	print(tabulate(df, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training MSE (custom) is :",sum(Utrain_e)/len(Utrain_e))
	print("Mean of Validation MSE (custom) is",sum(Utest_e)/len(Utest_e))
	print("Mean of Training MSE (sklearn)is :",sum(train_e)/len(train_e))
	print("Mean of Validation MSE(sklearn) is",sum(test_e)/len(test_e))

	  

def Q1_3():
		############################      ANSWER 1_3     #########################################################
	colnames=['Sex', 'Length', 'Diameter', 'Height','H_Weight','S_weight','V_weight','Sh_weight','Rings']  

	mat = pd.read_csv('Dataset.data',header=None,names=colnames,delimiter=' ')
	sample=np.array(mat[['Sex','Length', 'Diameter', 'Height','H_Weight','S_weight','V_weight','Sh_weight']])
	label=np.array(mat['Rings'])
	# print(sample.shape)

	for i in range(len(label)):
	  if sample[i][0]=='M':
	    sample[i][0]=1
	  elif sample[i][0]=='F':
	    sample[i][0]=0
	  else:
	    sample[i][0]=-1

	sample=np.array(sample,dtype='float')
	label=np.array(label,dtype='float')
	train_e3=[]
	test_e3=[]
	Utrain_e3=[]
	Utest_e3=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,sample,label)
	  m = len(train_sample)
	  train_sample_2=train_sample
	  x_bias = np.ones((m,1),dtype='float')
	  train_sample = np.append(x_bias,train_sample,axis=1)
	  x_transpose = np.transpose(train_sample)
	  x_transpose_dot_x = x_transpose.dot(train_sample)
	  temp_1 = np.linalg.inv(x_transpose_dot_x)
	  temp_2=x_transpose.dot(train_label)
	  theta =temp_1.dot(temp_2)

	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample_2))

	  for i in range(len(test_sample)):
	    y_predicted_test[i] = theta[0] + theta[1] * test_sample[i][0] + theta[2] * test_sample[i][1] + theta[3] * test_sample[i][2] + theta[4] * test_sample[i][3] + theta[5]* test_sample[i][4]+ theta[6] * test_sample[i][5] + theta[7] * test_sample[i][6] + theta[8] * test_sample[i][7]

	  for i in range(len(train_sample)):
	    y_predicted_train[i] = theta[0] + (theta[1] * train_sample_2[i][0]) + (theta[2] * train_sample_2[i][1]) + (theta[3] * train_sample_2[i][2]) + (theta[4] * train_sample_2[i][3]) + (theta[5] * train_sample_2[i][4]) + (theta[6] * train_sample_2[i][5]) + (theta[7] * train_sample_2[i][6]) + (theta[8] * train_sample_2[i][7])

	  ans1=U_mean(y_predicted_test,test_label)
	  ans2=U_mean(y_predicted_train,train_label)
	  
	  train_e3.append(mean_squared_error(y_predicted_train, train_label))
	  test_e3.append(mean_squared_error(y_predicted_test, test_label))
	  Utest_e3.append(ans1)
	  Utrain_e3.append(ans2)



	dd3=[x for x in range(1,6)]
	df3=pd.DataFrame({'K-Fold':dd3,
	                 'Training MSE(sklearn)':train_e3 ,
	                 "Validation MSE(sklearn)":test_e3,
	                  'Training MSE(Custom)':Utrain_e3,
	                 'Validation MSE(Custom)' : Utest_e3   
	})
	print(tabulate(df3, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training MSE (custom)is :",sum(Utrain_e3)/len(Utrain_e3))
	print("Mean of Validation MSE(custom) is",sum(Utest_e3)/len(Utest_e3))
	print("Mean of Training MSE (sklearn)is :",sum(train_e3)/len(train_e3))
	print("Mean of Validation MSE(sklearn) is",sum(test_e3)/len(test_e3))
  

  

def Q1_4():
	############################      ANSWER 1_4     #########################################################
	colnames=['Sex', 'Length', 'Diameter', 'Height','H_Weight','S_weight','V_weight','Sh_weight','Rings']  

	mat = pd.read_csv('Dataset.data',header=None,names=colnames,delimiter=' ')
	sample=np.array(mat[['Sex','Length', 'Diameter', 'Height','H_Weight','S_weight','V_weight','Sh_weight']])
	label=np.array(mat['Rings'])
	# print(label)

	for i in range(len(label)):
	  if sample[i][0]=='M':
	    sample[i][0]=1
	  elif sample[i][0]=='F':
	    sample[i][0]=0
	  else:
	    sample[i][0]=-1

	train_e=[]
	test_e=[]
	Utrain_e=[]
	Utest_e=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,sample,label)
	  regr = LinearRegression() 

	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  reg=regr.fit(train_sample, train_label)
	  y_predicted_test=regr.predict(test_sample)
	  y_predicted_train=regr.predict(train_sample)
	  ans1=U_mean(y_predicted_test,test_label)
	  ans2=U_mean(y_predicted_train,train_label)
	  
	  train_e.append(mean_squared_error(y_predicted_train, train_label))
	  test_e.append(mean_squared_error(y_predicted_test, test_label))
	  Utest_e.append(ans1)
	  Utrain_e.append(ans2)

	df=pd.DataFrame({'Training MSE(sklearn)':train_e ,
	                 "Validation MSE(sklearn)":test_e,
	                  'Training MSE(Custom)':Utrain_e,
	                 'Validation MSE(Custom)' : Utest_e   
	})
	print(tabulate(df, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training MSE (custom) is :",sum(Utrain_e)/len(Utrain_e))
	print("Mean of Validation MSE (custom) is",sum(Utest_e)/len(Utest_e))
	print("Mean of Training MSE (sklearn)is :",sum(train_e)/len(train_e))
	print("Mean of Validation MSE(sklearn) is",sum(test_e)/len(test_e))
	  



def Q2_1():
	########################################### ANSWER 2-a ##################################################
	mat1 = sio.loadmat('dataset_1.mat')
	l2=np.array(mat1['labels'][0])
	data2=[]

	s2=np.array(mat1['samples'])
	for i in range(len(s2)):
	  data2.append([s2[i][0],s2[i][1],int(l2[i])])
	rows=np.array(data2)
	columnNames=['x_value','y_value','label']
	dataframe = pd.DataFrame(data=rows, columns=columnNames)
	dataframe['label'] = dataframe['label'].astype(int)

	plt.figure(figsize=(10,6))
	sns.scatterplot(data=dataframe,x='x_value', y='y_value', hue='label',palette="deep")
	plt.legend(loc=4)
	plt.title("Scattered Plot of data",fontsize=20,color="w")
	plt.tight_layout()
	plt.show()

def Q2_b_c():
	################################################ Answer 2_c_b ########################################################
# In the upper cell
	mat2 = sio.loadmat('dataset_1.mat')

	samples=np.array(mat2['samples'])
	labels=np.array(mat2['labels'][0])

	train_e3=[]
	test_e3=[]
	Utrain_e3=[]
	Utest_e3=[]
	losss_t=[]
	losss_tra=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,samples,labels)
	  m = len(train_sample)
	  
	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  logg=LogRegression("binary")
	  logg.fit(train_sample, train_label,0.0001,1000,test_sample,test_label,0)

	  y_predicted_test=logg.predict(test_sample)
	  y_predicted_train=logg.predict(train_sample)
	  ans1=U_acc(y_predicted_test,test_label)
	  ans2=U_acc(y_predicted_train,train_label)
	  losss_t.append(zero_one_loss(y_predicted_test, test_label))
	  losss_tra.append(zero_one_loss(y_predicted_train, train_label))
	  
	  train_e3.append(accuracy_score(y_predicted_train, train_label))
	  test_e3.append(accuracy_score(y_predicted_test, test_label))
	  
	  Utest_e3.append(ans1)
	  Utrain_e3.append(ans2)



	dd3=[x for x in range(1,6)]
	df3=pd.DataFrame({'K-Fold':dd3,
	                 'Training Accuracy(sklearn)':train_e3 ,
	                 "Validation Accuracy(sklearn)":test_e3,
	                  'Training Accracy(Custom)':Utrain_e3,
	                 'Validation Accuracy(Custom)' : Utest_e3   
	})
	print(tabulate(df3, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Accuracy is :",sum(Utrain_e3)/len(Utrain_e3))
	print("Mean of Validation Accuracy is",sum(Utest_e3)/len(Utest_e3))

	dd33=[x for x in range(1,6)]
	df33=pd.DataFrame({'K-Fold':dd3,
	                 'Training Loss':losss_tra ,
	                 "Validation Loss":losss_t 
	})
	print(tabulate(df33, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Loss is :",sum(losss_tra)/len(losss_tra))
	print("Mean of Validation Loss is",sum(losss_t)/len(losss_t))



def Q2_d():
	################################################ Answer 2_d ########################################################

	mat2 = sio.loadmat('dataset_1.mat')

	samples=np.array(mat2['samples'])
	labels=np.array(mat2['labels'][0])
	# print(i)
	# GRID SEARCH TO FIND THE OPTIMAL 


	h=np.linspace(-10,10,500)
	# print(h)
	index=[x for x in range(0,1000) ]

	avg_h=[]
	c=0
	filename1 = 'optimal_model1.sav'
	filename2 = 'optimal_model2.sav'
	filename3 = 'optimal_model3.sav'
	filename4 = 'optimal_model4.sav'
	filename5 = 'optimal_model5.sav'
	for count in range(0,5):
	  train_sample1,train_label1,test_sample1,test_label1=U_kfold(count,samples,labels)
	  best_h=[]
	  for x in h:
	    logg=LogRegression('binary')
	    logg.fit(train_sample1, train_label1,0.0001,1000,test_sample1,test_label1,x)
	    y_predicted_test1=logg.predict(test_sample1)
	    ans1=U_acc(y_predicted_test1,test_label1)
	    # print(c)
	    # c+=1
	    # print(ans1)
	    best_h.append(ans1)
	  # print("Maximum Accuracy")
	  indexx= best_h.index(max(best_h))
	  # print("Index==",indexx)
	  # print("Max== ",max(best_h))
	  # print("MAX h value",)
	  avg_h.append(h[indexx])
	  # print(avg_h)
	  if count == 0:
	    joblib.dump(logg, filename1)
	  if count == 1:
	    joblib.dump(logg, filename2)
	  if count == 2:
	    joblib.dump(logg, filename3)
	  if count == 3:
	    joblib.dump(logg, filename4)
	  if count == 4:
	    joblib.dump(logg, filename5)

	maxx=sum(avg_h)/len(avg_h)

	print("Optimal L2 -- == " ,maxx)
	train_e3=[]
	test_e3=[]
	Utrain_e3=[]
	Utest_e3=[]
	losss_t=[]
	losss_tra=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,samples,labels)
	  m = len(train_sample)
	  
	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  logg=LogRegression("binary")
	  logg.fit(train_sample, train_label,0.0001,1000,test_sample,test_label,maxx)           #max=-1.2625
	  y_predicted_test=logg.predict(test_sample)
	  y_predicted_train=logg.predict(train_sample)
	  ans1=U_acc(y_predicted_test,test_label)
	  ans2=U_acc(y_predicted_train,train_label)

	  losss_t.append(zero_one_loss(y_predicted_test, test_label))
	  losss_tra.append(zero_one_loss(y_predicted_train, train_label))

	  
	  train_e3.append(accuracy_score(y_predicted_train, train_label))
	  test_e3.append(accuracy_score(y_predicted_test, test_label))
	  Utest_e3.append(ans1)
	  Utrain_e3.append(ans2)


	# filename = 'finalized_model_2_4.sav'
	# joblib.dump(model, filename)
	dd3=[x for x in range(1,6)]
	df3=pd.DataFrame({'K-Fold':dd3,
	                 'Training Accuracy(sklearn)':train_e3 ,
	                 "Validation Accuracy(sklearn)":test_e3,
	                  'Training Accracy(Custom)':Utrain_e3,
	                 'Validation Accuracy(Custom)' : Utest_e3   
	})
	print(tabulate(df3, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Accuracy is :",sum(Utrain_e3)/len(Utrain_e3))
	print("Mean of Validation Accuracy is",sum(Utest_e3)/len(Utest_e3))


	dd33=[x for x in range(1,6)]
	df33=pd.DataFrame({'K-Fold':dd3,
	                 'Training Loss':losss_tra ,
	                 "Validation Loss":losss_t 
	})
	print(tabulate(df33, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Loss is :",sum(losss_tra)/len(losss_tra))
	print("Mean of Validation Loss is",sum(losss_t)/len(losss_t))





def Q2_e():

################################################ Answer 2_4 ########################################################

	mat2 = sio.loadmat('dataset_1.mat')

	samples=np.array(mat2['samples'])
	labels=np.array(mat2['labels'][0])




	train_e3=[]
	test_e3=[]
	Utrain_e3=[]
	Utest_e3=[]
	losss_tra=[]
	losss_t=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,samples,labels)
	  m = len(train_sample)
	  
	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  clf = LogisticRegression(max_iter=1000)
	  clf=clf.fit(train_sample,train_label)
	  # logg=fit_log(train_sample, train_sample,0.0001,1000,test_sample,test_label,max)           #max=.521
	  y_predicted_test=clf.predict(test_sample)
	  y_predicted_train=clf.predict(train_sample)
	  ans1=U_acc(y_predicted_test,test_label)
	  ans2=U_acc(y_predicted_train,train_label)

	  losss_t.append(zero_one_loss(y_predicted_test, test_label))
	  losss_tra.append(zero_one_loss(y_predicted_train, train_label))
	  train_e3.append(clf.score(train_sample, train_label))
	  test_e3.append(clf.score(test_sample, test_label))
	  Utest_e3.append(ans1)
	  Utrain_e3.append(ans2)


	# filename = 'finalized_model_2_4.sav'
	# joblib.dump(model, filename)
	print("With L2 Regularization as in part d")
	dd3=[x for x in range(1,6)]
	df3=pd.DataFrame({'K-Fold':dd3,
	                 'Training Accuracy(sklearn)':train_e3 ,
	                 "Validation Accuracy(sklearn)":test_e3,
	                  'Training Accracy(Custom)':Utrain_e3,
	                 'Validation Accuracy(Custom)' : Utest_e3   
	})
	print(tabulate(df3, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Accuracy is :",sum(Utrain_e3)/len(Utrain_e3))
	print("Mean of Validation Accuracy is",sum(Utest_e3)/len(Utest_e3))

	dd33=[x for x in range(1,6)]
	df33=pd.DataFrame({'K-Fold':dd3,
	                 'Training Loss':losss_tra ,
	                 "Validation Loss":losss_t 
	})
	print(tabulate(df33, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Loss is :",sum(losss_tra)/len(losss_tra))
	print("Mean of Validation Loss is",sum(losss_t)/len(losss_t))


	train_e3=[]
	test_e3=[]
	Utrain_e3=[]
	Utest_e3=[]
	losss_t=[]
	losss_tra=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,samples,labels)
	  m = len(train_sample)
	  
	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  clf = LogisticRegression(max_iter=1000,C=10000)
	  clf=clf.fit(train_sample,train_label)
	  # logg=fit_log(train_sample, train_sample,0.0001,1000,test_sample,test_label,max)           #max=.521
	  y_predicted_test=clf.predict(test_sample)
	  y_predicted_train=clf.predict(train_sample)
	  ans1=U_acc(y_predicted_test,test_label)
	  ans2=U_acc(y_predicted_train,train_label)

	  losss_t.append(zero_one_loss(y_predicted_test, test_label))
	  losss_tra.append(zero_one_loss(y_predicted_train, train_label))

	  # train_label=train_label.reshape(-1,1)
	  # y_predicted_train=y_predicted_train.reshape(-1,1)
	  # train_label=np.transpose(train_label)
	  # print(y_predicted_train.shape)
	  train_e3.append(clf.score(train_sample, train_label))
	  test_e3.append(clf.score(test_sample, test_label))
	  Utest_e3.append(ans1)
	  Utrain_e3.append(ans2)


	# filename = 'finalized_model_2_4.sav'
	# joblib.dump(model, filename)
	print("Without l2 Regularization as in c")
	dd3=[x for x in range(1,6)]
	df3=pd.DataFrame({'K-Fold':dd3,
	                 'Training Accuracy(sklearn)':train_e3 ,
	                 "Validation Accuracy(sklearn)":test_e3,
	                  'Training Accracy(Custom)':Utrain_e3,
	                 'Validation Accuracy(Custom)' : Utest_e3   
	})
	print(tabulate(df3, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Accuracy is :",sum(Utrain_e3)/len(Utrain_e3))
	print("Mean of Validation Accuracy is",sum(Utest_e3)/len(Utest_e3))

	dd33=[x for x in range(1,6)]
	df33=pd.DataFrame({'K-Fold':dd3,
	                 'Training Loss':losss_tra ,
	                 "Validation Loss":losss_t 
	})
	print(tabulate(df33, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Loss is :",sum(losss_tra)/len(losss_tra))
	print("Mean of Validation Loss is",sum(losss_t)/len(losss_t))



def Q3_a():
	########################################### ANSWER 2-a ##################################################
	mat3 = sio.loadmat('dataset_2.mat')
	# print(mat3)
	l2=np.array(mat3['labels'][0])
	data2=[]

	s2=np.array(mat3['samples'])
	# print(s2.shape)
	# print(np.array(mat3['labels'].shape))
	for i in range(len(s2)):
	  data2.append([s2[i][0],s2[i][1],int(l2[i])])
	rows=np.array(data2)
	columnNames=['x_value','y_value','label']
	dataframe = pd.DataFrame(data=rows, columns=columnNames)
	dataframe['label'] = dataframe['label'].astype(int)

	plt.figure(figsize=(10,6))
	sns.scatterplot(data=dataframe,x='x_value', y='y_value', hue='label',palette="deep")
	plt.legend(loc=4)
	plt.title("Scattered Plot of data",fontsize=20,color="w")
	plt.tight_layout()
	plt.show()


def Q3_b():
		##################################################### ANSWER 3_B
	# import LogRegression as logg
	mat2 = sio.loadmat('dataset_2.mat')

	samples=np.array(mat2['samples'])
	labels=np.array(mat2['labels'][0])

	randomize = np.arange(len(labels))
	np.random.shuffle(randomize)
	samples=samples[randomize]
	labels=labels[randomize]

	# print(x_da)
	train_e3=[]
	test_e3=[]
	Utrain_e3=[]
	Utest_e3=[]
	losss_t=[]
	losss_tra=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,samples,labels)
	  m = len(train_sample)
	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  logg=LogRegression("OVR")
	  logg=logg.fit(train_sample,train_label,0.0001,1000,test_sample,test_label,0)           
	  y_predicted_test=logg.predict(test_sample)
	  y_predicted_train=logg.predict(train_sample)
	  ans1=U_acc(y_predicted_test,test_label)
	  ans2=U_acc(y_predicted_train,train_label)


	  
	  train_e3.append(accuracy_score(y_predicted_train, train_label))
	  test_e3.append(accuracy_score(y_predicted_test, test_label))
	  Utest_e3.append(ans1)
	  Utrain_e3.append(ans2)

	logg.OVR_plot_cost(logg.cost,logg.cost1)
	# filename = 'finalized_model_2_4.sav'
	# joblib.dump(model, filename)
	dd3=[x for x in range(1,6)]
	df3=pd.DataFrame({'K-Fold':dd3,
	                 'Training Accuracy(sklearn)':train_e3 ,
	                 "Validation Accuracy(sklearn)":test_e3,
	                  'Training Accracy(Custom)':Utrain_e3,
	                 'Validation Accuracy(Custom)' : Utest_e3   
	})
	print(tabulate(df3, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Accuracy is :",sum(Utrain_e3)/len(Utrain_e3))
	print("Mean of Validation Accuracy is",sum(Utest_e3)/len(Utest_e3))





def Q3_c():
		##################################################### ANSWER 3_C
	# import LogRegression as logg
	mat2 = sio.loadmat('dataset_2.mat')

	samples=np.array(mat2['samples'])
	labels=np.array(mat2['labels'][0])

	# print(x_da)
	train_e3=[]
	test_e3=[]
	Utrain_e3=[]
	Utest_e3=[]
	losss_t=[]
	losss_tra=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,samples,labels)
	  m = len(train_sample)
	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  logg=LogRegression("OVR")
	  logg=logg.fit(train_sample,train_label,0.0001,1000,test_sample,test_label,0.521)           
	  y_predicted_test=logg.predict(test_sample)
	  y_predicted_train=logg.predict(train_sample)
	  ans1=U_acc(y_predicted_test,test_label)
	  ans2=U_acc(y_predicted_train,train_label)


	  
	  train_e3.append(accuracy_score(y_predicted_train, train_label))
	  test_e3.append(accuracy_score(y_predicted_test, test_label))
	  Utest_e3.append(ans1)
	  Utrain_e3.append(ans2)

	logg.OVR_plot_cost(logg.cost,logg.cost1)
	# filename = 'finalized_model_2_4.sav'
	# joblib.dump(model, filename)
	dd3=[x for x in range(1,6)]
	df3=pd.DataFrame({'K-Fold':dd3,
	                 'Training Accuracy(sklearn)':train_e3 ,
	                 "Validation Accuracy(sklearn)":test_e3,
	                  'Training Accracy(Custom)':Utrain_e3,
	                 'Validation Accuracy(Custom)' : Utest_e3   
	})
	print(tabulate(df3, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Accuracy is :",sum(Utrain_e3)/len(Utrain_e3))
	print("Mean of Validation Accuracy is",sum(Utest_e3)/len(Utest_e3))





def Q3_d():
		######################################################### ANSWER 3d################################################
	mat2 = sio.loadmat('dataset_2.mat')

	samples=np.array(mat2['samples'])
	labels=np.array(mat2['labels'][0])

	randomize = np.arange(len(labels))
	np.random.shuffle(randomize)
	samples=samples[randomize]
	labels=labels[randomize]

	# print(x_da)
	train_e3=[]
	test_e3=[]
	Utrain_e3=[]
	Utest_e3=[]
	losss_t=[]
	losss_tra=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,samples,labels)
	  m = len(train_sample)
	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  clf = OneVsOneClassifier(
	  LinearSVC(random_state=0)).fit(train_sample, train_label)
	  # clf.predict(X_test[:10])         
	  y_predicted_test=clf.predict(test_sample)
	  y_predicted_train=clf.predict(train_sample)
	  ans1=U_acc(y_predicted_test,test_label)
	  ans2=U_acc(y_predicted_train,train_label)


	  
	  train_e3.append(accuracy_score(y_predicted_train, train_label))
	  test_e3.append(accuracy_score(y_predicted_test, test_label))
	  Utest_e3.append(ans1)
	  Utrain_e3.append(ans2)

	# logg.OVR_plot_cost(logg.cost,logg.cost1)
	# filename = 'finalized_model_2_4.sav'
	# joblib.dump(model, filename)
	dd3=[x for x in range(1,6)]
	print("ONE VS ONE Using Sklearn ")
	df3=pd.DataFrame({'K-Fold':dd3,
	                 'Training Accuracy(sklearn)':train_e3 ,
	                 "Validation Accuracy(sklearn)":test_e3,
	                  'Training Accracy(Custom)':Utrain_e3,
	                 'Validation Accuracy(Custom)' : Utest_e3   
	})
	print(tabulate(df3, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Accuracy is :",sum(Utrain_e3)/len(Utrain_e3))
	print("Mean of Validation Accuracy is",sum(Utest_e3)/len(Utest_e3))


	train_e3=[]
	test_e3=[]
	Utrain_e3=[]
	Utest_e3=[]
	losss_t=[]
	losss_tra=[]
	for i in range(5):
	  train_sample,train_label,test_sample,test_label=U_kfold(i,samples,labels)
	  m = len(train_sample)
	  y_predicted_test=np.empty(len(test_sample))
	  y_predicted_train=np.empty(len(train_sample))
	  clf=LogisticRegression(multi_class='ovr',max_iter=1000)
	  clf=clf.fit(train_sample,train_label)           #max=.521
	  y_predicted_test=clf.predict(test_sample)
	  y_predicted_train=clf.predict(train_sample)
	  ans1=U_acc(y_predicted_test,test_label)
	  ans2=U_acc(y_predicted_train,train_label)


	  
	  train_e3.append(accuracy_score(y_predicted_train, train_label))
	  test_e3.append(accuracy_score(y_predicted_test, test_label))
	  Utest_e3.append(ans1)
	  Utrain_e3.append(ans2)

	dd3=[x for x in range(1,6)]
	print("ONE VS Rest Using Sklearn ")
	df3=pd.DataFrame({'K-Fold':dd3,
	                 'Training Accuracy(sklearn)':train_e3 ,
	                 "Validation Accuracy(sklearn)":test_e3,
	                  'Training Accracy(Custom)':Utrain_e3,
	                 'Validation Accuracy(Custom)' : Utest_e3   
	})
	print(tabulate(df3, headers='keys', tablefmt='psql',showindex='never'))


	print("Mean of Training Accuracy is :",sum(Utrain_e3)/len(Utrain_e3))
	print("Mean of Validation Accuracy is",sum(Utest_e3)/len(Utest_e3))





  

if __name__ == "__main__":
    print("PhD19006")
    Q1_2()
    Q1_3()
    Q1_4()
    Q2_1()
    Q2_b_c()
    Q2_d()
    Q2_e()
    Q3_a()
    Q3_b()
    Q3_c()
    Q3_d()



  