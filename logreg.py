import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np 


#prediciting whether whether a person will have insurance or not based on his age
a = pd.read_csv(r"https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/insurance_data.csv")
dfs = pd.DataFrame(a)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dfs[['age']],dfs.bought_insuransce,train_size=0.8,random_state=100)
from sklearn.linear_model import LogisticRegression	
model = LogisticRegression()
model.fit(X_train, y_train)
cmatrix = confusion_matrix(y_test, model.predict(X_test))

class FunctionInfo:
	
	def __init__(self):
		
		# Function information dictionary - { function name: [ list of arguments, return type, funtion description ] }
		
		self.func_dict =   {	
								'TKPY_logistic_main': [['float'], ['array'], ['reg Model']],
								'TKPY_logistic_intercept' : [['float'],['float'],['intercept on Y-axis']],
								'TKPY_logistic_coef' : [['float'],['float'],['Coefficient for the equation']],
								'TKPY_logistic_acc' :[['float'],['float'],['accuracy of the given model']],
								'TKPY_logistic_predict': [['float'],['float'],['predicted values for any value']],
								'TKPY_logistic_confusionMatrix' : [['int'],['matrix'],["confusion_matrix"]]
							}
	def get_func_dict(self):
		return self.func_dict



def TKPY_logistic_predict(age):
	pred =model.predict([[age]])
	return float(pred)

def TKPY_logistic_intercept(dummy):
	i= model.intercept_
	return(float(i))

def TKPY_logistic_acc(dummy):
	Acc = model.score(X_test,y_test)
	return float(Acc)

def TKPY_logistic_coef(dummy):
	o= model.coef_	
	return(float(o))


def TKPY_logistic_confusionMatrix(a):
	cmatrix = confusion_matrix(y_test,model.predict(X_test))
	cmatrix = cmatrix + a - a
	return np.array(cmatrix)


def TKPY_logistic_main(a):
	k=TKPY_logistic_intercept(model)
	l=TKPY_logistic_acc(model)
	s=TKPY_logistic_coef(model)
	return (np.array([ l,k,s]))


print(TKPY_logistic_coef(10))