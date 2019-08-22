import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv(r"C:\Users\deepa\Desktop\data.csv")
models=["LogisticRegression","KNeighborsClassifier","SVC_linear      ","SVC_BRF         ","GaussianNB       ","DecisionTreeClassifier","RandomForestClassifier"]



'''partitioning datasets'''

x=data.iloc[:,2:32].values
y=data.iloc[:,1].values
accuracy=[]
'''visualising dataset'''
print(x.shape)
#print(data.head())


'''missing null points'''
data.isnull().sum()
data.isna().sum()

'''encoding to easier format'''
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

'''spliting dataset'''
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

'''scaling all the values between some limit'''
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





'''linear regression model'''
classifierL = LogisticRegression(random_state = 0)
classifierL.fit(X_train, Y_train)


'''prediction using trained model'''
Y_predL = classifierL.predict(X_test)

'''confussion matrix for accuracy calculation'''
cmL = confusion_matrix(Y_test, Y_predL)

#print(cmL)

aL=[((cmL[0][0]+cmL[1][1])/(cmL[0][0]+cmL[1][1]+cmL[0][1]+cmL[1][0]))*100]
accuracy.append(aL)






classifierK = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierK.fit(X_train, Y_train)


'''prediction using trained model'''
Y_predK = classifierK.predict(X_test)

'''confussion matrix for accuracy calculation'''
cmK = confusion_matrix(Y_test, Y_predK)

#print(cmK)

aK=[((cmK[0][0]+cmK[1][1])/(cmK[0][0]+cmK[1][1]+cmK[0][1]+cmK[1][0]))*100]
accuracy.append(aK)






classifierS = SVC(kernel = 'linear', random_state = 0)
classifierS.fit(X_train, Y_train)


'''prediction using trained model'''
Y_predS = classifierS.predict(X_test)

'''confussion matrix for accuracy calculation'''
cmS = confusion_matrix(Y_test, Y_predS)

#print(cmS)

aS=[((cmS[0][0]+cmS[1][1])/(cmS[0][0]+cmS[1][1]+cmS[0][1]+cmS[1][0]))*100]
accuracy.append(aS)




classifierSRBF =  SVC(kernel = 'rbf', random_state = 0)
classifierSRBF.fit(X_train, Y_train)


'''prediction using trained model'''
Y_predSRBF = classifierSRBF.predict(X_test)

'''confussion matrix for accuracy calculation'''
cmSRBF = confusion_matrix(Y_test, Y_predSRBF)

#print(cmSRBF)

aSRBF=[((cmSRBF[0][0]+cmSRBF[1][1])/(cmSRBF[0][0]+cmSRBF[1][1]+cmSRBF[0][1]+cmSRBF[1][0]))*100]
accuracy.append(aSRBF)




classifierG =  GaussianNB()
classifierG.fit(X_train, Y_train)


'''prediction using trained model'''
Y_predG = classifierG.predict(X_test)

'''confussion matrix for accuracy calculation'''
cmG = confusion_matrix(Y_test, Y_predG)

#print(cmG)

aG=[((cmG[0][0]+cmG[1][1])/(cmG[0][0]+cmG[1][1]+cmG[0][1]+cmG[1][0]))*100]
accuracy.append(aG)




classifierD =  classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierD.fit(X_train, Y_train)


'''prediction using trained model'''
Y_predD = classifierD.predict(X_test)

'''confussion matrix for accuracy calculation'''
cmD = confusion_matrix(Y_test, Y_predD)

#print(cmD)

aD=[((cmD[0][0]+cmD[1][1])/(cmD[0][0]+cmD[1][1]+cmD[0][1]+cmD[1][0]))*100]
accuracy.append(aD)






classifierR = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierR.fit(X_train, Y_train)


'''prediction using trained model'''
Y_predR = classifierR.predict(X_test)

'''confussion matrix for accuracy calculation'''
cmR = confusion_matrix(Y_test, Y_predR)

#print(cmR)

aR=[((cmR[0][0]+cmR[1][1])/(cmR[0][0]+cmR[1][1]+cmR[0][1]+cmR[1][0]))*100]
accuracy.append(aR)

'''matx[0]]
for a in range(30):
	mat.append(x[a+1])
#print(mat)
#print(mat[0][0])
print(mat[][0])
'''
vari=list(data.columns)
#print(vari)

for i in range(6):
	temp=list(data.iloc[:,i+2].values)
	#print(temp)
	#f=plt.figure(1)
	#print(temp.shape())
	plt.subplot(3,2,i+1)
	plt.scatter(temp,y)
	plt.xlabel(str(vari[i+2]))
	plt.ylabel("severity")



'''for i in range(6):
	temp=list(data.iloc[:,i+2+6].values)
	#print(temp)
	g =plt.figure(2)
	#print(temp.shape())
	plt.subplot(3,2,i+1)
	plt.scatter(temp,y)
	plt.xlabel(str(vari[i+2+6]))
	plt.ylabel("severity")'''
	
plt.show()
#g.show()

#input()





f=open("accuracy.txt","w+")

for i in range(len(accuracy)):
	f.write("%s\t"%str(models[i]))
	f.write("%s\n"%str(accuracy[i]))
f.close()

print(i)
print("accuracy:",accuracy)