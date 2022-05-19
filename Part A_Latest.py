#Importing Libraries
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_curve,auc, confusion_matrix, classification_report,roc_auc_score 
import seaborn as sns

### Importing The datasets
PRT_1_N= pd.read_csv("C:/Desktop/AIML/Project 2/Part1+-+Normal.csv")
PRT_1_H= pd.read_csv("C:/Desktop/AIML/Project 2/Part1+-+Type_H.csv")
PRT_1_S= pd.read_csv("C:/Desktop/AIML/Project 2/Part1+-+Type_S.csv")
### Checking first 5 rows
PRT_1_H.head(5)
PRT_1_N.head(5)
PRT_1_S.head(5)

### Checking Shape and cloumns of datasets and comparing columns
col_H=PRT_1_H.columns
PRT_1_H.shape
col_N=PRT_1_N.columns
PRT_1_N.shape
col_S=PRT_1_S.columns
PRT_1_S.shape
# Finding out Datatypes
PRT_1_N.dtypes
PRT_1_H.dtypes
PRT_1_S.dtypes
##### Variation In "Class" Col.
PRT_1_N.Class.describe()
PRT_1_N.Class.value_counts()
PRT_1_H.Class.describe()
PRT_1_H.Class.value_counts()
PRT_1_S.Class.describe()
PRT_1_S.Class.value_counts()

### Unifying variations in Class column
PRT_1_N.loc[PRT_1_N["Class"]=="Nrmal","Class"]="Normal"
PRT_1_N.Class.value_counts()
PRT_1_H.loc[PRT_1_H["Class"]=="type_h","Class"]="Type_H"
PRT_1_H.Class.value_counts()
PRT_1_S.loc[PRT_1_S["Class"]=="tp_s","Class"]="Type_S"
PRT_1_S.Class.value_counts()

###### Merging the 3 dataframes
Prt_1=PRT_1_N.append( PRT_1_H ,ignore_index=True)
Prt_1_Med=Prt_1.append(PRT_1_S,ignore_index=True)
Prt_1_Med.shape

## Taking 5 random samples 
Prt_1_Med.sample(5)

## Featurewise percentage of Null values

Percent_missing= Prt_1_Med.isnull().sum()*100/len(Prt_1_Med)
Percent_missing_df=pd.DataFrame({'column_name':Prt_1_Med.columns,
                                 'Percent_missing':Percent_missing})

## Creating 5 point Summary
Prt_1_Med.describe()

## Creating Heatmap

## Creating subset of the DF without Class column
Prt_1_Med_1=Prt_1_Med.drop(["Class"],axis=1)
## Check Correlation
x=Prt_1_Med_1.corr()
### Creating the Heatmap
sns.heatmap(Prt_1_Med_1,cmap="Blues")

## Creating pairplt

sns.pairplot(Prt_1_Med, hue="Class")

## Creating Joint Plot

sns.jointplot(x="P_incidence",y="S_slope",kind="scatter",data=Prt_1_Med)

## Creating Box Plot of all the features

Prt_1_Med_1.plot(kind='box')

## Labelling and featuring
Y=Prt_1_Med.iloc[:,6]
X=Prt_1_Med.iloc[:,0:6]
X.describe().round(3) # Summary Statistics

### Train and Test Split
x_train,x_test,y_train,y_test=train_test_split(X,Y,stratify=Y,test_size=0.2,random_state=0)

# feature scalling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# creating a KNN object
clf=KNeighborsClassifier(weights="distance",n_neighbors=5) #Default K=5

# train the model
clf=clf.fit(x_train,y_train)

# accuracy of the test set
print("Test set score:" , clf.score(x_test,y_test))

# predicted probability of y
probs=clf.predict_proba(x_test)[:,1] 
probs.round(3)[:10]

# predicted class of y
y_pred=clf.predict(x_test)
y_pred
Z=np.unique(y_pred,return_counts=True)
print(Z)
# Cofusion matrix
mat1=confusion_matrix(y_test,y_pred)

mat1_df=pd.DataFrame(mat1,
                     index=["Normal","Type_H","Type_S"],columns=["Normal","Type_H","Type_S"])

## Plotting the Confusion Matrix
sns.heatmap(mat1_df,annot=True,cbar=False,fmt="d")
plt.title("Confusion Matrix")
plt.ylabel("True class")
plt.xlabel("predicted class")

## Checking Performance of model with classification report
print(classification_report(y_test, y_pred))

##Tweaking Hyperparamets for Performance improvements of KNN

#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]

#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

#Create new KNN object
KNN_2 = KNeighborsClassifier()

#Use GridSearch
clf_2 = GridSearchCV(KNN_2, hyperparameters, cv=10)

#Fit the model
best_model = clf_2.fit(x_train,y_train)

#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

## Creating KNN object with n_neighbor as 13 and metric Manhattan

clf_3=KNeighborsClassifier(weights="distance",p=1, n_neighbors=13) #Default K=5

# train the model
clf_3=clf_3.fit(x_train,y_train)

# accuracy of the test set
print("Test set score:" , clf_3.score(x_test,y_test))

# predicted probability of y
probs_3=clf_3.predict_proba(x_test)[:,1] 
probs_3.round(3)[:10]

# predicted class of y
y_pred3=clf_3.predict(x_test)
y_pred3
Z3=np.unique(y_pred3,return_counts=True)
print(Z3)

# Cofusion matrix
mat3=confusion_matrix(y_test,y_pred3)

mat3_df=pd.DataFrame(mat3,
                     index=["Normal","Type_H","Type_S"],columns=["Normal","Type_H","Type_S"])

## Plotting the Confusion Matrix
sns.heatmap(mat3_df,annot=True,cbar=False,fmt="d")
plt.title("Confusion Matrix")
plt.ylabel("True class")
plt.xlabel("predicted class")

## Checking Performance of model with classification report
print(classification_report(y_test, y_pred3))










