#make sure you install opencv2, mediapipe and scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('coords.csv')

#print(df[df['class']=="Happy"])#to filter data in pandas

X=df.drop('class',axis=1)#features
Y=df['class']#target value

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1234)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler #standardizes the data so one feature over-shadows other

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier())
}
#print(list(pipelines.values())[2])

fit_models={} #blank dictionary
for algorithm,pipeline in pipelines.items():
    model=pipeline.fit(X_train.values, Y_train.values)#added Xtrain.values
    fit_models[algorithm]=model
print("Status:Traning has Completed")

#print(fit_models)
#print(fit_models['rf'].predict(X_test))

from sklearn.metrics import accuracy_score# accuracy metrics

for algorithm, model in fit_models.items():
    #predictions
    Y_hat=model.predict(X_test.values)#added .values
    print(algorithm,accuracy_score(Y_test,Y_hat))

#check
#print(list(fit_models['rf'].predict(X_test)))
#print(list(Y_test))

import pickle #save training model
with open('handTraining.pkl', 'wb') as f:#
    pickle.dump(fit_models['rf'],f)
#Random Forest accuracy was 99.47