import sklearn.model_selection as sms
import sklearn.linear_model as slm
import pandas as pd
import joblib
import os

data = pd.read_csv("C:data\clean\clean_membros.csv")

data = pd.get_dummies(data, columns=['p_plano'], drop_first=True)

X = data.drop(columns=['p_chave_membro', 'renovado'])
y = data['renovado']

xtrain, xtest, ytrain, ytest = sms.train_test_split(X, y, test_size=0.3, random_state=8)

xtest.to_csv(r'C:data\clean\xtest_membro.csv', index=False)
ytest.to_csv(r'C:data\clean\ytest_membro.csv', index=False)

model_2 = slm.LogisticRegression()
model_2.fit(xtrain, ytrain)

joblib.dump(model_2, r'C:models\model_2.joblib')
