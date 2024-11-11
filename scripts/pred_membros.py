import sklearn.linear_model as slm 
import pandas as pd
import joblib

model = joblib.load(r'C:models\model_2.joblib')

xtest = pd.read_csv(r'C:data\clean\xtest_membro.csv')
ytest = pd.read_csv(r'C:data\clean\ytest_membro.csv')

ypred = model.predict(xtest)
