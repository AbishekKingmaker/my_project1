from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import numpy as np
import joblib



### loading data
data_fin = np.load('vector_data.npy', allow_pickle=True)
lab_fin = np.load('lab_fin.npy',allow_pickle=True)
### Splitting train test percentage
X_train, X_test, Y_train, Y_test = train_test_split(data_fin, lab_fin, test_size=0.3, random_state=42)

## set hyper parameter
mdl = LogisticRegression(solver='saga',penalty='none',max_iter=100) ## newton-cg,lbfgs,sag,saga
mdl.fit(X_train,Y_train)
joblib.dump(mdl, 'log_reg.pkl')
predictions = mdl.predict(X_test)
evl_scores = cross_validate(mdl, data_fin, lab_fin, cv=5,
                        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

cross_val = np.array(
    [evl_scores['test_accuracy'], evl_scores['test_precision_macro'], evl_scores['test_recall_macro'],
     evl_scores['test_f1_macro']])

perf_score = np.array([accuracy_score(Y_test, predictions), precision_score(Y_test, predictions, average='weighted'),
               recall_score(Y_test, predictions, average='weighted'),
               f1_score(Y_test, predictions, average='weighted')])

print("the evaluated cross-validation scores is",cross_val)
print ("the evalatue performance metrics result is",perf_score)

print(("hi"))
