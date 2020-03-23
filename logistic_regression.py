import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def logistic_regression(data,labels, data_test, labels_test):

    logistic_regressor = LogisticRegression(multi_class='multinomial', max_iter=5000)
    logistic_regressor.fit(data, labels)
    y_train_soft = logistic_regressor.predict_proba(data_test)
    np.save('y_train_soft.npy', y_train_soft)

    print('the score is ',logistic_regressor.score(data_test, labels_test))
    print('the precision is ',precision_score(labels_test, np.argmax(y_train_soft,axis=-1), average='weighted'))
    print('the recall is ',recall_score(labels_test, np.argmax(y_train_soft,axis=-1), average='weighted'))

    return logistic_regressor

    #np.save('y_train_soft.npy', y_train_soft)

    
