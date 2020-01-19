import numpy as np
from sklearn import svm


def calc_f1_score_SVC(y_predicted, y_ground_truth):
    """
    We use this function to fine tune the SVM parameter C.
    A better f1-score means both good model performs good on both training set and validation set.

    - parameters:
        y_predicted: 1 dim numpy array.
        y_ground truth: 1 dim numpy array.

    - returns:
        f1_score: Float.
    """   
    y_predicted = y_predicted
    y_ground_truth = y_ground_truth
    p, n, tp, fn, fp, tn, correct = 0, 0, 0, 0, 0, 0, 0
    num_examples = len(y_predicted)
    
    for i in range(num_examples):
        if y_ground_truth[i] == 1:
            p += 1
            if y_predicted[i] == 1:
                tp += 1
            elif y_predicted[i] == -1:
                fn += 1
        elif y_ground_truth[i] == -1:
            n += 1
            if y_predicted[i] == 1:
                fp += 1
            elif y_predicted[i] == -1:
                tn += 1

    correct = tp + tn
    precision = tp / (tp + fp + .00001)
    recall = tp / (tp + fn + .00001)
    f1_score = (2 * precision * recall) / (precision + recall + .00001)
    accuracy = correct / num_examples
    return f1_score



def calc_err_rate_SVR(y_predicted, y_ground_truth):
    """
    We don't want to use MSE, because it's seeing big number and small number errors as the same weight.
    Instead we sum up the error rate of all examples.

    - parameters:
        y_predicted: 1 dim numpy array.
        y_ground_truth: 1 dim numpy array.

    - returns:
        error: Float.
    """
    num_examples = len(y_predicted)
    error = 0.0
    diff = np.abs(y_predicted - y_ground_truth)
    error_rate = diff / y_ground_truth
    error_power = np.power(error_rate, 1)
    error = np.sum(error_power) / num_examples
    return error





def run_SVC(X_train, X_val, y_train, y_val, kernel="rbf", log=True):
    """
    - Use sklearn.svm.SVC classifier.
    - See official scikit documents for details.
    """
    clf = None
    best_clf = None
    best_f1_score, best_f1_train, best_f1_val = -1, -1, -1
    exps = 2 ** np.arange(0, 22, 1)
    #print(exps)
    if kernel == "rbf":
        for C in 0.01 * exps:
            for gamma in 0.0000001 * exps:
                clf = svm.SVC(C=C, degree=3, kernel='rbf', gamma=gamma)
                clf.fit(X_train, y_train)
                # Fine tune C and gamma
                y_predict_val = clf.predict(X_val)
                y_predict_train = clf.predict(X_train)
                f1_score_train = calc_f1_score_SVC(y_predict_train, y_train)
                f1_score_val = calc_f1_score_SVC(y_predict_val, y_val)
                f1_score = f1_score_val# + 2 * (f1_score_train * f1_score_val) / (f1_score_train + f1_score_val + 0.0001) + f1_score_train
                
                if f1_score > best_f1_score:
                    #print(f1_score)
                    best_clf = clf
                    best_f1_score = f1_score
                    best_f1_train = f1_score_train
                    best_f1_val = f1_score_val

    if kernel == "linear":
        for C in [1, 2]:#0.001 * exps:
            clf = svm.SVC(C=C, kernel='linear')
            clf.fit(X_train, y_train)

            y_predict_train = clf.predict(X_train)
            f1_score_train = calc_f1_score_SVC(y_predict_train, y_train)

            y_predict_val = clf.predict(X_val)
            f1_score_val = calc_f1_score_SVC(y_predict_val, y_val)

            f1_score =  f1_score_val# + 2 * (f1_score_train * f1_score_val) / (f1_score_train + f1_score_val + 0.0001) + f1_score_train 
            #print(f1_score, f1_score_train, f1_score_val)
            if f1_score > best_f1_score:
                #print(f1_score)
                best_clf = clf
                best_f1_score = f1_score
                best_f1_train = f1_score_train
                best_f1_val = f1_score_val

    print(best_clf)
    print("Model best f1_score: %.4f, f1_score_training: %.4f, f1_score_val: %.4f" % (best_f1_score, best_f1_train, best_f1_val))
    if log:
        with open('SVM_Log.txt', 'a') as log_file:
            for i in range(2):
                print('-----------------------------------------------------------------------------------------', file=log_file)
            print(best_clf, file=log_file)
            print("Model best f1_score: %.4f, f1_score_training: %.4f, f1_score_val: %.4f" % (best_f1_score, best_f1_train, best_f1_val), file=log_file)
    return best_clf


def run_SVR(X_train, X_val, y_train, y_val, kernel="rbf", log=True):
    """
    - Use sklearn.svm.SVR regressor.
    - See official scikit documents for details.
    """
    clf = None
    best_clf = None
    best_error, best_error_train, best_error_val = 9999, 9999, 9999
    exps = 2 ** np.arange(0, 22, 1)
    if kernel == "rbf":
        for C in 0.1 * exps:
            for gamma in 0.000001 * exps:
                clf = svm.SVR(C=C, kernel='rbf', gamma=gamma)
                clf.fit(X_train, y_train)

                y_predict_val = clf.predict(X_val)
                y_predict_train = clf.predict(X_train)
                error_train = calc_err_rate_SVR(y_predict_train, y_train)
                error_val = calc_err_rate_SVR(y_predict_val, y_val)
                error = error_val# + error_train * 0.2
                
                if error < best_error:
                    #print(error)
                    best_clf = clf
                    best_error = error
                    best_error_train = error_train
                    best_error_val = error_val

    if kernel == "linear":
        for C in 0.001 * exps:
            clf = svm.SVR(C=C, kernel='linear')
            clf.fit(X_train, y_train)

            y_predict_val = clf.predict(X_val)
            y_predict_train = clf.predict(X_train)
            error_train = calc_err_rate_SVR(y_predict_train, y_train)
            error_val = calc_err_rate_SVR(y_predict_val, y_val)
            error = error_val# + error_train * 0.2
            
            if error < best_error:
                #print(error)
                best_clf = clf
                best_error = error
                best_error_train = error_train
                best_error_val = error_val

    print(best_clf)
    print("Model best error: %.4f, error_training: %.4f, error_val: %.4f" % (best_error, best_error_train, best_error_val))

    if log:
        with open('SVM_Log.txt', 'a') as log_file:
            for i in range(2):
                print('-----------------------------------------------------------------------------------------', file=log_file)
            print(best_clf, file=log_file)
            print("Model best error: %.4f, error_training: %.4f, error_val: %.4f" % (best_error, best_error_train, best_error_val), file=log_file)
    return best_clf


def run_SVC_k_fold(X_train, y_train, kernel="rbf", log=True, dont_show=False):
    """
    - Use sklearn.svm.SVC classifier.
    - See official scikit documents for details.
    """
    clf = None
    best_clf = None
    best_f1_score, best_f1_train, best_f1_val = -1, -1, -1
    exps = 2 ** np.arange(0, 22, 1)
    #print(exps)
    if kernel == "rbf":
        for C in 0.01 * exps:
            for gamma in 0.0000001 * exps:
                clf = svm.SVC(C=C, degree=3, kernel='rbf', gamma=gamma)
                clf.fit(X_train, y_train)
                # Fine tune C and gamma
                #y_predict_val = clf.predict(X_val)
                y_predict_train = clf.predict(X_train)
                f1_score_train = calc_f1_score_SVC(y_predict_train, y_train)
                #f1_score_val = calc_f1_score_SVC(y_predict_val, y_val)
                f1_score = f1_score_train# + 2 * (f1_score_train * f1_score_val) / (f1_score_train + f1_score_val + 0.0001) + f1_score_train
                
                if f1_score > best_f1_score:
                    #print(f1_score)
                    best_clf = clf
                    best_f1_score = f1_score
                    best_f1_train = f1_score_train
                    #best_f1_val = f1_score_val

    if kernel == "linear":
        for C in [1, 2]:#0.001 * exps:
            clf = svm.SVC(C=C, kernel='linear', probability=True)
            clf.fit(X_train, y_train)

            y_predict_train = clf.predict(X_train)
            f1_score_train = calc_f1_score_SVC(y_predict_train, y_train)

            #y_predict_val = clf.predict(X_val)
            #f1_score_val = calc_f1_score_SVC(y_predict_val, y_val)

            f1_score =  f1_score_train# + 2 * (f1_score_train * f1_score_val) / (f1_score_train + f1_score_val + 0.0001) + f1_score_train 
            #print(f1_score, f1_score_train, f1_score_val)
            if f1_score > best_f1_score:
                #print(f1_score)
                best_clf = clf
                best_f1_score = f1_score
                best_f1_train = f1_score_train
                #best_f1_val = f1_score_val

    if not dont_show:
        print(best_clf)
        print("Model best f1_score: %.4f, f1_score_training: %.4f" % (best_f1_score, best_f1_train))
    if log:
        with open('SVM_Log.txt', 'a') as log_file:
            for i in range(2):
                print('-----------------------------------------------------------------------------------------', file=log_file)
            print(best_clf, file=log_file)
            print("Model best f1_score: %.4f, f1_score_training: %.4f" % (best_f1_score, best_f1_train), file=log_file)
    return best_clf


def run_SVR_k_fold(X_train, y_train, kernel="rbf", log=True, dont_show=False):
    """
    - Use sklearn.svm.SVR regressor.
    - See official scikit documents for details.
    """
    clf = None
    best_clf = None
    best_error, best_error_train, best_error_val = 9999, 9999, 9999
    exps = 2 ** np.arange(0, 22, 1)
    if kernel == "rbf":
        for C in [1, 2]:
            for gamma in 0.000001 * exps:
                clf = svm.SVR(C=C, kernel='rbf', gamma='auto')
                clf.fit(X_train, y_train)

                #y_predict_val = clf.predict(X_val)
                y_predict_train = clf.predict(X_train)
                error_train = calc_err_rate_SVR(y_predict_train, y_train)
                #error_val = calc_err_rate_SVR(y_predict_val, y_val)
                error = error_train# + error_train * 0.2
                
                if error < best_error:
                    #print(error)
                    best_clf = clf
                    best_error = error
                    best_error_train = error_train
                    #best_error_val = error_val

    if kernel == "linear":
        for C in [1, 2]:
            clf = svm.SVR(C=C, kernel='linear')
            clf.fit(X_train, y_train)

            #y_predict_val = clf.predict(X_val)
            y_predict_train = clf.predict(X_train)
            error_train = calc_err_rate_SVR(y_predict_train, y_train)
            #error_val = calc_err_rate_SVR(y_predict_val, y_val)
            error = error_train# + error_train * 0.2
            
            if error < best_error:
                #print(error)
                best_clf = clf
                best_error = error
                best_error_train = error_train
                #best_error_val = error_val

    if not dont_show:
        print(best_clf)
        print("Model best error: %.4f, error_training: %.4f" % (best_error, best_error_train))

    if log:
        with open('SVM_Log.txt', 'a') as log_file:
            for i in range(2):
                print('-----------------------------------------------------------------------------------------', file=log_file)
            print(best_clf, file=log_file)
            print("Model best error: %.4f, error_training: %.4f" % (best_error, best_error_train), file=log_file)
    return best_clf
