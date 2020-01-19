import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn import preprocessing
from sklearn import svm



def observe_prediction_SVC(clf, X, y, patient_id, dont_show=True, log=True, setname=''):
    """
    - With SVM classifier, we predict sarcopenia, and see how it works.
    - Parameters:
        clf: sklearn.svc classifier object.
        X: 2-dim numpy array. Datasets.
        y: 1-dim numpy array. Ground truth of sarcopenia.
        patient_id: 1-dim numpy array. eg: [2, 5, 102...]
        dont_show: Don't print on screen.
    - Return:
        None. 
        Only print false predictions on screen.
    """
    
    y_predict = clf.predict(X)
    assert(len(y_predict) == len(y) == len(patient_id))
    patient_id = patient_id
    correctness = (y_predict == y)
    num_examples = len(y)

    if not dont_show:
        print('\nObserving %s Set:' % setname)
        if False in correctness:
            for i in range(num_examples):
                if correctness[i] == False:
                    print("Truth: %2d, Predicted: %2d, Patient id: %3d" %(y[i], y_predict[i], patient_id[i]))
        else:
            print("All correct.")
    if log:
        with open('SVM_Log.txt', 'a') as log_file:
            print('\nObserving %s Set:' % setname, file=log_file)
            if False in correctness:
                for i in range(num_examples):
                    if correctness[i] == False:
                        print("Truth: %2d, Predicted: %2d, Patient id: %3d" %(y[i], y_predict[i], patient_id[i]), file=log_file)
            else:
                print("All correct.", file=log_file)
    return


def observe_prediction_asm_SVR(clf, X, y, gender, height_squared, sarcopenia, patient_id, dont_show=True, log=True, setname=''):
    
    h2 = height_squared
    patient_id = patient_id
    sarcopenia_ground_truth = sarcopenia
    num_examples = len(y)

    y_predict = clf.predict(X) # asm
    diff = (y_predict - y) / y
    diff_percent = diff * 100
    diff_thresh = 10.0
    
    asm_over_h2 = y_predict / h2
    sarcopenia_predicted = []
    for i in range(num_examples):
        if((asm_over_h2[i] - 7.0) < 0.0001 and gender[i] == 1):
            sarcopenia_predicted.append(1)
        elif((asm_over_h2[i] - 5.4) < 0.0001 and gender[i] == 2):
            sarcopenia_predicted.append(1)
        else:
            sarcopenia_predicted.append(-1)

    correctness = (sarcopenia_ground_truth == sarcopenia_predicted)
    
    if not dont_show:
        print('\n%s Set:' % setname)
        for i in range(num_examples):
            if(np.abs(diff_percent[i]) > diff_thresh or correctness[i]==0):
                print("Truth: %.2f, Pred: %.2f, ASM/h2: %.2f, Error: %6.2f%%, Gender: %2d, GT: %2d, Pred: %2d, Correct: %2d, Patient_id: %3d" %
                    (y[i], y_predict[i], asm_over_h2[i], diff_percent[i], gender[i], sarcopenia_ground_truth[i], sarcopenia_predicted[i], correctness[i], patient_id[i]))
    if log:
        with open('SVM_Log.txt', 'a') as log_file:
            print('\nObserving %s Set:' % setname, file=log_file)
            for i in range(num_examples):
                if(np.abs(diff_percent[i]) > diff_thresh or correctness[i]==0):
                    print("Truth: %.2f, Pred: %.2f, ASM/h2: %.2f, Error: %6.2f%%, Gender: %2d, GT: %2d, Pred: %2d, Correct: %2d, Patient_id: %3d" % 
                        (y[i], y_predict[i], asm_over_h2[i], diff_percent[i], gender[i], sarcopenia_ground_truth[i], sarcopenia_predicted[i], correctness[i], patient_id[i]), file=log_file)

    return diff




def observe_prediction_asm_h2_SVR(clf, X, y, gender, sarcopenia, patient_id, dont_show=True, log=True, setname=''):
    
    y_predict = clf.predict(X)
    sarcopenia_ground_truth = sarcopenia
    sarcopenia_predicted = []
    diff = (y_predict - y) / y
    patient_id = patient_id
    diff_percent = diff * 100
    diff_thresh = 10.0
    num_examples = len(y)

    for i in range(num_examples):
        if((y_predict[i] - 7.0) < 0.0001 and gender[i] == 1):
            sarcopenia_predicted.append(1)
        elif((y_predict[i] - 5.4) < 0.0001 and gender[i] == 2):
            sarcopenia_predicted.append(1)
        else:
            sarcopenia_predicted.append(-1)
    correctness = (sarcopenia_ground_truth == sarcopenia_predicted)

    if not dont_show:
        print('\n%s Set:' % setname)
        for i in range(num_examples):
            if(np.abs(diff_percent[i])>diff_thresh or correctness[i]==0):
                print("Truth: %.2f, Predicted: %.2f, Error: %6.2f%%, Gender: %2d, GT: %2d, Pred: %2d, Correct: %2d, Patient_id: %3d" %
                    (y[i], y_predict[i], diff_percent[i], gender[i], sarcopenia_ground_truth[i], sarcopenia_predicted[i], correctness[i], patient_id[i]))
    if log:
        with open('SVM_Log.txt', 'a') as log_file:
            print('\nObserving %s Set:' % setname, file=log_file)
            for i in range(num_examples):
                if(np.abs(diff_percent[i])>diff_thresh or correctness[i]==0):
                    print("Truth: %.2f, Predicted: %.2f, Error: %6.2f%%, Gender: %2d, GT: %2d, Pred: %2d, Correct: %2d, Patient_id: %3d" %
                        (y[i], y_predict[i], diff_percent[i], gender[i], sarcopenia_ground_truth[i], sarcopenia_predicted[i], correctness[i], patient_id[i]), file=log_file)

    return diff


def eval_sarcopenia_asm(clf, X, gender, height_squared, sarcopenia_ground_truth):
    asm_predicted = clf.predict(X)
    asm = asm_predicted
    gender = gender
    h2 = height_squared
    ground_truth = sarcopenia_ground_truth
    asm_over_h2 = asm / h2
    sarcopenia = []
    num_patients = len(asm_predicted)
    assert len(gender) == num_patients
    assert len(ground_truth) == num_patients
    assert len(h2) == num_patients

    for i in range(num_patients):
        if((asm_over_h2[i] - 7.0) < 0.0001 and gender[i] == 1):
            sarcopenia.append(1)
        elif((asm_over_h2[i] - 5.4) < 0.0001 and gender[i] == 2):
            sarcopenia.append(1)
        else:
            sarcopenia.append(-1)
    return sarcopenia


def eval_sarcopenia_asm_h2(clf, X, gender, sarcopenia_ground_truth):
    asm_h2_predicted = clf.predict(X)
    gender = gender
    ground_truth = sarcopenia_ground_truth

    sarcopenia_predicted = []
    num_patients = len(asm_h2_predicted)
    assert len(gender) == num_patients
    assert len(ground_truth) == num_patients
    

    for i in range(num_patients):
        if((asm_h2_predicted[i] - 7.0) < 0.0001 and gender[i] == 1):
            sarcopenia_predicted.append(1)
        elif((asm_h2_predicted[i] - 5.4) < 0.0001 and gender[i] == 2):
            sarcopenia_predicted.append(1)
        else:
            sarcopenia_predicted.append(-1)
    return sarcopenia_predicted





def eval_sarcopenia_asm_h2_nn(asm_h2_predicted, gender, sarcopenia_ground_truth):
    asm_h2_predicted = asm_h2_predicted
    gender = gender
    ground_truth = sarcopenia_ground_truth

    sarcopenia_predicted = []
    num_patients = len(asm_h2_predicted)
    assert len(gender) == num_patients
    assert len(ground_truth) == num_patients
    

    for i in range(num_patients):
        if((asm_h2_predicted[i] - 7.0) < 0.0001 and gender[i] == 1):
            sarcopenia_predicted.append(1)
        elif((asm_h2_predicted[i] - 5.4) < 0.0001 and gender[i] == 2):
            sarcopenia_predicted.append(1)
        else:
            sarcopenia_predicted.append(-1)
    return sarcopenia_predicted


def eval_sarcopenia_asm_nn(asm_predicted, gender, height_squared, sarcopenia_ground_truth):
    asm = asm_predicted
    gender = gender
    h2 = height_squared
    ground_truth = sarcopenia_ground_truth
    asm_over_h2 = asm / h2
    sarcopenia = []
    num_patients = len(asm_predicted)
    assert len(gender) == num_patients
    assert len(ground_truth) == num_patients
    assert len(h2) == num_patients

    for i in range(num_patients):
        if((asm_over_h2[i] - 7.0) < 0.0001 and gender[i] == 1):
            sarcopenia.append(1)
        elif((asm_over_h2[i] - 5.4) < 0.0001 and gender[i] == 2):
            sarcopenia.append(1)
        else:
            sarcopenia.append(-1)
    return sarcopenia

def observe_prediction_asm_h2_nn(asm_h2_predicted, asm_h2_ground_truth, gender, sarcopenia, patient_id, dont_show=True, log=True, setname=''):
    print('Observing %s Set:' % setname)
    #h2 = height_squared
    patient_id = patient_id.cpu().numpy()
    sarcopenia_ground_truth = sarcopenia.cpu().numpy()
    num_examples = len(asm_h2_ground_truth)

    y_predict = asm_h2_predicted # asm/h2
    y = asm_h2_ground_truth.cpu().numpy()
    diff = (y_predict - y) / y
    diff_percent = diff * 100
    diff_thresh = 10.0
    
    #asm_over_h2 = y_predict / h2
    sarcopenia_predicted = []
    for i in range(num_examples):
        if((y_predict[i] - 7.0) < 0.0001 and gender[i] == 1):
            sarcopenia_predicted.append(1)
        elif((y_predict[i] - 5.4) < 0.0001 and gender[i] == 2):
            sarcopenia_predicted.append(1)
        else:
            sarcopenia_predicted.append(-1)

    correctness = (sarcopenia_ground_truth == sarcopenia_predicted)
    
    if not dont_show:
        for i in range(num_examples):
            if not False in correctness:
                print('All correct.')
                break
            if(np.abs(diff_percent[i]) > diff_thresh or correctness[i]==0):
                print("Truth: %.2f, Pred: %.2f, Error: %6.2f%%, Gender: %2d, GT: %2d, Pred: %2d, Correct: %2d, Patient_id: %3d" %
                    (y[i], y_predict[i], diff_percent[i], gender[i], sarcopenia_ground_truth[i], sarcopenia_predicted[i], correctness[i], patient_id[i]))
    if log:
        with open('NN_Log.txt', 'a') as log_file:
            print('\nObserving %s Set:' % setname, file=log_file)
            for i in range(num_examples):
                if(np.abs(diff_percent[i]) > diff_thresh or correctness[i]==0): 
                    print("Truth: %.2f, Pred: %.2f, Error: %6.2f%%, Gender: %2d, GT: %2d, Pred: %2d, Correct: %2d, Patient_id: %3d" % 
                        (y[i], y_predict[i], diff_percent[i], gender[i], sarcopenia_ground_truth[i], sarcopenia_predicted[i], correctness[i], patient_id[i]), file=log_file)

    return diff

    



def eval_classifier(sarcopenia_predicted, sarcopenia_ground_truth, show_detail=True, log=True, setname=''):
    
    num_patients = len(sarcopenia_predicted)
    assert num_patients == len(sarcopenia_ground_truth)

    y_predicted = sarcopenia_predicted
    y_ground_truth = sarcopenia_ground_truth
    p, n, tp, fn, fp, tn, correct = 0, 0, 0, 0, 0, 0, 0

    for i in range(num_patients):
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
    specificity = tn / (tn + fp + .00001)
    f1_score = (2 * precision * recall) / (precision + recall + .00001)
    if show_detail:
        print("\nEvaluating %s set:" % setname)
        print("Positive: %d, Negative: %d" % (p, n))
        print("TP: %d, FP: %d, TN: %d, FN: %d" % (tp, fp, tn, fn))
        print("Correct: %d(%d), Precision: %.3f, Recall: %.3f, Specificity: %.3f, F1-Score: %.3f\n" % (correct, num_patients, precision, recall, specificity, f1_score))
    if log:
        with open('SVM_Log.txt', 'a') as log_file:
            print("Evaluating %s set:" % setname, file=log_file)
            print("Positive: %d, Negative: %d" % (p, n), file=log_file)
            print("TP: %d, FP: %d, TN: %d, FN: %d" % (tp, fp, tn, fn), file=log_file)
            print("Correct: %d(%d), Precision: %.3f, Recall: %.3f, Specificity: %.3f, F1-Score: %.3f" % (correct, num_patients, precision, recall, specificity, f1_score), file=log_file)
    return


def eval_sarcopenia_asm_h2_leave_one_out(asm_h2_predicted, gender, sarcopenia_ground_truth):
    asm_h2_predicted = asm_h2_predicted
    gender = gender
    ground_truth = sarcopenia_ground_truth

    sarcopenia_predicted = []
    num_patients = len(asm_h2_predicted)
    assert len(gender) == num_patients
    assert len(ground_truth) == num_patients
    

    for i in range(num_patients):
        if((asm_h2_predicted[i] - 7.0) < 0.0001 and gender[i] == 1):
            sarcopenia_predicted.append(1)
        elif((asm_h2_predicted[i] - 5.4) < 0.0001 and gender[i] == 2):
            sarcopenia_predicted.append(1)
        else:
            sarcopenia_predicted.append(-1)
    return sarcopenia_predicted

def eval_sarcopenia_asm_leave_one_out(asm_predicted, gender, height_squared, sarcopenia_ground_truth):
    asm_predicted = asm_predicted
    asm = asm_predicted
    gender = gender
    h2 = height_squared
    ground_truth = sarcopenia_ground_truth
    asm_over_h2 = asm / h2
    sarcopenia = []
    num_patients = len(asm_predicted)
    assert len(gender) == num_patients
    assert len(ground_truth) == num_patients
    assert len(h2) == num_patients

    for i in range(num_patients):
        if((asm_over_h2[i] - 7.0) < 0.0001 and gender[i] == 1):
            sarcopenia.append(1)
        elif((asm_over_h2[i] - 5.4) < 0.0001 and gender[i] == 2):
            sarcopenia.append(1)
        else:
            sarcopenia.append(-1)
    return sarcopenia



def eval_classifier_k_fold(sarcopenia_predicted, sarcopenia_ground_truth):
    num_patients = len(sarcopenia_predicted)
    assert num_patients == len(sarcopenia_ground_truth)

    y_predicted = sarcopenia_predicted
    y_ground_truth = sarcopenia_ground_truth
    p, n, tp, fn, fp, tn, correct = 0, 0, 0, 0, 0, 0, 0

    for i in range(num_patients):
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

    ppv = (tp) / (tp + fp + .00001)
    npv = tn / (tn + fn + .00001)
    sensitivity = tp / (tp + fn + .00001)
    specificity = tn / (tn + fp + .00001)

    #f1_score = (2 * precision * recall) / (precision + recall + .00001)

    return ppv, npv, sensitivity, specificity

