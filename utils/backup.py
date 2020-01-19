
def eval_sarcopenia(clf, X, gender, height_squared, sarcopenia_ground_truth):
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

def observe_prediction_SVR(clf, X, y, patient_id, dont_show=True):
    y_predict = clf.predict(X)
    diff = (y_predict - y) / y
    patient_id = patient_id
    diff_percent = diff * 100
    num_examples = len(y)
    if not dont_show:
        for i in range(num_examples):
            if(np.abs(diff_percent[i])>5):
                print("Truth: %.2f, Predicted: %.2f, Error: %6.2f%%, patient_id: %3d" %(y[i], y_predict[i], diff_percent[i], patient_id[i]))
    return diff


def test(show=False):
    #total = 0.0
    net.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data['X'], data['asm_h2']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs).reshape(-1)
            if show:
                print(outputs)
                print(targets)
            #err = outputs - labels
            #print(err)
            loss_fn = nn.MSELoss()
            test_loss = loss_fn(outputs.data, targets.data)
            #total += torch.sum(abs(err))


    print("Test loss: %.8f" % test_loss)
test(True)
