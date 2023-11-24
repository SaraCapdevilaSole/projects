#### Before pruning

k_folds = 10

def cross_val_eval(data):
  confusions, accuracies, precisions, recalls, f1s = [], [], [], [], []

  folds = cross_validation(data)
  for i in range(len(folds[0])):
    train, test = folds[0][i], folds[1][i]
    train_input, train_output = train[:, :-1], train[:, -1]
    test_input, test_output = test[:, :-1], test[:, -1]

    Tree_trained, _ = Decision_tree_learning(train)
    Test_rooms_predicted = Decision_tree_classifier(Tree_trained, test_input)

    confusion = confusion_matrix(Test_rooms_predicted, test_output)
    accuracy = accuracy_from_confusion(confusion)
    prec = precision(confusion)
    rec = recall(confusion)
    f1 = F1(confusion)

    confusions.append(confusion)
    accuracies.append(accuracy)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

  sum_confusions = np.zeros((len(confusions[0]), len(confusions[0])))
  for x in confusions:
    sum_confusions += x

  avg_confusion = sum_confusions / len(confusions)
  avg_accuracy = np.sum(accuracies) / len(accuracies)
  avg_precision = np.sum(precisions, axis=0) / len(precisions)
  avg_recall = np.sum(recalls, axis=0) / len(recalls)
  avg_f1 = np.sum(f1s, axis=0) / len(f1s)

  return avg_confusion, avg_accuracy, avg_precision, avg_recall, avg_f1

cross_val_eval(clean_data)

#### After pruning

train_test_folds = cross_validation(clean_data)
for i in range(len(train_test_folds[0])):
  train_val, test = train_test_folds[0][i], train_test_folds[1][i]
  _, accuracy_train, _, _, _ = cross_val_eval(train_val)
