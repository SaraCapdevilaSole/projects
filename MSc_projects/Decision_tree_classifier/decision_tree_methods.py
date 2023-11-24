# packages

import numpy as np
import matplotlib.pyplot as plt
import sys
#print (sys.version)
from numpy.random import default_rng
import copy

class DT_Learning(object):
    def __init__(self, file_path):
        self._file_path = file_path
        self._data = []
    
    def read_data(self):
        self._data = np.loadtxt(self._file_path) 
        return self._data
    
    def _H(self, data):
        unique_labels, freqs = np.unique(data[:,-1], return_counts=True) #freqs: number of times each unique item appears in the dataset
        total = np.sum(freqs)
        p_k = freqs/total 
        entropy = -(np.dot(p_k,np.log(p_k)))
        return entropy
    
    def find_split(self, data):
        m, n = data.shape
        best_split = (0, 0, 0, 0, 0) # feature, split, gain, data_L, data_R
        for i in range(0, n - 1):
            sorted_data = data[data[:, i].argsort()]
            for s in range(1, m):
                data_L, data_R = sorted_data[:s, :], sorted_data[s:, :]
                gain_per_split = self._gain(data, data_L, data_R)
                if gain_per_split > best_split[2]:
                    best_split = (i, sorted_data[s, i], gain_per_split, data_L, data_R)
        return best_split
    
    def _remainder(self, data_L, data_R):
        m_L, m_R = [data.shape[0] for data in [data_L, data_R]]
        total = m_L + m_R
        H_L, H_R = [self._H(data_i) for data_i in [data_L, data_R]]
        remainder = m_L / total * H_L + m_R / total * H_R
        return remainder

    def _gain(self, data, data_L, data_R):
        return self._H(data) - self._remainder(data_L, data_R)
    
    def decision_tree_learning(self, data):
        rooms, counts = np.unique(data[:, -1], return_counts=True)
        voting = np.zeros(4)    # number of rooms
        # print(rooms, counts)
        if len(rooms) == 1:
            voting[int(rooms[0]) - 1] = counts[0]
            return {"Room": rooms[0],
                    "Voting": voting
                   }
        else:
            feature, split, gain, data_L, data_R = self.find_split(data)
            node = {"Router": feature,
                    "Split": split,
                    "Left": {},
                    "Right": {},
                    "IsPruned": False
                   }
            node["Left"] = self.decision_tree_learning(data_L)
            node["Right"] = self.decision_tree_learning(data_R)
    
            return node

class DT_Classifier(object):
    def __init__(self, Tree, data):
        self._Tree = Tree
        self._data = data # data is test data
        
    def _decision_tree_classifier_row(self, Tree, data_row):
        if 'Router' in Tree:
            if data_row[Tree['Router']] < Tree['Split']:
                return self._decision_tree_classifier_row(Tree['Left'], data_row)
            else:
                return self._decision_tree_classifier_row(Tree['Right'], data_row)
        else:
            room = Tree['Room'] # we have reached a leaf node
            return room

    def decision_tree_classifier(self):
        rooms = []
        for data_row in self._data:
            room = self._decision_tree_classifier_row(self._Tree, data_row)
            rooms.append(room)
        return np.array(rooms)

class Plot_DT():
    def _inv_depth(self, Tree):
        if 'Router' in Tree:
            return 1 + max(self._inv_depth(Tree['Left']), self._inv_depth(Tree['Right']))
        else:
            return 1
    
    def plot_decision_tree(self, Tree, x=0, y=0, width=100, depth=1, parent_x=None, parent_y=None, dy = 80):
        if 'Router' in Tree:
            router = Tree['Router']
            split = Tree['Split']
            left_subtree = Tree['Left']
            right_subtree = Tree['Right']
    
            # Calculate the coordinates of the decision node
            node_x = x + width / 2
            node_y = y
    
            if parent_x is not None and parent_y is not None:
                # Draw an edge connecting the decision node to its parent
                plt.plot([parent_x, node_x], [parent_y, node_y], color='black', linestyle='-', linewidth=1)
    
            # Plot the decision node
            plt.text(node_x, node_y, f"Router {router}\n < {split}", ha='center', va='center',
                     bbox=dict(facecolor='yellowgreen', edgecolor='black', boxstyle='round,pad=0.2'),
                     fontsize=14)
            
            width_scaling_factor = self._inv_depth(Tree['Left'])/(self._inv_depth(Tree['Left']) + self._inv_depth(Tree['Right']))*1.1 
            
            new_width1 = width * width_scaling_factor
            new_width2 = width * (1 - width_scaling_factor)
            # Plot left subtree
            self.plot_decision_tree(left_subtree, x - width/2, y - dy, new_width1, depth + 1, node_x, node_y)
    
            # Plot right subtree
            self.plot_decision_tree(right_subtree, x + width/2, y - dy, new_width2, depth + 1, node_x, node_y)
        else:
            # Leaf node, plot the room label
            room = Tree['Room']
            plt.text(x + width / 2, y, f"Room {int(room)}", ha='center', va='center',
                     bbox=dict(facecolor='lemonchiffon', edgecolor='black', boxstyle='round,pad=0.2'),
                     fontsize=16)
            
            if parent_x is not None and parent_y is not None:
                # Draw an edge connecting the leaf node to its parent decision node
                plt.plot([parent_x, x + width / 2], [parent_y, y], color='black', linestyle='-', linewidth=1)

class Cross_validation(object):
    def __init__(self, data, k = 10):
        self._k = k
        self._data = data
        self._n_samples = data.shape[0]
        self._rg = default_rng()
        
    #divide dataset into k splits
    def _k_fold_split(self):
        """
        Input 
            k: Splits the dataset into k segments
            n_samples: data.shape[0] i.e. the number of rows
        Output: the indices contained in each split (matirx of 10 x number of data points / 10)
        """
        shuffled_indices = self._rg.permutation(self._n_samples)
        split_indices = np.array(np.array_split(shuffled_indices, self._k)) 
        return split_indices
    
    def _train_test_k_fold_indices(self):
        """
        Input
            k_folds: number of cross-validation splits
        Output
            List of length k_folds, each element containing a list of train/test indices
            We get k train test splits
        """
        #split the dataset into k splits
        split_indices = self._k_fold_split()
        
        folds = []
        for k in range(self._k):
            test_indices = split_indices[k] #pick k as the test set
            #combine remaining splits as train
            train_indices = np.concatenate((split_indices[:k,:], split_indices[k+1:,:]))
            train_indices = train_indices.flatten() # list of indices in the train dataset
            folds.append([train_indices, test_indices])
        return folds
    
    def train_test_k_fold(self): 
        test_train_folds = self._train_test_k_fold_indices()
        train_data_folds = []
        test_data_folds = []
        for k in range(self._k):
            train_indices, test_indices = test_train_folds[k]
            train_data = [self._data[x, :] for x in train_indices]
            test_data = [self._data[x, :] for x in test_indices]
            train_data_folds.append(np.array(train_data))
            test_data_folds.append(np.array(test_data))
        return train_data_folds, test_data_folds
    
    def train_validation_test_k_fold(self):
        """
        Input
            k_folds: number of cross-validation splits
        Output
            List of length k_folds, each element containing a list of train/test indices
            We get k train test splits
        """
        #split the dataset into k splits
        split_indices = self._k_fold_split()
    
        folds=[]
        for k in range(self._k):
            test_indices = split_indices[k] #pick k as the test set
            #combine remaining splits as train
            train_validation_indices = np.concatenate((split_indices[:k,:], split_indices[k+1:,:]))
            train_validation_indices = train_validation_indices.flatten() # list of indices in the train dataset
            train_validation_indices = np.array(np.array_split(train_validation_indices, self._k-1))
            for k_ in range(self._k - 1):
                validation_indices = train_validation_indices[k_]
                train_indices = np.concatenate((train_validation_indices[:k_], train_validation_indices[k_+1:]))
                train_indices = train_indices.flatten()
                folds.append([train_indices, test_indices, validation_indices])
        return folds
    
class Evaluation(DT_Classifier): # just for one fold
    def __init__(self, trained_tree, test_data):
        self._n = 4
        self.trained_tree = trained_tree
        self.test_data = test_data # containing room values
        super().__init__(Tree = trained_tree, data = test_data[:, :-1])
        self.room_predictions = self.decision_tree_classifier()
        self.rooms = test_data[:, -1]
        self.confusion = np.zeros((self._n, self._n)) # initialise confusion matrix
    
    def evaluate(self): # returns accuracy        
        assert len(self.room_predictions) == len(self.rooms), "length of predicted and actual do not match!"
        T = sum(self.room_predictions == self.rooms)
        F = sum(self.room_predictions != self.rooms)
        return T/(T+F)
    
    def compute_confusion_matrix(self):
        assert len(self.room_predictions) == len(self.rooms), "length of predicted and actual do not match!"
        room_numbers = [x + 1 for x in range(self._n)]
    
        for room in room_numbers: # iterate through each class
            class_indices = (self.rooms == room)
            predicted_room_c = self.room_predictions[class_indices]
    
            unique_labels, counts = np.unique(predicted_room_c, return_counts = True)
    
            freq_dict = dict(zip(unique_labels, counts))
    
            for room_ in room_numbers:
                self.confusion[room - 1, room_ - 1] = freq_dict.get(room_, 0) # 0 so that it returns 0 instead of None

    def get_confusion(self):
        return self.confusion
    
    def accuracy(self):
        assert np.all(self.confusion==np.zeros((self._n, self._n))) == False, "You must compute the confusion matrix first"
        T = np.sum(np.diag(self.confusion))
        total = np.sum(self.confusion)
        return T/total
    
    def precision(self):
        assert np.all(self.confusion==np.zeros((self._n, self._n))) == False, "You must compute the confusion matrix first"
        precision = np.zeros(self._n)
        for room in range(self._n):
            total_predictions = np.sum(self.confusion[:, room])
            if total_predictions > 0:
                precision[room] = self.confusion[room, room] / total_predictions
        return precision
    
    def recall(self):
        assert np.all(self.confusion==np.zeros((self._n, self._n))) == False, "You must compute the confusion matrix first"
        recall = np.zeros(self._n)
        for room in range(self._n):
            total_predictions = np.sum(self.confusion[room, :])
            if total_predictions > 0:
                recall[room] = self.confusion[room, room] / total_predictions
        return recall
    
    def F1(self):
        assert np.all(self.confusion==np.zeros((self._n, self._n))) == False, "You must compute the confusion matrix first"
        p = self.precision()
        r = self.recall()
        F_1 = 2 * p * r / (p + r)
        return F_1

def prune(tree, pruned_nodes):
    if len(pruned_nodes) > 0:
      return tree
    elif 'Room' in tree:
      return tree
    elif tree.get('IsPruned') and tree['IsPruned'] == True:
      return tree
    elif 'Room' in tree['Left'].keys() and 'Room' in tree['Right'].keys():
      pruned_nodes.append(tree)
      total_voting = tree['Left']['Voting'] + tree['Right']['Voting']
      return {
        'Room': np.argmax(total_voting) + 1,
        'Voting': total_voting
      }

    return {
        "Router": tree["Router"],
        "Split": tree["Split"],
        "Left": prune(tree['Left'], pruned_nodes),
        "Right": prune(tree['Right'], pruned_nodes),
        "IsPruned": False
        }
        
def average_metrics_over_nested_k_folds(file_path, k_folds = 10):
    DT = DT_Learning(file_path)
    dataset = DT.read_data()
    folds = Cross_validation(dataset, k_folds).train_validation_test_k_fold()
    
    confusions, accuracies, precisions, recalls, f1s = [], [], [], [], []
    
    kth = 0
    for train, val, test in folds:
        print("Pruning on k={} nested fold...".format(kth))
        kth += 1
        train, val, test = dataset[train, :], dataset[val, :], dataset[test, :]
        Nonpruned_Tree = DT.decision_tree_learning(train)
        
        evaluate = Evaluation(Nonpruned_Tree, val)
        evaluate.compute_confusion_matrix()
        Nonpruned_accuracy = evaluate.accuracy()
        
        while True:
            pruned_nodes = []
            Pruned_Tree = prune(Nonpruned_Tree, pruned_nodes)
            
            evaluate = Evaluation(Pruned_Tree, val)
            evaluate.compute_confusion_matrix()
            Pruned_accuracy = evaluate.accuracy()
            
            if Pruned_accuracy >= Nonpruned_accuracy:
                Nonpruned_Tree = copy.deepcopy(Pruned_Tree)
                Nonpruned_accuracy = Pruned_accuracy
            else:
                pruned_nodes[0]["IsPruned"] = True
            if len(pruned_nodes) == 0:
                break
        
        evaluate = Evaluation(Nonpruned_Tree, test)
        evaluate.compute_confusion_matrix()
        
        confusion = evaluate.get_confusion()
        accuracy = evaluate.accuracy()
        prec = evaluate.precision()
        rec = evaluate.recall()
        f1 = evaluate.F1()
        
        confusions.append(confusion)
        accuracies.append(accuracy)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    return average_metrics(confusions, accuracies, precisions, recalls, f1s)

def average_metrics(confusions, accuracies, precisions, recalls, f1s):
    
    sum_confusions = np.zeros((len(confusions[0]), len(confusions[0])))
    for x in confusions:
        sum_confusions += x

    avg_confusion = sum_confusions / len(confusions)
    avg_accuracy = np.sum(accuracies) / len(accuracies)
    avg_precision = np.sum(precisions, axis=0) / len(precisions)
    avg_recall = np.sum(recalls, axis=0) / len(recalls)
    avg_f1 = np.sum(f1s, axis=0) / len(f1s)
    
    return avg_confusion, avg_accuracy, avg_precision, avg_recall, avg_f1

def average_metrics_over_k_folds(file_path, k_folds = 10):

    DT = DT_Learning(file_path)
    data = DT.read_data()
    train, test = Cross_validation(data, k_folds).train_test_k_fold()

    confusions, accuracies, precisions, recalls, f1s = [], [], [], [], []

    for k, training_k in enumerate(train):

        Tree = DT.decision_tree_learning(training_k)
        Test_k = test[k]
        evaluate = Evaluation(Tree, Test_k)
        evaluate.compute_confusion_matrix()
        
        confusion = evaluate.get_confusion()
        accuracy = evaluate.accuracy()
        prec = evaluate.precision()
        rec = evaluate.recall()
        f1 = evaluate.F1()
        
        confusions.append(confusion)
        accuracies.append(accuracy)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    return average_metrics(confusions, accuracies, precisions, recalls, f1s)

def macro_average(metrics):
    return [metrics[1], np.mean(metrics[2]), np.mean(metrics[3]), np.mean(metrics[4])]







        
        
        