from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

infty = 999999999


def entropy(Y):
    y_malignent = len([y for y in Y if y == 1])
    y_benign = len([y for y in Y if y == 0])
    y = len(Y)
    val_one = y_malignent/y
    if val_one != 0:
        val_one *= np.log2(val_one)
    val_two = y_benign/y
    if val_two != 0:
        val_two *= np.log2(val_two)
    return -val_one - val_two


class Node:
    def __init__(self, X, Y, parent=None):
        self.parent = parent
        self.leaf = False
        self.left_child = None
        self.right_child = None
        self.data = X
        self.target = Y
        self.label = self.get_label(Y)
        self.split_feature = None
        self.split_value = None

    @staticmethod
    def get_label(Y):
        class_0 = len([y for y in Y if y == 0])
        class_1 = len([y for y in Y if y == 1])
        return 0 if class_0 > class_1 else 1

    def devide(self):
        if entropy(self.target) == 0:
            self.leaf = True
            return
        best_feature = 0
        best_alpha = 0
        least_entropy = infty
        for i in range(30):
            values = [x[i] for x in self.data]
            for v in values:
                left = [self.target[j] for j in range(len(self.target)) if self.data[j][i] <= v]
                right = [self.target[j] for j in range(len(self.target)) if self.data[j][i] > v]
                if not right:
                    continue
                new_entropy = (len(left)/len(self.target))*entropy(left) + (len(right)/len(self.target))*entropy(right)
                if new_entropy < least_entropy:
                    least_entropy = new_entropy
                    best_alpha = v
                    best_feature = i
        self.split_feature = best_feature
        self.split_value = best_alpha
        left_indices = [i for i in range(len(self.data)) if self.data[i][best_feature] <= best_alpha]
        right_indices = [i for i in range(len(self.data)) if self.data[i][best_feature] > best_alpha]
        left_X = self.data[left_indices]
        left_y = self.target[left_indices]
        right_X = self.data[right_indices]
        right_y = self.target[right_indices]
        self.left_child = Node(left_X, left_y, self)
        self.right_child = Node(right_X, right_y, self)
        self.left_child.devide()
        self.right_child.devide()

    def classify(self, x):
        if self.leaf:
            return self.label
        if x[self.split_feature] <= self.split_value:
            return self.left_child.classify(x)
        else:
            return self.right_child.classify(x)

    def print(self, id):
        if self.leaf:
            return ""
        id_l = id + 'l'
        id_r = id + 'r'
        l_split = r_split = ""
        if not self.left_child.leaf:
            l_split = r'X[{0}] <= {1}\n'.format(self.left_child.split_feature, self.left_child.split_value)
        if not self.right_child.leaf:
            r_split = r'X[{0}] <= {1}\n'.format(self.right_child.split_feature, self.right_child.split_value)
        l_def = r'[label="{0}entropy = {1}\nsamples = {2}\nlabel = {3}"]'.format(l_split,
                                                                                 entropy(self.left_child.target),
                                                                                 len(self.left_child.target),
                                                                                 self.left_child.label)
        r_def = r'[label="{0}entropy = {1}\nsamples = {2}\nlabel = {3}"]'.format(r_split,
                                                                                 entropy(self.right_child.target),
                                                                                 len(self.right_child.target),
                                                                                 self.right_child.label)
        text = '\t{} {}\n'.format(id_l, l_def)
        text += '\t{} {}\n'.format(id_r, r_def)
        text += '\t{} -> {}\n'.format(id, id_l)
        text += '\t{} -> {}\n'.format(id, id_r)
        text += self.left_child.print(id_l)
        text += self.right_child.print(id_r)
        return text


def print_graph(tree):
    text = 'digraph g {\n'
    i_def = r'[label="X[{0}] <= {1}\nentropy = {2}\nsamples = {3}\nlabel = {4}"]'.format(tree.split_feature,
                                                                                         tree.split_value,
                                                                                         entropy(tree.target),
                                                                                         len(tree.target),
                                                                                         tree.label)
    text += '\t{} {}\n'.format('O', i_def)
    text += tree.print('O')
    text += '}'
    print(text)


def score(test_X, test_Y, tree):
    all_samples = len(test_Y)
    good_results = 0
    for i in range(len(test_X)):
        res = tree.classify(test_X[i])
        if res == test_Y[i]:
            good_results += 1
    return float(good_results)/all_samples


if __name__ == '__main__':
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train, y_train)
    res2 = clf.score(X_test, y_test)
    print('Sklearn accuracy is {}'.format(res2))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    root = Node(X_train, y_train)
    root.devide()
    res = score(X_test, y_test, root)
    print('Our accuracy is {}'.format(res))
    print_graph(root)

