
class Classifier:
    def __init__(self):
        self.gini = []
        self.information_gain = []

    # Finds gini of the parent node
    @staticmethod
    def get_counts(data):
        _, counts = np.unique(data, return_counts=True)
        return counts

    def find_gini_node(self, data):
        gini_node = 0
        _, counts = np.unique(data[:, -1], return_counts=True)
        total = np.sum(counts)

        for count in counts:
            probability = count / total
            gini_node += -probability**2

        self.gini.append(gini_node)
        # print("gini")
        # print(self.gini)

        return gini_node

    # Finds gini of every other feature except parent to find the next node

    def find_gini_feature(self, data, feature):  # Since table is converted into numpy
        gini_feature = 0
        counts = self.get_counts(data[:, feature])
        total = np.sum(counts)

        for count in counts:
            probability = count / total
            gini_feature += -probability**2

        return gini_feature

    # Finds best feature to be taken as parent/next node of the tree
    def find_best_feature(self, table):
        # print("\ndata: ", str(data))
        info_gain = []
        gini_node = self.find_gini_node(table)

        for i in range(len(table[0]) - 1):
            info_gain.append(gini_node - self.find_gini_feature(table, i))

        # print(info_gain)
        self.information_gain.append(max(info_gain))
        # print(self.information_gain)

        return info_gain.index(max(info_gain))


class Tree:

    def __init__(self, train_data, features):
        self.g = Classifier()
        self.train_data = train_data
        self.features = features

    def fit(self):
        return self.build_tree(self.train_data, self.features)

    def build_tree(self, table, features, tree=None):
        parent_node = self.g.find_best_feature(table)  # Get the best node for splitting
        # n = features[parent_node]
        feature_values = np.unique(table[:, parent_node])  # Extract all feature values to form subtree

        if tree is None:
            tree = {features[parent_node]: {}}

        for value in feature_values:
            subtable, f = self.split_table(table, parent_node, value,
                                           features)  # Getting subtree of the particular (feature, value) pair
            f_val, counts = np.unique(subtable[:, -1],
                                      return_counts=True)  # Returns counts of each value of target feature in the
            # subtable
            #             print("\nf_val" + str(f_val))
            #             print("\ncounts =" + str(counts))

            if len(f_val) == 1:
                tree[features[parent_node]][value] = f_val[0]

            else:
                tree[features[parent_node]][value] = self.build_tree(subtable, f)

        return tree

    @staticmethod
    def split_table(table, node, value, features):
        table = table[table[:, node] == value]
        return np.delete(table, node, 1), np.delete(features, node)

    def root_node(self):
        return self.g.gini[0], self.g.information_gain[0]


def predict(row, tree, features):
    global prediction
    for node in tree.keys():
        ind = list(features).index(node)  # Finding index of feature in features list
        value = row[ind]  # Getting value of feature the input row
        tree = tree[node][value]  # Getting subtree of (feature, value) pair

        if isinstance(tree, dict):  # If the value returned is a tree then needs further probing
            prediction = predict(row, tree, features)
        else:
            prediction = tree  # Is true when we reach a leaf node
            break

    return prediction

def print_data(data, tree, spacing=""):
    #     pprint(tree)
    if isinstance(tree, dict):
        node = list(tree.keys())[0]
        print("|" + node + " = " + str(data[node]))
        tree = tree[node][data[node]]
        data.pop(node)
        #         print(data)
        print_data(data, tree, spacing)

    else:
        if data:
            for key in data.keys():
                spacing += " "
                print("|" + key + " = " + str(data[key]))


def get_accuracy(predictions, original):
    true = 0
    for i, j in zip(predictions, original):
        if i == j:
            true += 1

    accuracy = true/len(original)

    return accuracy


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from pprint import pprint
    np.set_printoptions(precision=3)

    # Preparing Data
    train_data = pd.read_excel("/home/nakul/Documents/Train Data.xlsx")
    test_data = pd.read_excel("/home/nakul/Documents/Test Data.xlsx")

    # Converting data to numpy array since it is faster than pandas
    features = train_data.columns[:-1].values  # Extracting all feature names
    train_data = train_data.values  # Extracting training data
    test_data = test_data.values  # Extracting test data
    target = test_data[:, -1]
    test_data = test_data[:, :-1]

    obj = Tree(train_data, features)
    tree = obj.fit()  # Tree object
    gini_root, info_gain_root = obj.root_node()

    print("Root Gini Index = {:.3f} ".format(gini_root))
    print("Root Information Gain = {:.3f}".format(info_gain_root))

    print("\nDecision Tree")
    pprint(tree)
    print()

    ans_list = []
    for row in test_data:

        #     print(row)
        #     print([*zip(features, row)])
        print_data(dict(zip(features, row)), tree)
        ans = predict(row, tree, features)
        ans_list.append(ans)
        print("Profitable? : ", ans, "\n")

    # Get Accuracy
    print("Accuracy? = {:.2%} ".format(get_accuracy(ans_list, target)))
