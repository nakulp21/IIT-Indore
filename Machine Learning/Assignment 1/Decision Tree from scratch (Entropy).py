class Classifier:
    # def __init__(self, test_data):
    #     self.test_data = test_data

    # Finds entropy of the parent node
    @staticmethod
    def find_entropy_node(data):
        entropy = 0
        _, counts = np.unique(data[:, -1], return_counts=True)
        total = np.sum(counts)

        for count in counts:
            probability = count / total
            entropy += -probability * np.log2(probability)

        return entropy

    # Finds entropy of every other feature except parent to find the next node
    @staticmethod
    def find_entropy_feature(data, feature):    # Since table is converted into numpy
        entropy = 0
        _, counts = np.unique([data[:, feature]], return_counts=True)
        total = np.sum(counts)

        for count in counts:
            probability = count / total
            entropy += -probability * np.log2(probability)

        return entropy

    # Finds best feature to be taken as parent/next node of the tree
    def find_best_feature(self, data):
        # print("\ndata: ", str(data))
        info_gain = []
        entropy_node = self.find_entropy_node(data)
        for i in range(len(data[0]) - 1):
            e = self.find_entropy_feature(data, i)
            info_gain.append(entropy_node - e)

        # print(info_gain)

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
        n = features[parent_node]
        feature_values = np.unique(table[:, parent_node])   # Extract all feature values to form subtree

        if tree is None:
            tree = {}
            tree[features[parent_node]] = {}

        for value in feature_values:
            subtable, f = self.split_table(table, parent_node, value, features) # Getting subtree of the particular (feature, value) pair
            f_val, counts = np.unique(subtable[:, -1], return_counts=True)  # Returns counts of each value of target feature in the subtable
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


def predict(row, tree, features):

    for node in tree.keys():
        ind = list(features).index(node) # Finding index of feature in features list
        value = row[ind] # Getting value of feature the input row
        tree = tree[node][value] # Getting subtree of (feature, value) pair

        if isinstance(tree, dict): # If the value returned is a tree then needs further probing
            prediction = predict(row, tree, features)
        else:
            prediction = tree  # Is true when we reach a leaf node
            break

    return prediction


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from pprint import pprint

    # Preparing Data
    train_data = pd.read_excel("/home/nakul/Documents/Train Data.xlsx")
    test_data = pd.read_excel("/home/nakul/Documents/Test Data.xlsx")

    # Converting data to numpy array since it is faster than pandas
    features = train_data.columns[:-1].values  # Extracting all feature names
    train_data = train_data.values  # Extracting training data
    test_data = test_data.values  # Extracting test data
    test_data = test_data[:, :-1]

    tree = Tree(train_data, features).fit() # Tree object

    print("Decision Tree")
    pprint(tree)

    for row in test_data:
        for i, j in zip(features, row):
            print(i, ": ", str(j))
        ans = predict(row, tree, features)
        print("Profitable: ",  ans)
        print()
