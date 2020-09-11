def train_test_split(data, test_size=0.2):
    test_size = round(test_size * len(data))
    index = random.sample(population=list(range(len(data))), k=test_size)

    train_data = np.delete(data, index, axis=0)
    test_data = data[index]

    return train_data, test_data


def check_purity(data):
    unique_values = np.unique(data[:, -1])
    if len(unique_values) == 1:
        return True

    else:
        return False


def classify_data(data):
    unique_classes, counts_unique_classes = np.unique(data[:, -1], return_counts=True)
    i = counts_unique_classes.argmax()
    classification = int(unique_classes[i])

    return classification


def possible_splits(data):
    potential_split = {}
    _, cols = data.shape
    for col_index in range(cols - 1):
        potential_split[col_index] = []
        unique_val = np.unique(data[:, col_index])

        for index in range(len(unique_val)):
            if index != 0:
                current_value = unique_val[index]
                previous_value = unique_val[index - 1]
                potential_split[col_index].append((current_value + previous_value) / 2)

    return potential_split


def gini(data):
    _, counts = np.unique(data[:, -1], return_counts=True)
    size = len(data)
    gini_parent = 1 - sum((count / size) ** 2 for count in counts)

    return gini_parent


def gini_split(left_subtree, right_subtree, size):
    size_left = len(left_subtree)
    size_right = len(right_subtree)

    return (size_left * gini(left_subtree) + size_right * gini(right_subtree)) / size


def determine_best_split(data, potential_splits):
    best_gini = gini(data)
    best_split_column, best_split_value = None, None

    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            left_subtree, right_subtree = split_data(data, split_column=column_index, split_value=value)
            current_gini = gini_split(left_subtree, right_subtree, len(data))

            if current_gini <= best_gini:
                best_gini = current_gini
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value, best_gini


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    left_subtree = data[split_column_values <= split_value]
    right_subtree = data[split_column_values > split_value]

    return left_subtree, right_subtree


def classifier(data, features, counter=0, min_samples=10, max_depth=4):

    if check_purity(data) or len(data) < min_samples or counter == max_depth:
        classification = classify_data(data)
        # print(counter)

        return classification

    else:
        counter += 1

        potential_splits = possible_splits(data)
        split_column, split_value, gini = determine_best_split(data, potential_splits)

        if counter - 1 == 0:
            print("Root Node Gini = ", str(gini))

        left_subtree, right_subtree = split_data(data, split_column, split_value)
        question = "{} <= {:.4f}".format(features[split_column], split_value)
        sub_tree = {question: []}

        yes_answer = classifier(left_subtree, features, counter)
        no_answer = classifier(right_subtree, features, counter)

        if yes_answer == no_answer:
            sub_tree = int(yes_answer)

        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def predict(row, tree, features):

    for node in tree.keys():
        label, comparator, value = node.split(" ")
        ind = features.index(label)

        if row[ind] <= float(value):
            answer = tree[node][0]

        else:
            answer = tree[node][1]

        if not isinstance(answer, dict):
            return answer

        else:
            answer = predict(row, answer, features)

    return answer


def accuracy(answers, test_data):
    correct = 0
    for i, j in zip(answers, test_data):
        if i == j:
            correct += 1

    accuracy = correct / len(test_data)

    return accuracy


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import random
    from pprint import pprint
    headers = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

    data = pd.read_csv('/home/nakul/Documents/data_banknote_authentication.csv').values

    train_data, test_data = train_test_split(data)
    tree = classifier(train_data, headers)
    pprint(tree)
    ans = []

    for row in test_data[:, :-1]:
        ans.append(predict(row, tree, headers))

    acc = accuracy(ans, test_data[:, -1])
    print("{:.2%}".format(acc))






