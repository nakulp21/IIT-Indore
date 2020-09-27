class Graph:
    def __init__(self, graph):
        # residual graph
        self.graph = graph
        self.girls = len(graph)
        self.boys = len(graph[0])

        # A DFS based recursive function

    def bpm(self, u, matchR, seen):

        for v in range(self.boys):
            if self.graph[u][v] and seen[v] == False:

                # Mark v as visited
                seen[v] = True
                if matchR[v] == -1 or self.bpm(matchR[v], matchR, seen):
                    matchR[v] = u
                    return True
        return False

    # Returns maximum number of matching
    def maxBPM(self):
        matchR = [-1] * self.boys

        result = 0
        for i in range(self.girls):

            seen = [False] * self.boys

            if self.bpm(i, matchR, seen):
                result += 1
        return result


if __name__ == '__main__':

    """
    General method to create a adjacency matrix of the graph
    """
    # n_girls, n_boys = map(int, input("Enter number of girls and boys: ").split(" "))
    # adj_matrix = [[0 for _ in range(n_boys)] for _ in range(n_girls)]
    #
    # for girl in range(n_girls):
    #     neighbours = list(map(int, input("Add neighbour(s) of girl {0}: ".format(girl + 1)).split()))
    #     if len(neighbours):
    #         for boy in neighbours:
    #             adj_matrix[girl][boy - 1] = 1

    adj_matrix = [[1, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1],
                  [0, 0, 1, 0, 1]]

    g = Graph(adj_matrix)

    print("Number of maximum matching = %d " % g.maxBPM())
