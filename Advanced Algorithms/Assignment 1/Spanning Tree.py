""""
1. Problem - Find number of spanning trees using Kirchoff's Theorem.
2. Code not optimised. Check
3. Label the vertices starting from 0 e.g. 0,1,2.....
4.Use of inbuilt modules is limited since its an Algo assignment
For Laplacian matrix use - from scipy.sparse.csgraph import laplacian
For LU Decomposition use - from scipy.linalg.lu
"""
import numpy as np

np.set_printoptions(precision=3)

class Graph:
    def __init__(self, size):
        self.adj_matrix = [[0 for _ in range(size)] for _ in range(size)]

    def add_edge(self, v1, neighbour):
        if not neighbour:
            return

        for vertex in neighbour:
            if v1 != vertex:
                self.adj_matrix[v1][vertex] = 1

    def laplacian(self):
        lap_matrix = np.copy(np.array(self.adj_matrix))
        for vertex in range(size):
            deg = 0
            for val in range(size):
                if lap_matrix[vertex][val]:
                    deg += 1
                    lap_matrix[vertex][val] = -1

            lap_matrix[vertex][vertex] = deg

        print(lap_matrix)
        return lap_matrix

    def print_mat(self):  # Note: There might be a diff way to print this using string formatting. Check
        for row in self.adj_matrix:
            print("  ".join(str(val) for val in row))

        print("\n")
        

def luDecomposition(mat, n):
    # Initializing the matrices
    lower = np.array([[0.0]*n for _ in range(n)])
    for i in range(n):
        lower[i][i] = 1.0

    upper = np.array([[0.0]*n for _ in range(n)])

    # Decomposing matrix into Upper
    # and Lower triangular matrix
    for k in range(n):
        upper[k][k] = mat[k][k]

        for i in range(k + 1, n):
            lower[i][k] = mat[i][k] / upper[k][k]
            upper[k][i] = mat[k][i]

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                mat[i][j] -= lower[i][k] * upper[k][j]

    return lower, upper


# Using Doolittle's Algorithm
# def luDecomposition(mat, n):
#     lower = np.array([[0]*n for _ in range(n)], dtype=float)
#     upper = np.array([[0]*n for _ in range(n)], dtype=float)

#     # Decomposing matrix into Upper
#     # and Lower triangular matrix
#     for i in range(n):

#         # Upper Triangular
#         for k in range(i, n):

#             # Summation of L(i, j) * U(j, k)
#             total = 0.0
#             for j in range(i):
#                 total += (lower[i][j] * upper[j][k])

#                 # Evaluating U(i, k)
#             upper[i][k] = mat[i][k] - total

#             # Lower Triangular
#         for k in range(i, n):
#             if i == k:
#                 lower[i][i] = 1.0  # Diagonal as 1
#             else:

#                 # Summation of L(k, j) * U(j, i)
#                 total = 0.0
#                 for j in range(i):
#                     total += (lower[k][j] * upper[j][i])

#                     # Evaluating L(k, i)
#                 lower[k][i] = (mat[k][i] - total) / upper[i][i]

#     print("\nLower Triangular Matrix:")
#     print(lower)

#     print("\nUpper Triangular Matrix:")
#     print(upper)

#     return upper


def determinant(matrix):
    return np.product(np.diag(matrix))


if __name__ == '__main__':
    size = int(input("Enter number of vertices in the graph: "))

    g = Graph(size)

    # Getting neighbours of every vertex
    for i in range(size):
        neighbours = list(map(int, input("Add neighbours of vertex {0}: ".format(i)).split()))
        g.add_edge(i, neighbours)

    print("Adjacency Matrix of the given graph")
    g.print_mat()

    print("The corresponding Laplacian Matrix")
    lap_mat = g.laplacian()

    # Deleting 1st row and column from matrix
    m1 = np.delete(lap_mat, 0, 0)
    m2 = np.array(np.delete(m1, 0, 1), dtype=float)
    print("\nSubmatrix\n")
    print(m2)

    U = luDecomposition(m2, size - 1)

    det_upper = determinant(U)

    print("\nNumber of spanning tree = ", str(int(det_upper)))
