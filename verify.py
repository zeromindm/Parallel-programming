import numpy as np
import sys

def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        mat = []
        for _ in range(n):
            row = list(map(float, f.readline().split()))
            mat.append(row)
    return np.array(mat)

def main():
    size = int(sys.argv[1])

    A = read_matrix("data/matrix1.txt")
    B = read_matrix("data/matrix2.txt")

    C_prog = read_matrix("results/result.txt")
    
    C_numpy = np.dot(A, B)

    if np.allclose(C_prog, C_numpy):
        print("OK")
    else:
        print("FAILED")

if __name__ == "__main__":
    main()