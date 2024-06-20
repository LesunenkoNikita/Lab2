import numpy as np
def get_eigen(square_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(square_matrix)
    print("Eigenvalues:")
    for i in range(len(eigenvalues)):
        print(f"λ{i + 1} = {eigenvalues[i]}")

    print("Eigenvectors")
    for i in range(len(eigenvectors)):
        print(f"v{i + 1} = {eigenvectors[:, i]}")

    for i in range(len(eigenvalues)):
        if np.allclose(np.dot(square_matrix, eigenvectors[i]), eigenvalues[i] * eigenvectors[i]):
            print(f"Рівність A*v{i + 1} = λ{i + 1}*v{i + 1} істинна")
        else:
            print(f"Рівність A*v{i + 1} = λ{i + 1}*v{i + 1} хибна")
