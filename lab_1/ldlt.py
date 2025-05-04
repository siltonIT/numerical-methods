import numpy as np 
from typing import Tuple

class LDLTSolver():
    def __init__(self, matrix: np.ndarray, free_terms: np.ndarray):
        if matrix.shape[0] != free_terms.shape[0]:
            raise ValueError("Количество строк в матрице должно совпадать с количеством совбодных членов.")

        self.matrix = matrix.astype(float).copy()
        self.free_terms = free_terms.astype(float).copy()

    def print_equation_system(self) -> None:
        """Выводит систему уравнений в читаемом формате."""
        print("Система уравнений:")
        for i, row in enumerate(self.matrix):
            print(f"{row} | {self.free_terms[i]}")
        print()

    def solve_equation_system(self) -> np.ndarray:
        """Решает систему уравнений используя LDLT"""
        L, D = self.__LDLT_decomposition(self.matrix)
        y = np.linalg.solve(L, self.free_terms)
        z = y / D

        roots = np.linalg.solve(L.T, z)
        return roots

    def __LDLT_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Разлагает симметричную матрицу в произведение LDLT"""
        n_rows = matrix.shape[0]
        L = np.eye(n_rows)
        D = np.zeros(n_rows)

        for i in range(n_rows):
            D[i] = matrix[i, i] - sum(L[i, k]**2 * D[k] for k in range(i))
            for j in range(i + 1, n_rows):
                L[j, i] = (matrix[j, i] - sum(L[j, k] * L[i, k] * D[k] for k in range(i))) / D[i]

        return L, D

def main():
    a = np.array([[6, 13, -17], [13, 29, -38], [-17, -38, 50]], float)
    b = np.array([2, 4, -5], float)

    ldlt_solver = LDLTSolver(a, b) 
    ldlt_solver.print_equation_system()

    x = ldlt_solver.solve_equation_system()
    print("Решение системы уравнений:", x) 

if __name__ == "__main__":
    main()
