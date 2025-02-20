import numpy as np
from typing import Any

class GaussianSolver:
    def __init__(self, matrix: np.ndarray, free_terms: np.ndarray):
        if matrix.shape[0] != free_terms.shape[0]:
            raise ValueError("Количество строк в матрице должно совпадать с количеством совбодных членов.")

        self.matrix = matrix.astype(float)
        self.free_terms = free_terms.astype(float)

    def print_equation_system(self) -> None:
        """Выводит систему уравнений в читаемом формате."""
        print("Система уравнений:")
        for i, row in enumerate(self.matrix):
            print(f"{row} | {self.free_terms[i]}")
        print()

    def solve_equation_system(self) -> np.ndarray:
        """Находит решение системы линейных уравнений."""
        self.__transform_to_upper_triangular_matrix()
        return self.__find_solution_vector()

    def __transform_to_upper_triangular_matrix(self) -> None:
        """Приводит матрицу к верхнетреугольному виду."""
        for i in range(self.matrix.shape[0] - 1):
            self.__find_pivot_element(i)
            self.__nullify_column(i)

    def __find_pivot_element(self, index: int) -> None:
        """Находит ключевой элемент в матрице и меняет его местами с элементом в строке index."""
        key_index = index + np.argmax(self.matrix[index:, index])
        
        if index != key_index:
            self.matrix[[key_index, index]] = self.matrix[[index, key_index]]
            self.free_terms[[key_index, index]] = self.free_terms[[index, key_index]]

    def __nullify_column(self, index: int) -> None:
        """Зануляет элементы ниже указанного индекса в заданном столбце."""
        for i in range(index + 1, len(self.matrix)):
            coeff = self.matrix[i][index] / self.matrix[index][index]
            self.matrix[i] -= self.matrix[index] * coeff
            self.free_terms[i] -= self.free_terms[index] * coeff

    def __find_solution_vector(self) -> np.ndarray:
        """Находит результирующий вектор решения системы уравнений."""
        n_rows = self.matrix.shape[0]
        vector = np.zeros(n_rows)
        
        vector[n_rows - 1] = self.free_terms[n_rows - 1] / self.matrix[n_rows - 1][n_rows - 1]
        
        for i in range(n_rows - 2, -1, -1):
            sum_args = sum(self.matrix[i][j] * vector[j] for j in range(i + 1, n_rows))
            vector[i] = (self.free_terms[i] - sum_args) / self.matrix[i][i]
        
        return vector

def find_residual_vector(matrix: np.ndarray, roots: np.ndarray, free_terms: np.ndarray) -> np.ndarray:
    """Находит вектор невязки"""
    return np.dot(matrix, roots) - free_terms

def find_residual_norm(residual_vector: np.ndarray) -> np.floating[Any]:
    """Нормализирует вектор невязки"""
    return np.linalg.norm(residual_vector)

def estimate_relative_error(matrix: np.ndarray, roots: np.ndarray) -> np.floating[Any]:
    """Находит решение вспомогательной системы уравнений и оценивает относительную погрешность"""
    new_free_terms = np.dot(matrix, roots)

    gaussian_solver = GaussianSolver(matrix, new_free_terms)
    new_roots = gaussian_solver.solve_equation_system()

    return np.linalg.norm(roots - new_roots) / np.linalg.norm(roots) 

def main():
    a = np.array([[2.31, 31.49, 1.52], [4.21, 22.42, 3.85], [3.49, 4.85, 28.72]], float)
    b = np.array([40.95, 30.24, 42.81], float)

    gaussian_solver = GaussianSolver(a, b)
    gaussian_solver.print_equation_system()

    x = gaussian_solver.solve_equation_system()
    print("Решение системы уравнений:", x)

    F = find_residual_vector(a, x, b)
    print("Вектор невязки:", F)

    F_norm = find_residual_norm(F)
    print("Норма вектора невязки:", F_norm)

    error = estimate_relative_error(a, x)
    print("Относительная погрешность:", error)

if __name__ == "__main__":
    main()
