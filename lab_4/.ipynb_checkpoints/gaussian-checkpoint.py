import numpy as np
import sympy as sp
from typing import Tuple, Any

class GaussianSolver:
    @staticmethod
    def print_equation_system(matrix: np.ndarray, free_terms: np.ndarray) -> None:
        """Выводит систему уравнений в читаемом формате."""
        print("Система уравнений:")
        for i, row in enumerate(matrix):
            print(f"{row} | {free_terms[i]}")
        print()


    @staticmethod
    def build_polynomial(
            x: np.ndarray,
            y: np.ndarray,
            polynomial_degree: int = 2
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Создает полином n-ой степени"""
        matrix = np.zeros((polynomial_degree + 1, polynomial_degree + 1))
        free_terms = np.zeros(polynomial_degree + 1)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                matrix[i, j] = np.sum(x**(i + j))

        for i in range(free_terms.shape[0]):
            free_terms[i] = np.sum(x**i * y)
        
        return matrix, free_terms


    @staticmethod
    def get_polynomial_func(coeffs: np.ndarray, polinomial_degree: int = 2) -> Any:
        """Создает функцию полинома n-ой степени"""
        polynomyal_str = " + ".join([f"{coeffs[i]}*x**{i}" for i in range(polinomial_degree + 1)])
        x = sp.symbols("x")

        return sp.lambdify(x, polynomyal_str, "numpy")


    @staticmethod
    def solve(matrix: np.ndarray, free_terms: np.ndarray) -> np.ndarray:
        """Находит решение системы линейных уравнений."""
        GaussianSolver.__transform_to_upper_triangular_matrix(matrix, free_terms)
        return GaussianSolver.__find_solution_vector(matrix, free_terms)


    @staticmethod
    def __transform_to_upper_triangular_matrix(matrix: np.ndarray, free_terms: np.ndarray) -> None:
        """Приводит матрицу к верхнетреугольному виду."""
        for i in range(matrix.shape[0] - 1):
            GaussianSolver.__find_pivot_element(matrix, free_terms, i)
            GaussianSolver.__nullify_column(matrix, free_terms, i)


    @staticmethod
    def __find_pivot_element(matrix: np.ndarray, free_terms: np.ndarray, index: int) -> None:
        """Находит ключевой элемент в матрице и меняет его местами с элементом в строке index."""
        key_index = index + np.argmax(matrix[index:, index])
        
        if index != key_index:
            matrix[[key_index, index]] = matrix[[index, key_index]]
            free_terms[[key_index, index]] = free_terms[[index, key_index]]


    @staticmethod
    def __nullify_column(matrix: np.ndarray, free_terms: np.ndarray, index: int) -> None:
        """Зануляет элементы ниже указанного индекса в заданном столбце."""
        for i in range(index + 1, matrix.shape[0]):
            coeff = matrix[i][index] / matrix[index][index]
            matrix[i] -= matrix[index] * coeff
            free_terms[i] -= free_terms[index] * coeff


    @staticmethod
    def __find_solution_vector(matrix: np.ndarray, free_terms: np.ndarray) -> np.ndarray:
        """Находит результирующий вектор решения системы уравнений."""
        n_rows = matrix.shape[0]
        vector = np.zeros(n_rows)
        
        vector[n_rows - 1] = free_terms[n_rows - 1] / matrix[n_rows - 1][n_rows - 1]
        
        for i in range(n_rows - 2, -1, -1):
            sum_args = sum(matrix[i][j] * vector[j] for j in range(i + 1, n_rows))
            vector[i] = (free_terms[i] - sum_args) / matrix[i][i]
        
        return vector


def main():
    a = np.array([[2.31, 31.49, 1.52], [4.21, 22.42, 3.85], [3.49, 4.85, 28.72]], float)
    b = np.array([40.95, 30.24, 42.81], float)

    x = GaussianSolver.solve(a, b)
    print("Решение системы уравнений:", x)

if __name__ == "__main__":
    main()
