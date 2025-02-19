from typing import Callable, Tuple

import numpy as np
import sympy as sp

class NewtonSolver:
    @staticmethod
    def solve(
            system: Callable[..., np.ndarray],
            jacobian: Callable[..., np.ndarray],
            roots: np.ndarray,
            epsilon: float = 1e-9,
            max_iter: int = 100,
            use_finite_diff: bool = False,
            M: float = 0.01
        ) -> np.ndarray:
        """Решает систему нелинейных уравнений методом Ньютона."""
        for i in range(max_iter):
            system_val = np.array(system(*roots), dtype=float).flatten()

            if not use_finite_diff:
                jacobian_val = np.array(jacobian(*roots), dtype=float)
            else:
                jacobian_val = get_finite_difference_jacobian(system, roots, M)

            try:
                delta_roots = np.linalg.solve(jacobian_val, -system_val)
            except np.linalg.LinAlgError:
                raise ValueError("Якобиан вырожден.")

            new_roots = roots + delta_roots

            norm_system = np.max(np.abs(system_val))
            norm_delta_roots = np.max(np.abs(delta_roots) / np.maximum(np.abs(new_roots), 1))

            print(
                f"Итерация {i+1}:",
                f"delta_1: {norm_system}",
                f"delta_2: {norm_delta_roots}",
                f"x: {new_roots}\n",
                sep="\n"
            )

            if norm_system < epsilon and norm_delta_roots < epsilon:
                return new_roots

            roots = new_roots

        raise ValueError(f"Метод Ньютона не сошелся за {max_iter} итераций.")

    
def get_initial_roots_approximation() -> Tuple[np.ndarray, int]:
    """Запрашивает у пользователя количество переменных и их начальное приближение."""
    vars_amount = int(input("Введите количество переменных: "))

    roots = np.zeros(vars_amount, dtype=float)
    for i in range(vars_amount):
        roots[i] = float(input(f"Введите x{i + 1}: "))

    return roots, vars_amount


def get_vars_name(vars_amount: int) -> Tuple[sp.Symbol, ...]:
    """Возвращает кортеж имен переменных системы уравнений."""
    return sp.symbols(" ".join([f"x{i + 1}" for i in range(vars_amount)]))


def get_equations_system(vars_amount: int) -> sp.Matrix:
    """Запрашивает у пользователя систему уравнений и возвращает её в виде матрицы."""
    equations = []
    for i in range(vars_amount):
        equation = input(f"Введите {i + 1}-е уравнение: ")
        equations.append(sp.sympify(equation))

    return sp.Matrix(equations)


def get_analytical_jacobian(system: sp.Matrix, vars_name: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    """Вычесляет Якобиан аналитически."""
    return system.jacobian(vars_name)


def get_finite_difference_jacobian(
        system_func: Callable[..., np.ndarray], 
        roots: np.ndarray, 
        M: float = 0.01
    ) -> np.ndarray:
    """Вычисляет Якобиан численно с использованием метода конечных разностей."""
    roots_amount = len(roots)
    jacobian = np.zeros((roots_amount, roots_amount))
    
    for i in range(roots_amount):
        for j in range(roots_amount):
            h = M * max(np.abs(roots[i]), np.abs(roots[j]), 1e-6)
            temp_roots = roots.copy() 
            temp_roots[j] += h 
            
            temp_system_val = system_func(*temp_roots).flatten()
            system_val = system_func(*roots).flatten()
            jacobian[i, j] = (temp_system_val[i] - system_val[i]) / h
    
    return jacobian


def turn_to_numpy_func(vars_name: Tuple[sp.Symbol, ...], func: sp.Matrix) -> Callable:
    """Преобразует sympy-функции в числовые и возвращает их."""
    return sp.lambdify(vars_name, func, "numpy")


def main():
    roots, vars_amount = get_initial_roots_approximation()
    vars_name = get_vars_name(vars_amount)
    system = get_equations_system(vars_amount)
    jacobian = get_analytical_jacobian(system, vars_name)

    jacobian_func = turn_to_numpy_func(vars_name, jacobian)
    system_func = turn_to_numpy_func(vars_name, system)

    roots = NewtonSolver.solve(system_func, jacobian_func, roots)
    print(
        f"Аналитический Якобиан:",
        f"Решение системы: {roots}\n",
        sep="\n"
    )

    print("Якобиан с использованием метода конечных разностей")

    roots = NewtonSolver.solve(system_func, jacobian_func, roots, use_finite_diff=True, M=0.01)
    print(f"Решение системы при M = 0.01: {roots}\n")

    roots = NewtonSolver.solve(system_func, jacobian_func, roots, use_finite_diff=True, M=0.05)
    print(f"Решение системы при M = 0.05: {roots}\n")

    roots = NewtonSolver.solve(system_func, jacobian_func, roots, use_finite_diff=True, M=0.1)
    print(f"Решение системы при M = 0.1: {roots}\n")

if __name__ == "__main__":
    main()
