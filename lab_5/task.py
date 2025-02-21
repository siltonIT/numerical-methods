from typing import Any, Tuple

import sympy as sp
import numpy as np

from trapezoid import TrapezoidSolver
from sypson import SypsonSolver

def get_vars_name(vars_amount: int) -> Tuple[sp.Symbol, ...]:
    """Возвращает кортеж имен переменных системы уравнений."""
    return sp.symbols(" ".join([f"x{i + 1}" for i in range(vars_amount)]))

def build_func() -> Tuple[str, Any]:
    """Создает функцию по пользовательскому вводу"""
    vars_amount = int(input("Введите количество переменных: "))
    func_str = input("Введите функцию: ")

    return func_str, sp.lambdify(get_vars_name(vars_amount), func_str, "numpy")

def main(): 
    func_str, func = build_func()
    a, b = input("Введите границы интегрирования функции одной переменной: ").split(" ")
    
    print(
            "Метод трапеций:",
            f"Результат интегрирования для функции {func_str}:",
            f"res: {TrapezoidSolver.solve(func, float(a), float(b))}\n",
            sep="\n"
         )

    print(
            "Метод симпосона:",
            f"Результат интегрирования для функции {func_str}:",
            f"res: {SypsonSolver.solve(func, float(a), float(b))}\n",
            sep="\n"
         )
    
    func_str, func = build_func()
    a, b, c, d = input("Введите границы интегрирования функции двух переменных: ").split(" ")

    print(
            "Метод симпосона:",
            f"Результат интегрирования для функции {func_str}:",
            f"res: {SypsonSolver.dsolve(func, float(a), float(b), float(c), float(d))}",
            sep="\n"
         )
    
if __name__ =="__main__":
    main()
