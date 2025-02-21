from typing import Any

import numpy as np

class SypsonSolver:
    @staticmethod
    def solve(
            func: Any, 
            a: float, 
            b: float, 
            n: int = 100, 
            epsilon: float = 1e-5
        ) -> float:
        """Интегрирует функцию одной переменной методом Симпсона"""
        if n % 2 == 1:
            n += 1
        
        prev_res = 0
        for i in range(1, n + 1):
            step = (b - a) / 2**i
            x = np.linspace(a, b, 2**i + 1)
            y = func(x)

            res = step / 3* (y[0] + 4 * np.sum(y[1:2**i + 1:2]) + 2 * np.sum(y[2:2**i:2]) + y[-1])
            if np.abs(res - prev_res) <= 15 * epsilon:
                break
            prev_res = res
        
        return res     

    @staticmethod
    def dsolve(
            func: Any,
            a: float,
            b: float,
            c: float,
            d: float,
            n: int = 100,
            m: int = 100
        ) -> float:
        """Интегрирует функцию двух пременных методом Симпсона"""
        if n % 2 == 1:
            n += 1
        if m %  2 == 1:
            m += 1

        x = np.linspace(a, b, n)
        y = np.linspace(c, d, m)
        x_step = (b - a) / (n - 1)
        y_step = (d - c) / (m - 1)
        
        res = 0
        for i in range(0, n-2, 2):
            for j in range(0, m-2, 2):
                res += (
                    func(x[i], y[j]) + 4*func(x[i+1], y[j]) + func(x[i+2], y[j]) +
                    4*func(x[i], y[j+1]) + 16*func(x[i+1], y[j+1]) + 4*func(x[i+2], y[j+1]) +
                    func(x[i], y[j+2]) + 4*func(x[i+1], y[j+2]) + func(x[i+2], y[j+2])
                )
    
        return x_step * y_step * res / 9

def main():
    a, b = 0.0, np.pi/2
    func = lambda x: np.sin(x)
    
    res = SypsonSolver.solve(func, a, b)
    print(
            "Результат для функции sin(x):",
            f"a: {a}, b: {b}",
            f"res: {res}\n",
            sep="\n"
          )
    
    c, d = 0.0, np.pi/2
    func = lambda x, y: np.sin(x) + np.sin(y) 
    
    res = SypsonSolver.dsolve(func, a, b, c, d)
    print(
            "Результат для функции sin(x) + sin(y):",
            f"a: {a}, b: {b}",
            f"c: {c}, d: {d}",
            f"res: {res}",
            sep="\n"
         )

if __name__ == "__main__":
    main()
