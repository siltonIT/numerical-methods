from typing import Any

import numpy as np

class TrapezoidSolver:
    @staticmethod
    def solve(
            func: Any, 
            a: float, 
            b: float, 
            n: int = 100, 
            epsilon: float = 1e-5
        ) -> float:
        """Интегрирует функцию методом трапеций"""
        prev_res = 0.0
        for i in range(1, n + 1):
            step = (b - a) / 2**i
            x = np.linspace(a, b, 2**i + 1)
            y = func(x)

            res = step / 2 * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
            if np.abs(res - prev_res) <= 3 * epsilon:
                break
            prev_res = res
        return res

def main():
    a, b = 0.0, np.pi/2
    func = lambda x: np.sin(x)
    
    res = TrapezoidSolver.solve(func, a, b, n=10)
    print(
            "Результат для функции sin(x):",
            f"a: {a}",
            f"b: {b}",
            f"res: {res}",
            sep="\n"
          )

if __name__ == "__main__":
    main()
