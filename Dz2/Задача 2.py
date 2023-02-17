#Задача 2
#Написать код для решения задачи оптимизации параметров промежуточного перегрева P_пп и t_пп для свободных начальных параметров p_0,t_0, p_к. Сделаем допущение, что начальная точка процесса расширения всегда находится в зоне перегретого пара. Решение должно выдавать параметры промежуточного перегрева и термический КПД при них.

import iapws
import array
from typing import Optional, Tuple, List, Union
from iapws import IAPWS97 as gas
from scipy.optimize import minimize
import numpy as np
point_type = iapws.iapws97.IAPWS97
##Упростим себе жизнь
MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
to_kelvin = lambda x: x + 273.15 if x else None
def check_is_valid_numerical(values: List[Union[None, float]]) -> None:
    for value in values:
        if value:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Argument should be int or float value! Given {type(value)}")
def construct_cycle_points(
    p_0: Union[int, float],
    p_k: Union[int, float],
    p_middle: Union[int, float],
    t_0: Optional[Union[int, float]] = None,
    t_middle: Optional[Union[int, float]] = None,
    x_0: Optional[Union[int, float]] = None,
) -> Tuple[point_type, point_type, point_type]:
    check_is_valid_numerical([p_0, p_k, p_middle, t_0, t_middle, x_0])
    if not t_0 and not x_0:
        raise ValueError("Neither t_0 not x_0 is not provided for start expansion point!")
    if not (p_0 and p_k):
        raise ValueError("P_0 and p_k must be specified!")
    if x_0 and (x_0 > 1 or x_0 < 0):
        raise ValueError("x_0 should be between 0 and 1")
        
    point_0_start = gas(P=p_0 * unit, T=t_0)
    point_0_end = gas(P=p_middle * unit, s=point_0_start.s)
    point_1_start = gas(P=p_middle * unit, T=t_middle)
    point_condenser = gas(P=p_k * unit, s=point_1_start.s)
    point_pump = gas(P=p_k * unit, x=0)
    
    return point_0_start, point_0_end, point_1_start, point_condenser, point_pump

def compute_cycle_efficiency(point_0_start: point_type, point_0_end: point_type, point_1_start: point_type, point_condenser: point_type, point_pump: point_type) -> float:
    useful_energy = (point_0_start.h - point_0_end.h) + (point_1_start.h - point_condenser.h)
    full_energy = (point_0_start.h - point_pump.h) + (point_1_start.h - point_0_end.h)
    efficiency = useful_energy/full_energy
    return efficiency
def solve_exercise(
    p_0: Union[int, float],
    p_k: Union[int, float],
    p_middle: Union[int, float],
    t_0: Optional[Union[int, float]],
    t_middle: Optional[Union[int, float]],
    x_0: Optional[Union[int, float]] = None,) -> float:
    point_0_start, point_0_end, point_1_start, point_condenser, point_pump = construct_cycle_points(
        p_0=p_0,
        p_k=p_k,
        p_middle=p_middle,
        t_0=to_kelvin(t_0),
        t_middle=to_kelvin(t_middle),
        x_0=x_0
    )
    
    efficiency = compute_cycle_efficiency(point_0_start=point_0_start, point_0_end=point_0_end, point_1_start=point_1_start, point_condenser=point_condenser, point_pump=point_pump)
    
    return efficiency
 
def optinal_params( p_0: Union[int, float],t_0: Union[int, float], p_k: Union[int, float]):
    def loss_function(middle_params, p_0 = p_0, t_0 = t_0, p_k = p_k):
        p_div_p_0, t_div_t_0 = middle_params[0], middle_params[1]
        p_middle = p_0 * p_div_p_0
        t_middle = t_0 * t_div_t_0
        efficiency = solve_exercise(p_0=p_0, p_k=p_k, p_middle = p_middle, t_0 = t_0 , t_middle = t_middle)
        return 1 - efficiency
    initial_params = np.array([0.5, 0.5])
    bounds = ([0.01, 1], [0.01, 1])
    result = minimize(loss_function, x0 = initial_params, bounds = bounds, tol = 1e-8)
    p_div_p_0, t_div_t_0 = result.x
    p_middle = p_div_p_0 * p_0
    t_middle = t_div_t_0 * t_0
    print("Superheating pressure",p_middle*unit,"MPa")
    print("Temperature of industrial superheating",t_middle,"degrees celsius")
    return p_middle, t_middle
p_0 = float(input("Enter p_0 in MPa "))
p_0 = p_0 * MPa
t_0 = float(input("Enter t_0 in degrees celsius ")) 
p_k = float(input("Enter p_k in kPa "))
p_k = p_k * kPa
rez = (optinal_params(p_0,t_0,p_k))
