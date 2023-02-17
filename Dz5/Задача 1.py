
import iapws
from iapws import IAPWS97 as gas
MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
to_kelvin = lambda x: x + 273.15 if x else None
electrical_power = 250 * (10 ** 6)
p0 = 23.5 * MPa
t0 = 540
pk = 6.9 * kPa
t_feed_water = 263
p_feed_water = 1.4 * p0
z = 8

internal_efficiency = 0.85
mechanical_efficiency = 0.992
generator_efficiency = 0.99

delta_p0 = 0.05 * p0

real_p0 = p0 - delta_p0
def find_points(p0, real_p0, t0, pk, p_feed_water, t_feed_water):
    point_0t = gas(P=p0*unit, T=to_kelvin(t0))
    point_0 = gas(P=real_p0 * unit, h=point_0t.h)
    
    point_2t = gas(P=pk * unit, s=point_0t.s)

    point_k_water = gas(P=pk * unit, x=0)
    point_feed_water = gas(P=p_feed_water * unit, T=to_kelvin(t_feed_water))

    heat_drop = (point_0t.h - point_2t.h) * internal_efficiency
    h_2 = point_0.h - heat_drop
    point_2 = gas(P=pk * unit, h=h_2)
    
    return point_0, point_0t, point_2, point_2t, point_k_water, point_feed_water
def calculate_ksi(point_0, point_0t, point_2, point_k_water, point_feed_water, z):
    numenator_without = point_2.T * (point_0t.s - point_k_water.s)
    denumenator_without = (point_0.h - point_k_water.h)
    without_part = 1 - (numenator_without / denumenator_without)
    
    numenator_infinity = point_2.T * (point_0t.s - point_feed_water.s)
    denumenator_infinity = (point_0.h - point_feed_water.h)
    infinity_part = 1 - (numenator_infinity / denumenator_infinity)

    ksi_infinity = 1 - (without_part / infinity_part)

    coeff = (point_feed_water.T - point_2.T) / (to_kelvin(374.2) - point_2.T)
    print("Значение коэфициетнта", coeff)
    print("Число оборотов", z)
    draf = float(input("По значению коэфициетта введите значения с графика "))
    ksi = draf * ksi_infinity
    return ksi
def calculate_estimated_heat_drop(point_0, point_0t, point_2t, point_k_water, point_feed_water,ksi):
    eff_num = (point_0t.h - point_2t.h) * internal_efficiency  
    eff_denum = (point_0.h - point_k_water.h)

    efficiency = (eff_num / eff_denum) * (1 / (1 - ksi))
    print ("КПД",round(efficiency*100 ,4), "%")

    estimated_heat_drop = efficiency * ((point_0.h - point_feed_water.h))
    return estimated_heat_drop, efficiency
def calculate_mass_flow(estimated_heat_drop, electrical_power, mechanical_efficiency, generator_efficiency, efficiency):   
    inlet_mass_flow = electrical_power / (estimated_heat_drop * 1000 * mechanical_efficiency * generator_efficiency)

    condenser_mass_flow = (
    electrical_power /
    ((point_2.h - point_k_water.h) * 1000 * mechanical_efficiency * generator_efficiency) * ((1 / efficiency) - 1))
    
    print("Массовый расход в турбину на входе", inlet_mass_flow)
    print("Массовый расход в конденсатор:", condenser_mass_flow)
point_0, point_0t, point_2, point_2t, point_k_water, point_feed_water = find_points(p0, real_p0, t0, pk, p_feed_water, t_feed_water)

ksi = calculate_ksi(point_0, point_0t, point_2, point_k_water, point_feed_water, z)

estimated_heat_drop, efficiency = calculate_estimated_heat_drop(point_0, point_0t, point_2t, point_k_water, point_feed_water,ksi)

calculate_mass_flow(estimated_heat_drop, electrical_power, mechanical_efficiency, generator_efficiency, efficiency)
