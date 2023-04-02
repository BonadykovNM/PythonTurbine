import iapws
from iapws import IAPWS97 as gas
#Инициализация параметров расчета  
MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
to_kelvin = lambda x: x + 273.15 if x else None

def real_point(p0,p_middle): 
        delta_p0 = 0.05 *  p0
        delta_p_middle = 0.1 *  p_middle
        delta_p_1 = 0.03 *  p_middle

        real_p0 =  p0 - delta_p0
        real_p1t =  p_middle + delta_p_middle
        real_p_middle =  p_middle - delta_p_1
        return real_p0,real_p1t,real_p_middle

def find_point0(p0,t0,p_middle):  #Функция поиска теоретического и действительного значения начальной точки 
        real_points = real_point(p0,p_middle)
        point_0t = gas(P =  p0 * unit, T=to_kelvin(t0))
        point_0 = gas(P= real_points[0] * unit, h=point_0t.h)
        return point_0t, point_0
    
def find_point1(p0,t0,p_middle,internal_efficiency):  #Функция поиска теоретического и действительного значения первой точки
        real_points = real_point(p0,p_middle)
        point0 = find_point0(p0,t0,p_middle)
        point_1t = gas(P= real_points[1] * unit, s= point0[0].s)
        hp_heat_drop = (point0[0].h - point_1t.h) *  internal_efficiency
        h_1 =  point0[1].h - hp_heat_drop
        point_1 = gas(P= real_points[1] * unit, h=h_1)
        return point_1t, point_1
    
def find_point_middle(p_middle,t_middle,p0): #Функция поиска теоретического и действительного значения точки пром перергрева 
        real_points = real_point(p0,p_middle)
        point_middle_t = gas(P= p_middle * unit, T=to_kelvin( t_middle))
        point_middle = gas(P= real_points[2] * unit, h=point_middle_t.h)
        return point_middle_t, point_middle
    
def find_point2(p_middle,t_middle,p0,pk,internal_efficiency): #Функция поиска теоретического и действительного значения точки конденсатора 
        point_middle = find_point_middle(p_middle,t_middle,p0)
        point_2t = gas(P= pk * unit, s= point_middle[0].s)
        lp_heat_drop = (point_middle[0].h - point_2t.h) *  internal_efficiency
        h_2 = point_middle[1].h - lp_heat_drop
        point_2 = gas(P= pk * unit, h=h_2)
        return point_2t, point_2

def find_point_water(pk,p_feed_water,t_feed_water): 
        point_k_water = gas(P= pk * unit, x=0)
        point_feed_water = gas(P= p_feed_water * unit, T=to_kelvin(t_feed_water))
        return point_k_water, point_feed_water

def coeff (pk,p_feed_water,t_feed_water,internal_efficiency,p_middle,t_middle,p0):
        point_water = find_point_water(pk,p_feed_water,t_feed_water)
        point2 = find_point2(p_middle,t_middle,p0,pk,internal_efficiency)
        coeff = (( point_water[1].T -  point2[1].T) / (to_kelvin(374.2) -  point2[1].T))
        cf = 0.9  
        return cf

def complite_ksi_infinity(p0,t0,p_middle,t_middle,pk,p_feed_water,t_feed_water,internal_efficiency):
        point2 = find_point2(p_middle,t_middle,p0,pk,internal_efficiency)
        point_middle = find_point_middle(p_middle,t_middle,p0)
        point_water = find_point_water(pk,p_feed_water,t_feed_water)
        point0 = find_point0(p0,t0,p_middle)
        point1 = find_point1(p0,t0,p_middle,internal_efficiency)
        coef = coeff (pk,p_feed_water,t_feed_water,internal_efficiency,p_middle,t_middle,p0)
        numenator_without = point2[1].T * ( point_middle[0].s -  point_water[0].s)
        denumenator_without = (point0[1].h - point1[0].h) + (point_middle[1].h - point_water[0].h)
        without_part = 1 - (numenator_without / denumenator_without)
        numenator_infinity =  point2[1].T * (point_middle[0].s -  point_water[1].s)
        denumenator_infinity = (point0[1].h - point1[0].h) + (point_middle[1].h - point_water[1].h)
        infinity_part = 1 - (numenator_infinity / denumenator_infinity)
        ksi_infinity = 1 - (without_part / infinity_part)
        ksi = coef * ksi_infinity
        return ksi

def complite_estimated_heat_drop(p0,t0,p_middle,t_middle,pk,p_feed_water,t_feed_water,internal_efficiency):
        point0 = find_point0(p0,t0,p_middle)
        point1 = find_point1(p0,t0,p_middle,internal_efficiency)
        point2 = find_point2(p_middle,t_middle,p0,pk,internal_efficiency)
        point_middle = find_point_middle(p_middle,t_middle,p0)
        point_water = find_point_water(pk,p_feed_water,t_feed_water)
        ksi = complite_ksi_infinity(p0,t0,p_middle,t_middle,pk,p_feed_water,t_feed_water,internal_efficiency)
        eff_num = ( point0[0].h - point1[0].h) *  internal_efficiency + (point_middle[0].h - point2[0].h) *  internal_efficiency
        eff_denum = ( point0[0].h - point1[0].h) *  internal_efficiency + (point_middle[1].h -  point_water[0].h)
        efficiency = (eff_num / eff_denum) * (1 / (1 - ksi))
        estimated_heat_drop = efficiency * (( point0[0].h - point_water[1].h) + (point_middle[1].h - point1[1].h))
        return estimated_heat_drop , efficiency

def complite_inlet_mass_flow(p0,t0,p_middle,t_middle,pk,p_feed_water,t_feed_water,internal_efficiency,mechanical_efficiency,generator_efficiency,electrical_power): #Расчет расхода в турбину на входе
        estimated_heat_drop = complite_estimated_heat_drop(p0,t0,p_middle,t_middle,pk,p_feed_water,t_feed_water,internal_efficiency)
        inlet_mass_flow =  electrical_power / ( estimated_heat_drop[0] * 1000 *  mechanical_efficiency *  generator_efficiency)
        return inlet_mass_flow

def complite_condenser_mass_flow(p0,t0,p_middle,t_middle,pk,p_feed_water,t_feed_water,internal_efficiency,mechanical_efficiency,generator_efficiency,electrical_power): #Расчет расхода в конденсатор
        estimated_heat_drop = complite_estimated_heat_drop(p0,t0,p_middle,t_middle,pk,p_feed_water,t_feed_water,internal_efficiency)
        point2 = find_point2(p_middle,t_middle,p0,pk,internal_efficiency)
        point_water = find_point_water(pk,p_feed_water,t_feed_water)
        condenser_mass_flow = ( electrical_power / ((point2[1].h - point_water[0].h) * 1000 *  mechanical_efficiency *  generator_efficiency) * ((1 / estimated_heat_drop[1]) - 1))
        return condenser_mass_flow