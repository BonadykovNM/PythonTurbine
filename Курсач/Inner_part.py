from typing import List, Tuple, Optional
from iapws import IAPWS97
from iapws import IAPWS97 as gas
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd 
import os
import math

#Инициализация параметров расчета  
MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
to_kelvin = lambda x: x + 273.15 if x else None


def real_point(p0, p_middle):
        """
        Функция поиска действительного значения точек цикла
        :param float p0: Давление начальной точки в Па
        :param float p_middle: Давление промежуточного перегрева в Па
        :return Tuple[IAPWS97, IAPWS97,IAPWS97]: Действительно значения давления в начальной точке, первой точке и в точке пром. перегрева в Па
        """
        delta_p0 = 0.05 *  p0
        delta_p_middle = 0.1 *  p_middle
        delta_p_1 = 0.03 *  p_middle

        real_p0 =  p0 - delta_p0
        real_p1t =  p_middle + delta_p_middle
        real_p_middle =  p_middle - delta_p_1
        return real_p0,real_p1t,real_p_middle


def find_point0(p0, t0, p_middle):
        """
        Функция поиска теоретического и действительного значения начальной точки 
        :param float p0: Давление начальной точки в Па
        :param float t0: Температура начальной точки в К
        :param float p_middle: Давление промежуточного перегрева в Па
        :return Tuple[IAPWS97, IAPWS97,IAPWS97]:Параметры теоретического значения начальной точки, Параметры действительного значения начальной точки
        """ 
        real_points = real_point(p0, p_middle)
        point_0t = gas(P =  p0 * unit, T=to_kelvin(t0))
        point_0 = gas(P= real_points[0] * unit, h=point_0t.h)
        return point_0t, point_0
    

def find_point1(p0, t0, p_middle, internal_efficiency):  
        real_points = real_point(p0, p_middle)
        point0 = find_point0(p0, t0, p_middle)
        point_1t = gas(P= real_points[1] * unit, s= point0[0].s)
        hp_heat_drop = (point0[0].h - point_1t.h) *  internal_efficiency
        h_1 =  point0[1].h - hp_heat_drop
        point_1 = gas(P= real_points[1] * unit, h=h_1)
        return point_1t, point_1
    

def find_point_middle(p_middle, t_middle, p0): 
        """
        Функция поиска теоретического и действительного значения точки пром перергрева 
        :param float p_middle: Давление промежуточного перегрева в Па
        :param float t_middle: Температура промежуточного перегрева в К
        :param float p0: Давление начальной точки в Па
        :return Tuple[IAPWS97, IAPWS97,IAPWS97]:Параметры теоретического значения точки пром. перегрева, Параметры действительного значения точки пром. перегрева
        """ 
        real_points = real_point(p0, p_middle)
        point_middle_t = gas(P= p_middle * unit, T=to_kelvin( t_middle))
        point_middle = gas(P= real_points[2] * unit, h=point_middle_t.h)
        return point_middle_t, point_middle
    

def find_point2(p_middle, t_middle, p0, pk, internal_efficiency): 
        """
        Функция поиска теоретического и действительного значения точки конденсатора
        :param float p_middle: Давление промежуточного перегрева в Па
        :param float t_middle: Температура промежуточного перегрева в К
        :param float p0: Давление начальной точки в Па
        :param float pk: Давление точки конденсатора в Па
        :param float internal_efficiency: КПД цилиндров турбины
        :return Tuple[IAPWS97, IAPWS97,IAPWS97]:Параметры теоретического значения точки конденсатора, Параметры действительного значения точки конденсатора
        """
        point_middle = find_point_middle(p_middle, t_middle, p0)
        point_2t = gas(P= pk * unit, s= point_middle[0].s)
        lp_heat_drop = (point_middle[0].h - point_2t.h) *  internal_efficiency
        h_2 = point_middle[1].h - lp_heat_drop
        point_2 = gas(P= pk * unit, h=h_2)
        return point_2t, point_2


def find_point_water(pk, p_feed_water, t_feed_water):
        """
        Функция поиска параментров воды после конденсатора и памаметров питательной воды
        :param float pk: Давление точки конденсатора в Па
        :param float p_feed_water: Давление питательной воды в Па
        :param float t_feed_water: Температура питательной воды в К
        :return Tuple[IAPWS97, IAPWS97,IAPWS97]:Параметры воды после конденсатора, Параметры питательной воды
        """        
        point_k_water = gas(P= pk * unit, x=0)
        point_feed_water = gas(P= p_feed_water * unit, T=to_kelvin(t_feed_water))
        return point_k_water, point_feed_water


def check_graf(z,coef):
        """
        Функция поска значения коэффициента по графику
        :param integer z: количество регенеративных подогревателей
        :param float coef: значение коефициента для графика
        :return float: Значение коэфициента 
        """ 
        if z > 7:
            z = 10
        xl = pd.read_excel('data.xlsx',sheet_name =f'z = {z}')
        coef_ = pd.DataFrame(xl,columns=['coef'])
        y_ = pd.DataFrame(xl,columns=['y'])
        ks = []
        y = []
        for vl in coef_:
            for value in coef_[vl]:
                ks.append(value)
        for vl in y_:
                for value in y_[vl]:
                    y.append(value)
        f_y = np.interp(coef,ks,y)
        #plt.plot(ks, y)
        #if z > 7:
            #plt.plot(coef, f_y,label = 'z = ∞')
        #else:
            #plt.plot(coef, f_y,label = f'z = {z}')
        #plt.legend()
        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.grid()
        #plt.show()
        return f_y


def coeff (pk, p_feed_water, t_feed_water, internal_efficiency, p_middle, t_middle, p0, z):
        """
        Функция расчета коэфициента
        :param float pk: Давление точки конденсатора в Па
        :param float p_feed_water: Давление питательной воды в Па
        :param float t_feed_water: Температура питательной воды в К
        :param float internal_efficiency: КПД цилиндров турбины
        :param float p_middle: Давление промежуточного перегрева в Па
        :param float t_middle: Температура промежуточного перегрева в К
        :param float p0: Давление начальной точки в Па
        :param float z: число регениративных подогревателей 
        :return float: Значение коэфициента
        """  
        point_water = find_point_water(pk, p_feed_water, t_feed_water)
        point2 = find_point2(p_middle, t_middle, p0, pk, internal_efficiency)
        coeff = (( point_water[1].T -  point2[1].T) / (to_kelvin(374.2) -  point2[1].T))
        #print("Коэфициент для нахождения коэфициента", coeff)
        ###
        ###image = mpimg.imread("graf.png")
        ###plt.axis('off')
        ###plt.imshow(image)
        ###plt.show()
        cf = check_graf(z,coeff)
        ###print (cf)
        return cf


def complite_ksi_infinity(p0,
                          t0,
                          p_middle,
                          t_middle,
                          pk,
                          p_feed_water,
                          t_feed_water,
                          internal_efficiency,
                          z
                          ):
        """
        Функция расчета коэфициента ksi
        :param float p0: Давление начальной точки в Па
        :param float p_middle: Давление промежуточного перегрева в Па
        :param float t_middle: Температура промежуточного перегрева в К
        :param float pk: Давление точки конденсатора в Па
        :param float p_feed_water: Давление питательной воды в Па
        :param float t_feed_water: Температура питательной воды в К
        :param float internal_efficiency: КПД цилиндров турбины
        :param float z: число регениративных подогревателей
        :return float: Значение коэфициента ksi
        """ 

        point2 = find_point2(p_middle, t_middle, p0, pk, internal_efficiency)
        point_middle = find_point_middle(p_middle, t_middle, p0)
        point_water = find_point_water(pk, p_feed_water, t_feed_water)
        point0 = find_point0(p0, t0, p_middle)
        point1 = find_point1(p0, t0, p_middle, internal_efficiency)
        coef = coeff (pk, p_feed_water, t_feed_water, internal_efficiency, p_middle, t_middle, p0, z)
        numenator_without = point2[1].T * ( point_middle[0].s -  point_water[0].s)
        denumenator_without = (point0[1].h - point1[0].h) + (point_middle[1].h - point_water[0].h)
        without_part = 1 - (numenator_without / denumenator_without)
        numenator_infinity =  point2[1].T * (point_middle[0].s -  point_water[1].s)
        denumenator_infinity = (point0[1].h - point1[0].h) + (point_middle[1].h - point_water[1].h)
        infinity_part = 1 - (numenator_infinity / denumenator_infinity)
        ksi_infinity = 1 - (without_part / infinity_part)
        ksi = coef * ksi_infinity
        return ksi


def complite_estimated_heat_drop(p0,
                                 t0,
                                 p_middle,
                                 t_middle,
                                 pk,
                                 p_feed_water,
                                 t_feed_water,
                                 internal_efficiency,
                                 z
                                 ):
        """
        Функция расчета теплоперепада и эффективности 
        :param float p0: Давление начальной точки в Па
        :param float t0: Температура начальной точки в К
        :param float p_middle: Давление промежуточного перегрева в Па
        :param float t_middle: Температура промежуточного перегрева в К
        :param float pk: Давление точки конденсатора в Па
        :param float p_feed_water: Давление питательной воды в Па
        :param float t_feed_water: Температура питательной воды в К
        :param float internal_efficiency: КПД цилиндров турбины
        :param float z: число регениративных подогревателей
        :return Tuple[]: Значение теплоперепада, значение эффективности 
        """ 
        point0 = find_point0(p0, t0, p_middle)
        point1 = find_point1(p0, t0, p_middle, internal_efficiency)
        point2 = find_point2(p_middle, t_middle, p0, pk, internal_efficiency)
        point_middle = find_point_middle(p_middle, t_middle,p0)
        point_water = find_point_water(pk, p_feed_water, t_feed_water)
        ksi = complite_ksi_infinity(p0, t0, p_middle, t_middle, pk, p_feed_water, t_feed_water, internal_efficiency, z)
        eff_num = ( point0[0].h - point1[0].h) *  internal_efficiency + (point_middle[0].h - point2[0].h) *  internal_efficiency
        eff_denum = ( point0[0].h - point1[0].h) *  internal_efficiency + (point_middle[1].h -  point_water[0].h)
        efficiency = (eff_num / eff_denum) * (1 / (1 - ksi))
        estimated_heat_drop = efficiency * (( point0[0].h - point_water[1].h) + (point_middle[1].h - point1[1].h))
        return estimated_heat_drop, efficiency


def complite_inlet_mass_flow(p0,
                             t0,
                             p_middle,
                             t_middle,
                             pk,
                             p_feed_water,
                             t_feed_water,
                             internal_efficiency,
                             mechanical_efficiency,
                             generator_efficiency,
                             electrical_power,
                             z
                             ):
        """
        Функция расчета расхода в турбину на входе
        :param float p0: Давление начальной точки в Па
        :param float t0: Температура начальной точки в К
        :param float p_middle: Давление промежуточного перегрева в Па
        :param float t_middle: Температура промежуточного перегрева в К
        :param float pk: Давление точки конденсатора в Па
        :param float p_feed_water: Давление питательной воды в Па
        :param float t_feed_water: Температура питательной воды в К
        :param float internal_efficiency: КПД цилиндров турбины
        :param float mechanical_efficiency:  Механический КПД турбины
        :param float generator_efficiency: КПД электрического генератора
        :param float electrical_power: Электрическая мощность
        :param float z: число регениративных подогревателей
        :return float: Значение расхода в турбину на входе
        """ 
        estimated_heat_drop = complite_estimated_heat_drop(p0,
                                                           t0,
                                                           p_middle,
                                                           t_middle,
                                                           pk,
                                                           p_feed_water,
                                                           t_feed_water,
                                                           internal_efficiency,
                                                           z
                                                           )
        inlet_mass_flow =  electrical_power / ( estimated_heat_drop[0] * 1000 *  mechanical_efficiency *  generator_efficiency)
        return inlet_mass_flow


def complite_condenser_mass_flow(p0,
                                 t0,
                                 p_middle,
                                 t_middle,
                                 pk,
                                 p_feed_water,
                                 t_feed_water,
                                 internal_efficiency,
                                 mechanical_efficiency,
                                 generator_efficiency,
                                 electrical_power,
                                 z
                                 ): 
        """
        Функция расчета расхода в конденсатор
        :param float p0: Давление начальной точки в Па
        :param float t0: Температура начальной точки в К
        :param float p_middle: Давление промежуточного перегрева в Па
        :param float t_middle: Температура промежуточного перегрева в К
        :param float pk: Давление точки конденсатора в Па
        :param float p_feed_water: Давление питательной воды в Па
        :param float t_feed_water: Температура питательной воды в К
        :param float internal_efficiency: КПД цилиндров турбины
        :param float mechanical_efficiency:  Механический КПД турбины
        :param float generator_efficiency: КПД электрического генератора
        :param float electrical_power: Электрическая мощность
        :param float z: число регениративных подогревателей
        :return float: Значение расхода в конденсатор
        """ 
        estimated_heat_drop = complite_estimated_heat_drop(p0,
                                                           t0,
                                                           p_middle,
                                                           t_middle,
                                                           pk,
                                                           p_feed_water,
                                                           t_feed_water,
                                                           internal_efficiency,
                                                           z
                                                           )
        point2 = find_point2(p_middle, t_middle, p0, pk, internal_efficiency)
        point_water = find_point_water(pk,p_feed_water,t_feed_water)
        condenser_mass_flow = (electrical_power / ((point2[1].h - point_water[0].h) * 1000 *  mechanical_efficiency *  generator_efficiency) * ((1 / estimated_heat_drop[1]) - 1))
        return condenser_mass_flow


def point_s(p0, t0, p_middle, t_middle, pk, internal_efficiency):
    """
    Функция расчета расхода в конденсатор
    :param float p0: Давление начальной точки в Па
    :param float t0: Температура начальной точки в К
    :param float p_middle: Давление промежуточного перегрева в Па
    :param float t_middle: Температура промежуточного перегрева в К
    :param float pk: Давление точки конденсатора в Па
    :param float internal_efficiency: КПД цилиндров турбины
    :return float: Tочки процесса
    """
    point0 = find_point0(p0, t0, p_middle)            
    point1 = find_point1(p0, t0, p_middle, internal_efficiency)
    point2 = find_point2(p_middle, t_middle, p0, pk, internal_efficiency)
    point_middle = find_point_middle(p_middle, t_middle, p0)

    return point0, point1, point2, point_middle


def tab_point(p0,t0,p_middle,t_middle,pk,internal_efficiency):
    """
    функция создания DataFrame для точке процесса 
    :param float p0: Давление начальной точки в Па
    :param float t0: Температура начальной точки в К
    :param float p_middle: Давление промежуточного перегрева в Па
    :param float t_middle: Температура промежуточного перегрева в К
    :param float pk: Давление точки конденсатора в Па
    :param float internal_efficiency: КПД цилиндров турбины
    :return DataFrame: Tочки процесса
    """
    point0, point1, point2, point_middle = point_s(p0, t0, p_middle, t_middle, pk,internal_efficiency)
    df1 = pd.DataFrame([(point0[1].P, point1[1].P, point_middle[1].P, point2[1].P),
                        (point0[1].T, point1[1].T, point_middle[1].T, point2[1].T),
                        (point0[1].h, point1[1].h, point_middle[1].h, point2[1].h),
                        (point0[1].s, point1[1].s, point_middle[1].s, point2[1].s)],
                        index=["P, МПа", "T, K", r"h, $\frac{кДж}{кг}$", r"S, $\frac{кДж}{кг * K}$ "], columns=["point 0", "point 1", "point middle", "point 2"])
    return df1


def tab_flow(p0,
            t0,
            p_middle,
            t_middle,
            pk,
            p_feed_water,
            t_feed_water,
            internal_efficiency,
            mechanical_efficiency,
            generator_efficiency,
            electrical_power,
            z
            ):
        """
        функция создания DataFrame для вывода расходов процесса
        :param float p0: Давление начальной точки в Па
        :param float t0: Температура начальной точки в К
        :param float p_middle: Давление промежуточного перегрева в Па
        :param float t_middle: Температура промежуточного перегрева в К
        :param float pk: Давление точки конденсатора в Па
        :param float p_feed_water: Давление питательной воды в Па
        :param float t_feed_water: Температура питательной воды в К
        :param float internal_efficiency: КПД цилиндров турбины
        :param float mechanical_efficiency:  Механический КПД турбины
        :param float generator_efficiency: КПД электрического генератора
        :param float electrical_power: Электрическая мощность
        :param float z: число регениративных подогревателей
        :return DataFrame: Расходы данного процесса
        """
        inlet_mass_flow = complite_inlet_mass_flow(p0,
                             t0,
                             p_middle,
                             t_middle,
                             pk,
                             p_feed_water,
                             t_feed_water,
                             internal_efficiency,
                             mechanical_efficiency,
                             generator_efficiency,
                             electrical_power,
                             z
                             )
        condenser_mass_flow = complite_condenser_mass_flow(p0,
                            t0,
                            p_middle,
                            t_middle,
                            pk,
                            p_feed_water,
                            t_feed_water,
                            internal_efficiency,
                            mechanical_efficiency,
                            generator_efficiency,
                            electrical_power,
                            z
                            )
        d = {
             'Показатель': ["Массовый расход в турбину на входе","Массовый расход в конденсатор",], 
             'Параметры': [r'$G_{0}$', r'$G_{k}$'],
             'Значение': [inlet_mass_flow, condenser_mass_flow]
            }
        df = pd.DataFrame(data=d)
        blankIndex=[''] * len(df)
        df.index=blankIndex
        display(df.transpose())
        pass


def legend_without_duplicate_labels(ax: plt.Axes) -> None:
    """
    Убирает дубликаты из легенды графика
    :param plt.Axes ax: AxesSubplot с отрисованными графиками
    :return None:
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

    
def plot_process(ax: plt.Axes, points: List[IAPWS97], **kwargs) -> None:
    """
    Отрисовка процесса расширения по точкам
    :param plt.Axes ax: AxesSubplot с отрисованными графиками
    :param List[IAPWS97] points: Список инициализиованных точек процесса
    :param kwargs:
    :return None:
    """
    ax.plot([point.s for point in points], [point.h for point in points], **kwargs)

    
def get_isobar(point: IAPWS97) -> Tuple[List[float], List[float]]:
    """
    Собрать координаты изобары в hs осях
    :param IAPWS97 point: Точка для изобары
    :return Tuple[List[float], List[float]]:
    """
    s = point.s
    s_values = np.arange(s * 0.9, s * 1.1, 0.2 * s / 1000)
    h_values = [gas(P=point.P, s=_s).h for _s in s_values]
    return s_values, h_values


def _get_isoterm_steam(point: IAPWS97) -> Tuple[List[float], List[float]]:
    """
    Собрать координаты изотермы для пара в hs осях
    :param IAPWS97 point: Точка для изотермы
    :return Tuple[List[float], List[float]]:
    """
    t = point.T
    p = point.P
    s = point.s
    s_max = s * 1.2
    s_min = s * 0.8
    p_values = np.arange(p * 0.8, p * 1.2, 0.4 * p / 1000)
    h_values = np.array([gas(P=_p, T=t).h for _p in p_values])
    s_values = np.array([gas(P=_p, T=t).s for _p in p_values])
    mask = (s_values >= s_min) & (s_values <= s_max)
    return s_values[mask], h_values[mask]


def _get_isoterm_two_phases(point: IAPWS97) -> Tuple[List[float], List[float]]:
    """
    Собрать координаты изотермы для влажного пара в hs осях
    :param IAPWS97 point: Точка для изотермы
    :return Tuple[List[float], List[float]]:
    """
    x = point.x
    p = point.P
    x_values = np.arange(x * 0.9, min(x * 1.1, 1), (1 - x) / 1000)
    h_values = np.array([gas(P=p, x=_x).h for _x in x_values])
    s_values = np.array([gas(P=p, x=_x).s for _x in x_values])
    return s_values, h_values


def get_isoterm(point) -> Tuple[List[float], List[float]]:
    """
    Собрать координаты изотермы в hs осях
    :param IAPWS97 point: Точка для изотермы
    :return Tuple[List[float], List[float]]:
    """
    if point.phase == 'Two phases':
        return _get_isoterm_two_phases(point)
    return _get_isoterm_steam(point)


def plot_isolines(ax: plt.Axes, point: IAPWS97) -> None:
    """
    Отрисовка изобары и изотермы
    :param plt.Axes ax: AxesSubplot на котором изобразить линии
    :param IAPWS97 point: Точка для изображения изолиний
    :return None:
    """
    s_isobar, h_isobar = get_isobar(point)
    s_isoterm, h_isoterm = get_isoterm(point)
    ax.plot(s_isobar, h_isobar, color='green', label='Изобара')
    ax.plot(s_isoterm, h_isoterm, color='blue', label='Изотерма')

    
def plot_points(ax: plt.Axes, points: List[IAPWS97]) -> None:
    """
    Отрисовать точки на hs-диаграмме
    :param plt.Axes ax: AxesSubplot на котором изобразить точки
    :param List[IAPWS97] points: Точки для отображения
    return None
    """
    for point in points:
        ax.scatter(point.s, point.h, s=50, color="red")
        plot_isolines(ax, point)
        
def get_humidity_constant_line(
    point: IAPWS97,
    max_p: float,
    min_p: float,
    x: Optional[float]=None
) -> Tuple[List[float], List[float]]:
    """
    Собрать координаты линии с постоянной степенью сухости в hs осях
    :param IAPWS97 point: Точка для изолинии
    :param float max_p: Максимальное давление для линии
    :param float min_p: Минимальное давление для линии
    :param Optional[float] x: Степень сухости для отрисовки
    :return Tuple[List[float], List[float]]:
    """
    _x = x if x else point.x
    p_values = np.arange(min_p, max_p, (max_p - min_p) / 1000)
    h_values = np.array([gas(P=_p, x=_x).h for _p in p_values])
    s_values = np.array([gas(P=_p, x=_x).s for _p in p_values])
    return s_values, h_values

def plot_humidity_lines(ax: plt.Axes, points: List[IAPWS97]) -> None:
    """
    Отрисовать изолинии для степеней сухости на hs-диаграмме
    :param plt.Axes ax: AxesSubplot на котором изобразить изолинии
    :param List[IAPWS97] points: Точки для отображения
    return None
    """
    pressures = [point.P for point in points]
    min_pressure = min(pressures) if min(pressures) > 700/1e6 else 700/1e6
    max_pressure = max(pressures) if max(pressures) < 22 else 22
    for point in points:
        if point.phase == 'Two phases':
            s_values, h_values = get_humidity_constant_line(point, max_pressure, min_pressure, x=1)
            ax.plot(s_values, h_values, color="gray")
            s_values, h_values = get_humidity_constant_line(point, max_pressure, min_pressure)
            ax.plot(s_values, h_values, color="gray", label='Линия сухости')
            ax.text(s_values[10], h_values[10], f'x={round(point.x, 2)}')

def plot_hs_diagram(ax: plt.Axes, points: List[IAPWS97]) -> None:
    """
    Построить изобары и изотермы для переданных точек. Если степень сухости у точки не равна 1, то построется
    дополнительно линия соответствующей степени сухости
    :param plt.Axes ax: AxesSubplot на котором изобразить изолинии
    :param List[IAPWS97] points: Точки для отображения
    return None
    """
    plot_points(ax, points)
    plot_humidity_lines(ax, points)
    ax.set_xlabel(r"S, $\frac{кДж}{кг * K}$", fontsize=14)
    ax.set_ylabel(r"h, $\frac{кДж}{кг}$", fontsize=14)
    ax.set_title("HS-диаграмма процесса расширения", fontsize=18)
    ax.legend()
    ax.grid()
    legend_without_duplicate_labels(ax)

def plot_hs(p0, t0, p_middle, t_middle, pk,internal_efficiency):    
    point0, point1, point2, point_middle = point_s(p0, t0, p_middle, t_middle, pk,internal_efficiency)
    fig, ax  = plt.subplots(1, 1, figsize=(15, 15))
    plot_hs_diagram(
        ax,
        points=[point0[0], point0[1], point1[0], point1[1], point_middle[0], point_middle[1], point2[1], point2[0]]
    )
    plot_process(ax, points=[point0[0], point0[1], point1[1]], color='black')
    plot_process(ax, points=[point_middle[0], point_middle[1], point2[1]], color='black')
    plot_process(ax, points=[point0[0], point0[1], point1[0]], alpha=0.5, color='grey')
    plot_process(ax, points=[point_middle[0], point_middle[1], point2[0]], alpha=0.5, color='grey')


def speed_u (d_sr,n):
    u = np.pi * d_sr * n
    return u 

def calculation_nozzle_params(point0, d_sr, n, p, H0, inlet_mass_flow):
    H0_c = (1-p) * H0 # Изоэнтальпильный теплоперепад сопловой решотки по параметрам тормажения 
    H0_p = p * H0 # Изоэнтальпильный теплоперепад рабочей решотки по статическим параметрам
    h1t = point0[0].h - H0_c # Теоретическая энтальпия за сапловой решоткой
    point_1_t = gas(h = h1t, s = point0[0].s)
    p1 = gas(h = h1t, s = point0[0].s).P # Давление за сопловой решоткой
    v1t = gas(h = h1t,s = point0[0].s).v # Удельный обьем за сопловой решоткой(теор)
    c1t = np.sqrt(2 * H0_c * 1000) # Теоретиеская скорость выхода из соплавых лопаток
    k = 1.4 # Показатель изоэнтропы
    a1t = np.sqrt(k* p1 * MPa * v1t)
    M1t = c1t / a1t #Число Маха
    mu1 = 0.97
    F1_ = (inlet_mass_flow * np.array(v1t)) / (mu1 * c1t) # Выходная площадь сопловой решетки (пред) 
    alf1_e = 13 # Угол направления скорости
    el1 = F1_ / (np.pi * d_sr * math.sin(math.radians(alf1_e)))# Произведение el1
    e_opt = 4 * np.sqrt(el1) # Оптимальное значение степени парциальности
    if e_opt > 0.85: 
        e_opt = 0.85
    l1_ = el1 / e_opt # Высота сопловых лопаток предварительнвя

    return H0_c, H0_p, h1t, v1t, F1_, c1t, a1t, M1t, l1_, point_1_t


def selection_nozzle_profile():
    name = "C-90-12A"
    alpha1_e = range(10, 15)
    alpha_0 = range(70, 120)
    t_otn = np.arange(0.72, 0.88, 0.01) 
    M1t_ = 0.85
    b1 = 52.5
    f1 = 4.09 # в см^2
    I_1_min = 0.591 # в см^4
    W_1_min = 0.575 # в см^3
    alpha_install = 33.2
    return name,alpha1_e,alpha_0, t_otn, M1t_, b1, f1, I_1_min, W_1_min, alpha_install


def grid_tab():
    _,alpha1_e,alpha_0, t_otn, M1t_, b1, f1, I_1_min ,W_1_min, _ = selection_nozzle_profile()
    alpha1_e = str(f'{alpha1_e[0]}' + ' - ' + f'{alpha1_e[-1]}')
    alpha_0 = str(f'{alpha_0[0]}' + ' - ' + f'{alpha_0[-1]}')
    t_otn = str(f'{t_otn[0]}' + ' - ' + f'{round(t_otn[-1],2)}')
    if  type(M1t_) != float:
        M1t_ = str(f'{M1t_[0]}' + ' - ' + f'{M1t_[-1]}')
    d = {
        'Показатель': ["Угол выхода потока из решётки", 
                "Угол входа в решётку", 
                "Оптимальный шаг решётки", 
                "Максимальное допустимое число Маха", 
                "Хорда сопловой решётки, мм", 
                "Площадь поперечного сечения сопловой решётки, см^4", 
                "Момент инерции сопловой решётки", 
                "Момент сопротивления сопловой решётки"],
        'Параметр': [r'$\alpha_{1э}$, град ', r'$\alpha_{0расч}$, град', r"$t_{отн}$", r"$M_{1t}$", r"$b_{1}$, мм", r"F, $см^{2}$", r"$I_{min}, см^{4}$", r"$W_{min}$, $см^{3}$"],
    'Значение': [alpha1_e,alpha_0, t_otn, M1t_, b1, f1, I_1_min ,W_1_min]
    }
    df2 = pd.DataFrame(data = d)
    return df2


def correction_params(point0, d_sr, n, p, H0, inlet_mass_flow):
    _, _, _, v1t, _, c1t, _, _, _, _ = calculation_nozzle_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _,_,_, t_otn, _, b1, _, _, _, _ = selection_nozzle_profile()
    mu1_ = 0.97
    F1 = (inlet_mass_flow * v1t) / (mu1_ * c1t) # Выходная площадь сопловой решетки (пред) 
    alf1_e = 13 # Угол направления скорости
    el1 = F1 / (np.pi * d_sr * math.sin(math.radians(alf1_e)))# Произведение el1
    e_opt = 4 * np.sqrt(el1) # Оптимальное значение степени парциальности
    if e_opt > 0.85: 
        e_opt = 0.85
    l1 = el1 / e_opt # Высота сопловых лопаток
    mu1 = 0.982 - 0.005 * (b1 * 10**(-3) / l1) # уточняем коэффициент расхода сопловой решетки
    t_otn = sum(t_otn) / len(t_otn) 
    z1 = round((np.pi * d_sr * e_opt) / (b1 * 10**(-3) * t_otn))
    if z1 % 2 == 1:
        z1+=1
    t_1 = (np.pi * d_sr * e_opt)/(b1 * 10**(-3) * z1)
    return el1, F1, alf1_e, e_opt, l1, mu1 ,z1 ,t_1


def atlas_params(point0, d_sr, n, p, H0, inlet_mass_flow):
    _,_,_, _, _, b1, _, _ ,_, _ = selection_nozzle_profile()
    _, _, alf1_e, _, l1, mu1 ,_ ,_ = correction_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, c1t, _, _, _, _ = calculation_nozzle_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    ksi_p = 2  #коэффициент профильных потерь
    ksi_s = 4.4 
    ksi_k = ksi_s - ksi_p 
    phi_s = np.sqrt(1 - ksi_s/100)
    b1_l1 = (b1 * 10 ** -3) / l1
    phi_s_ = 0.98 - 0.008 * (b1 * 10 ** (-3) / l1) #проверочный коэфициент скорости сопловой решотки 
    delt = ((phi_s_- phi_s) / phi_s_) * 100 
    if delt > 1:
        raise ValueError("ищи ошибку") 
    c1 = c1t * phi_s
    alpha_1 = math.degrees(math.asin((mu1 / phi_s) * math.sin(math.radians(alf1_e))))
    return ksi_p, ksi_s, ksi_k, phi_s, b1_l1, c1, alpha_1

def calculation_working_grid(point0, d_sr, n, p, H0, inlet_mass_flow):
    u = speed_u (d_sr,n)
    _, H0_p, _, _, _, c1t, _, _, _, point_1_t = calculation_nozzle_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, l1, _ ,_ ,_ = correction_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, phi_s, _, c1, alpha_1 = atlas_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    w_1 = math.sqrt(c1 ** 2 + u ** 2 - 2 * c1 * u * math.cos(math.radians(alpha_1)))
    beta_1 = math.degrees(math.atan(math.sin(math.radians(alpha_1)) / (math.cos(math.radians(alpha_1)) - u / c1)))
    delta_Hc = (0.5 * c1t ** 2) * (1 - phi_s ** 2)
    h1 = point_1_t.h + delta_Hc/1000
    point_1_ = gas(P = point_1_t.P, h = h1)  
    h2t = point_1_.h - H0_p
    point_2_t = gas(s = point_1_.s, h = h2t)
    w2t = math.sqrt(2 * H0_p * 1000 + w_1 ** 2)
    delta = 0.004
    l2 = l1 + delta  
    k2 = 1.3
    a2t = math.sqrt(k2 * (point_2_t.P * MPa) * point_2_t.v)
    M2t = w2t / a2t
    return w_1, beta_1, w2t, l2, a2t, M2t, delta_Hc, point_2_t, point_1_


def grid_working_selection():
    
    name = "P-23-14A"
    beta_2_e = range(11, 16)
    beta_1_ = range(20, 30)
    t_otn = np.arange(0.60, 0.75, 0.01) 
    M2t_ = 0.95
    b2 = 26
    f2 = 2.44 # в см^2
    I_2_min = 0.43 # в см^4
    W_2_min = 0.39 # в см^3
    alpha_install = 77.8

    return name, beta_2_e, beta_1_, t_otn, M2t_, b2, f2, I_2_min, W_2_min, alpha_install


def grid_tab_work():
    name, beta_2_e, beta_1_, t_otn, M2t_, b2, f2, I_2_min, W_2_min, alpha_install = grid_working_selection()
    beta_2_e = str(f'{beta_2_e[0]}' + ' - ' + f'{beta_2_e[-1]}')
    beta_1_ = str(f'{beta_1_[0]}' + ' - ' + f'{beta_1_[-1]}')
    t_otn = str(f'{t_otn[0]}' + ' - ' + f'{round(t_otn[-1],2)}')
    if  type(M2t_) != float:
        M2t_ = str(f'{M2t_[0]}' + ' - ' + f'{M2t_[-1]}')
    
    c = {
        'Показатель': ["Угол выхода потока из решётки", 
                "Угол входа в решётку", 
                "Оптимальный шаг решётки", 
                "Максимальное допустимое число Маха", 
                "Хорда сопловой решётки, мм", 
                "Площадь поперечного сечения сопловой решётки, см^4", 
                "Момент инерции сопловой решётки" , 
                "Момент сопротивления сопловой решётки"],
        'Параметр': [r'$\beta_{2э}$, град ', r'$\beta_{1расч}$, град', r"$t_{отн}$", r"$M_{2t}$", r"$b_{2}$, мм", r"F, $см^{2}$", r"$I_{min}, см^{4}$", r"$W_{min}$, $см^{3}$"],
    'Значение': [beta_2_e, beta_1_, t_otn, M2t_, b2, f2, I_2_min, W_2_min]
    }    
     
    df2 = pd.DataFrame(c)
    return df2


def specification_working_grid_parameters(point0, d_sr, n, p, H0, inlet_mass_flow):
    _, _, w2t, l2, _, _, _, point_2_t, point_1_ = calculation_working_grid(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, e_opt, _, _ ,_ ,_ = correction_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, t_otn, M2t_, b2, _, _, _, _ = grid_working_selection()
    mu2 = 0.965 - 0.01 * (b2 * 10 ** -3 / l2)
    F2 = (inlet_mass_flow * point_2_t.v) / (mu2 * w2t)
    beta2_e = math.degrees(F2/(e_opt * math.pi * d_sr * l2))
    t2opt = (t_otn[0]+t_otn[-1])/2
    z2 = (math.pi * d_sr) / (b2 * 10 ** -3 * t2opt)
    z_2 = round(z2+0.5)-1 if (round(z2) % 2) else round(z2+0.5)
    t2opt = (math.pi * d_sr) / (b2 * 10 ** -3 * z2)
    beta2_ust = beta2_e - 3.25 * (t2opt - 0.70) + 63.8
    b2_l2 = (b2 * 10 ** -3) / l2  
    return mu2, F2, beta2_e, z_2, t2opt, beta2_ust, b2_l2


def parameters_working_atlas(point0, d_sr, n, p, H0, inlet_mass_flow):
    u = speed_u (d_sr,n)
    mu2, _, beta2_e, _, _, _, _ = specification_working_grid_parameters(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, w2t, l2, _, _, _, _, point_1_ = calculation_working_grid(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, b2, _, _, _, _ = grid_working_selection()
    ksi_grid = 5.5    
    ksi_sum_g = 9.8 
    ksi_end_grid = ksi_sum_g - ksi_grid 
    psi = math.sqrt(1 - ksi_sum_g/100)
    psi_ = 0.96 - 0.014 * (b2 * 10 ** -3 / l2)
    delta_psi = (psi - psi_) / psi
    w_2 = w2t * psi
    beta_2 = math.degrees(math.asin((mu2 / psi) * math.sin(math.radians(beta2_e))))
    c_2 = math.sqrt(w_2 ** 2 + u ** 2 - 2 * w_2 * u * math.cos(math.radians(beta_2)))
    #alpha_2 = math.degrees(math.atan((math.sin(math.radians(beta_2)))/(math.cos(math.radians(beta_2)) - u/w_2)))
    alpha_2 = math.degrees(math.asin((w_2*math.sin(math.radians(beta_2))) / c_2))

    return ksi_grid, ksi_sum_g, ksi_end_grid, psi, beta_2, c_2, alpha_2, w_2


def plot_triangles(point0, d_sr, n, p, H0, inlet_mass_flow):
    u = speed_u (d_sr,n)
    _, _, _, _, _, c_1, alpha_1 = atlas_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, beta_2, _, _, w_2 =  parameters_working_atlas(point0, d_sr, n, p, H0, inlet_mass_flow)
    sin_alpha_1 = math.sin(math.radians(alpha_1))
    cos_alpha_1 = math.cos(math.radians(alpha_1))
    sin_beta_2 = math.sin(math.radians(beta_2))
    cos_beta_2 = math.cos(math.radians(beta_2))


    c1_plot = [[0, -c_1 * cos_alpha_1], [0, -c_1 * sin_alpha_1]]
    u1_plot = [[-c_1 * cos_alpha_1, -c_1 * cos_alpha_1 + u], [-c_1 * sin_alpha_1, -c_1 * sin_alpha_1]]
    w1_plot = [[0, -c_1 * cos_alpha_1 + u], [0, -c_1 * sin_alpha_1]]
    w2_plot = [[0, w_2 * cos_beta_2], [0, -w_2 * sin_beta_2]]
    u2_plot = [[w_2 * cos_beta_2, w_2 * cos_beta_2 - u], [-w_2 * sin_beta_2, -w_2 * sin_beta_2]]
    c2_plot = [[0, w_2 * cos_beta_2 - u], [0, -w_2 * sin_beta_2]]

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(c1_plot[0], c1_plot[1], label='C_1', c='red')
    ax.plot(u1_plot[0], u1_plot[1], label='u_1', c='blue')
    ax.plot(w1_plot[0], w1_plot[1], label='W_1', c='green') 
    ax.plot(w2_plot[0], w2_plot[1], label='W_2', c='green')
    ax.plot(u2_plot[0], u2_plot[1], label='u_2', c='blue')
    ax.plot(c2_plot[0], c2_plot[1], label='C_2', c='red')
    ax.set_title("Треугольник скоростей",)
    ax.legend()
    ax.grid()


def calculation_velocity_ratio(point0, d_sr, n, p, H0, inlet_mass_flow):
    
    u = speed_u (d_sr,n)
    _, _, _, phi_s, _, _, alpha_1 = atlas_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    cf = math.sqrt(2 * H0 * 1000)
    u_cf = u / cf
    u_cf_opt = phi_s * math.cos(math.radians(alpha_1)) / (2 * math.sqrt(1 - p))
    return cf, u_cf, u_cf_opt


def calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow):
    u = speed_u (d_sr,n)
    w_1, beta_1, w2t, _, _, _, delta_Hc, point_2_t, point_1_ = calculation_working_grid(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, c_1, alpha_1 = atlas_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, psi, beta_2, c_2, alpha_2, w_2 =  parameters_working_atlas(point0, d_sr, n, p, H0, inlet_mass_flow)
    delta_Hp = (0.5 * w2t ** 2) * (1 - psi ** 2)
    h2 = point_2_t.h + delta_Hp / 1000
    point_2_ = gas(P = point_2_t.P, h = h2)
    point_t_konec = gas(h =point0[1].h - H0, P = point_2_.P)
    delta_Hvc = ((c_2 ** 2) / 2) 
    x_vc = 0
    E0 = H0 * 1000 - x_vc * delta_Hvc  
    eff = (E0 - delta_Hc - delta_Hp - (1 - x_vc) * delta_Hvc) / E0
    #eff_ = (u * (c_1 * math.cos(math.radians(alpha_1)) + c_2 * math.cos(math.radians(alpha_2)))) / E0 
    eff_ = (u * (w_1 * math.cos(math.radians(beta_1)) + w_2 * math.cos(math.radians(beta_2)))) / E0 
    delta_eff = (eff - eff_) / eff   
    return delta_Hp, delta_Hvc, E0, eff, eff_, delta_eff, point_2_, point_t_konec


def efficiency_graph(point0, d_sr, n, p, H0, inlet_mass_flow):
    u_cf = []
    eff = []
    eff_ = []
    for i in H0:
        u_cf.append(calculation_velocity_ratio(point0, d_sr, n, p, i, inlet_mass_flow)[1])
        eff.append(calculation_blade_efficiency(point0, d_sr, n, p, i, inlet_mass_flow)[3])
        eff_.append(calculation_blade_efficiency(point0, d_sr, n, p, i, inlet_mass_flow)[4])
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    X1 = u_cf
    X2 = H0
    Y1 = eff_
    Y2 = eff
    

    ax.plot(X1,Y1, label = 'По расчёту через скорости', color = 'blue')
    ax.plot(X1,Y2, label = 'По расчёту через потери энергии', color = 'red')
    ax.set_title("Зависимость лопаточного КПД от u/сф")
    ax.set_ylabel("Лопаточный КПД")
    ax.set_xlabel("U/сф")
    ax.legend()
    ax.grid()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.plot(X2,Y1, label = 'По расчёту через скорости', color = 'blue')
    plt.plot(X2,Y2, label = 'По расчёту через потери энергии', color = 'red')
    plt.title("Зависимость лопаточного КПД от H0")
    plt.ylabel("Лопаточный КПД")
    plt.xlabel("H0")
    plt.legend()
    plt.grid()
    plt.show()

    
def data_output(point0, d_sr, n, p, H0, inlet_mass_flow):
    H0_c, H0_p, __, _, F1_, c1t, a1t, M1t, _, _ = calculation_nozzle_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    name,alpha1_e,alpha_0, t_otn, M1t_, b1, f1, I_1_min, W_1_min, alpha_install = selection_nozzle_profile()
    d = {
        'Показатель': ["Теплоперепад в сопловой решётке", 
                "Теплоперепад в рабочей решётке",
                "Распалагаемый теплоперепад ", 
                "Теоретическая абсолютная скорость на выходе из сопловой решётки", 
                "Скорость звука на выходе из сопловой решётки", 
                "Число Маха на выходе из сопловой решётки", 
                "Предварительная площадь выхода потока из сопловой решётки",
                "Стерень реактивности"],
        'Параметр': [r"$H_{0c} \space \frac{кДж}{кг}$", r"$H_{0p} \space \frac{кДж}{кг}$", r"$H_{0} \space \frac{кДж}{кг}$",r"$c_{1t}, \space \frac{м}{с}$", r"$a_{1t}, \space \frac{м}{с}$", r"$M_{1t}$", r"$F_{1}, \space м^{2}$",r"$\rho$"],
    'Значение': [round(H0_c,1),round(H0_p,1),round(H0,1), round(c1t,3), round(a1t,3), round(M1t,3),round(F1_,3), p]
    }

    el1, _, _, e_opt, l1, mu1 ,z1 ,t_1 = correction_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    g = {
        'Показатель': ["Произведение el1", 
            "Оптимальное значение степени парциальности", 
            "Высота сопловых лопаток, мм",
            "Хорда сопловой решётки, мм",
            "Уточняем коэффициент расхода сопловой решетки", 
            "Количество лопаток в сопловой решетке",
            "Уточненый оптимальный относительный шаг"],
        'Параметр': [r"$el_{1}, м$", r"$e_{opt}$",r"$l_{1}$, м",r"$b_{1}$, м", r"$\mu_{1}$", r"$z_{1}$", r"$t_{1}$"],
    'Значение': [el1 ,e_opt, l1, b1, mu1, z1, t_1]
    }

    ksi_p, ksi_s, ksi_k, phi_s, b1_l1, c1, alpha_1 = atlas_params(point0, d_sr, n, p, H0, inlet_mass_flow)

    a = {
        'Показатель': ["Угол установки профиля решётки", 
              "Отношение: b1/l1", 
              "Коэффициент профильных потерь", 
              "Коэффициент суммарных потерь", 
              "Коэффициент концевых потерь", 
              "Коэффициент скорости сопловой решетки", 
              "Скорость выхода пара из сопловой решетки",
              "Угол выхода потока из сопловой решётки"],
        'Параметр': [r"$\alpha_{уст}, град$", r"$\frac{b_{1}}{l_{1}}$",r"$\xi_{проф}$", r"$\xi_{сум}$",r"$\xi_{конц}$", r"$\phi$", r"$c_{1} \frac{м}{с}$", r"$\alpha_{1}, град$"],
    'Значение': [alpha_install ,round(b1_l1,3), ksi_p, ksi_s, ksi_k, phi_s, c1, alpha_1]
    }
    
    w_1, beta_1, w2t, l2, a2t, M2t, delta_Hc, _, _ = calculation_working_grid(point0, d_sr, n, p, H0, inlet_mass_flow)
    name2, beta_2_e, beta_1_, t_otn, M2t_, b2, f2, I_2_min, W_2_min, alpha_install2 = grid_working_selection() 
    b = {
        'Показатель': ["Относительная скорость на выходе из сопловой решётки", 
              "Угол направления относительной скорости потока на выходе из сопловой решётки", 
              "Теоретическая относительная скорость на выходе из рабочей решётки", 
              "Высота рабочих лопаток",
              "Хорда  решётки, мм",
              "Скорость звука за рабочей решеткой (теоретическая)", 
              "Теоретическое число Маха за рабочей решёткой", 
              "Потери в сопловой решетке"],
        'Параметр': [r"$w_{1} \space \frac{м}{с}$", r"$\beta_{1}, град$", r"$w_{2t}, \space \frac{м}{с}$",r"$l_{2}, м$" ,r"$b_{2}, м$" ,r"$a_{2t}, \space \frac{м}{с}$", r"$M_{2t}$", r"$\Delta H_{c} \frac{кДж}{кг}$"],
    'Значение': [w_1, beta_1, w2t, l2, b2, a2t, M2t, delta_Hc/1000,]
    }

 
    mu2, F2, beta2_e, z_2, t2opt, beta2_ust, b2_l2 = specification_working_grid_parameters(point0, d_sr, n, p, H0, inlet_mass_flow)
    с = {
        'Показатель':["Коэффициент расхода рабочей решётки", 
              "Выходная площадь рабочей решётки", 
              "Эффективный угол выхода потока из рабочей решётки", 
              "Количество лопаток в рабочей решётке", 
              "Оптимальный шаг рабочей решётки", 
              "Угол установки рабочих лопаток", 
              r"Отношение $\frac{b_{2}}{l_{2}}$"],
        'Параметр': [r"$\mu_{2}$", r"$F_{2} \space м^{2}$",r"$\beta_{2e}, град$", r"$z_{2}$", r"$t_{2opt}$",r"$\beta_{уст}, град$", r"$\frac{b_{2}}{l_{2}}$"],
    'Значение': [mu2, F2, beta2_e, z_2, t2opt, beta2_ust, b2_l2]
    }
    ksi_grid, ksi_sum_g, ksi_end_grid, psi, beta_2, c_2, alpha_2, w_2 =  parameters_working_atlas(point0, d_sr, n, p, H0, inlet_mass_flow)
    v = {
        'Показатель': ["Коэффициент профильных потерь в решётке", 
              "Коэффициент суммарных потерь", 
              "Коэффициент концевых потерь", 
              "Коэффициент скорости рабочей решётки", 
              "Угол направления относительной скорости на выходе из рабочей решётки", 
              "Абсолютная скорость на выходе из рабочей решётки",
              "Угол выхода абсолютной скорости из рабочей решётки",
              "Действительная относительная скорость на выходе из рабочей решётки"],
        'Параметр': [r"$\xi_{проф}$", r"$\xi_{сум}$",r"$\xi_{конц}$", r"$\psi$", r"$\beta_{2}, град$",r"$c_{2}, \space \frac{м}{с}$", r"$\alpha_{2}, град$", r"$w_{2}, \space \frac{м}{с}$" ],
    'Значение': [ksi_grid, ksi_sum_g, ksi_end_grid, psi, beta_2, c_2, alpha_2, w_2]
    }
    delta_Hp, delta_Hvc, E0, eff, eff_, delta_eff, point_2_, point_t_konec = calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow)
    eff_oi, N_i, delta_Htr, delta_Hy, delta_Hparc = inside_kpd(point0, d_sr, n, p, H0, inlet_mass_flow)



    df = pd.DataFrame(data=d)
    df2 = grid_tab()
    df3 = pd.DataFrame(data=g)
    df4 = pd.DataFrame(data=a)
    df5 = pd.DataFrame(data=b)
    df7 = pd.DataFrame(data=с)
    df8 = pd.DataFrame(data=v)
    #df9 = pd.DataFrame(data=o)

    print('Предрасчет сопловой решетки')
    display(df)
    #print(f'Подходячий тип сопловой решетки {name}')
    #display(df2)
    print('Расчет дополнительных параметров сопловой решетки')
    display(df3)
    print('Расчет параметров сопловой решетки из аталаса')
    display(df4)
    print('Предрасчет рабочей решетки')
    display(df5)
    #print(f'Подходячий тип рабочей решетки {name2}')
    print('Расчет дополнительных параметров рабочей решетки')
    display(df7)
    print('Расчет параметров рабочей решетки из аталаса')
    display(df8)
    print('')


    pass


def inside_kpd(point0, d_sr, n, p, H0, inlet_mass_flow):
    _, _, _, l2, _, _, delta_Hc, _,_ = calculation_working_grid(point0, d_sr, n, p, H0, inlet_mass_flow)
    delta_Hp, delta_Hvc, E0, eff, _, _, _, _ = calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, u_cf, _ = calculation_velocity_ratio(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, F1, alf1_e, e_opt, _, _ ,_ ,_ = correction_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, beta2_ust, _ = specification_working_grid_parameters(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, b2, _, _, _, _ = grid_working_selection()   
    x_vc = 0
    d_п = d_sr + l2
    mu_a = 0.5
    delta_a = 0.0025
    mu_r = 0.75
    delta_r = 0.001 * d_п
    z_gr = 7
    delta_э = (1/(mu_a * delta_a) ** 2 + z_gr/(mu_r * delta_r) ** 2) ** (-1/2)
    ksi_by = (math.pi * d_п * delta_э * eff * math.sqrt((p + 1.8 * l2)/d_sr))/F1
    delta_Hy = ksi_by * E0
    k_tr = 0.7 * 10 ** (-3)
    ksi_tr = k_tr * (d_sr ** 2) * (u_cf) ** 3 / F1
    delta_Htr = ksi_tr * E0
    k_v = 0.065
    m = 1
    ksi_v = k_v * (1 - e_opt) * m *( u_cf ) **3  / (math.sin(math.radians(alf1_e)) * e_opt)
    B_2 = b2 * 10**-3 * math.sin(math.radians(beta2_ust))
    i = 4
    ksi_segm = 0.25 * B_2 * l2 * u_cf * eff * i / F1
    ksi_parc = ksi_v + ksi_segm
    delta_Hparc = ksi_parc * E0
    H_i = E0 - delta_Hc - delta_Hp - (1-x_vc) * delta_Hvc - delta_Hy - delta_Htr - delta_Hparc
    eff_oi = H_i / E0
    N_i = inlet_mass_flow * H_i
    eff_oi,N_i, delta_Htr, delta_Hy
    return eff_oi, N_i, delta_Htr, delta_Hy, delta_Hparc

def plot_reg(point0, d_sr, n, p, H0, inlet_mass_flow):
    delta_Hp, delta_Hvc, _, _, _, _, _, _ = calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, _, _, _, _, point_1_t = calculation_nozzle_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, _, _, point_2_t, point_1_ = calculation_working_grid(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, _, point_2_, point_t_konec = calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow)
    eff_oi,N_i, delta_Htr, delta_Hy, delta_Hparc = inside_kpd(point0, d_sr, n, p, H0, inlet_mass_flow)  
    h_  = point_t_konec.h + delta_Hp/1000 + delta_Htr/1000 + delta_Hparc/1000 + delta_Hy/1000 + ((1 - 0) * delta_Hvc/1000)
    point_d_ = gas(P = point_2_.P, h = h_)
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.set_xlim(6.22, 6.25)
    ax.set_ylim(3250, 3375)
    ax.grid()
    plot_hs_diagram(ax = ax, points = [point0[0], point_1_t, point_1_, point_2_t, point_2_, point_t_konec, point_d_])
    plot_process(ax,[point0[0], point_1_], color='black')
    plot_process(ax,[point_1_, point_2_],  color='black')
    plot_process(ax,[point_2_, point_d_],  color='black')
    plot_process(ax,[point0[0], point_1_t, point_t_konec], alpha=0.5, color='grey')
    plot_process(ax,[point_1_, point_2_t],  alpha=0.5, color='grey')
    ax.set_title("HS-диаграмма процесса расширения в регулирующей ступени", fontsize=18)
    ax.grid()

def tab_point_reg(point0, d_sr, n, p, H0, inlet_mass_flow):
    delta_Hp, delta_Hvc, _, _, _, _, _, _ = calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, _, _, _, _, point_1_t = calculation_nozzle_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, _, _, point_2_t, point_1_ = calculation_working_grid(point0, d_sr, n, p, H0, inlet_mass_flow)
    _, _, _, _, _, _, point_2_, point_t_konec = calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow)
    eff_oi,N_i, delta_Htr, delta_Hy, delta_Hparc = inside_kpd(point0, d_sr, n, p, H0, inlet_mass_flow)  
    h_  = point_t_konec.h + delta_Hp/1000 + delta_Htr/1000 + delta_Hparc/1000 + delta_Hy/1000 + ((1 - 0) * delta_Hvc/1000)
    point_d_ = gas(P = point_2_.P, h = h_)
    df1 = pd.DataFrame([(point0[0].P, point_1_.P, point_2_.P, point_d_.P),
                        (point0[0].T, point_1_.T, point_2_.T, point_d_.T),
                        (point0[0].h, point_1_.h, point_2_.h, point_d_.h),
                        (point0[0].s, point_1_.s, point_2_.s, point_d_.s)],
                        index=["P, МПа", "T, K", r"h, $\frac{кДж}{кг}$", r"S, $\frac{кДж}{кг * K}$ "], 
                        columns=["point 0", "point 1", "point 2", "point konec"])
    return df1


def determination_of_the_number_of_steps(point0, d_sr, n, p, H0, inlet_mass_flow, internal_efficiency,n_stages,veernost_1):
    _, _, _, _, _, _, point_2_, point_t_konec = calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow)

    p0 = point_t_konec.P * MPa
    h0 = point_t_konec.h 
    pz = 3.74 * MPa
    point_0 = gas(P= p0 * unit, h=h0)
    delta_diam = 0.2
    speed_coefficient = 0.93
    root_reaction_degree = 0.05
    discharge_coefficient = 0.96
    alpha_1 = 14
    overlapping = 0.003
    efficiency = internal_efficiency
    avg_diam_1 = d_sr - delta_diam
    def get_reaction_degree(root_dor, veernost):
        return root_dor + (1.8 / (veernost + 1.8))

    def get_u_cf(dor):
        cos = np.cos(np.deg2rad(alpha_1))
        return speed_coefficient * cos / (2 * (1 - dor) ** 0.5)

    def get_heat_drop(diameter, u_cf):
        first = (diameter / u_cf) ** 2
        second = (n / 50) ** 2
        return 12.3 * first * second

    avg_reaction_degree_1 = get_reaction_degree(root_reaction_degree, veernost_1)
    u_cf_1 = get_u_cf(avg_reaction_degree_1)
    heat_drop_1 = get_heat_drop(avg_diam_1, u_cf_1)
    h1 = point_0.h - heat_drop_1
    point_2 = gas(h=h1, s=point_0.s)

    upper = inlet_mass_flow * point_2.v * u_cf_1
    lower = discharge_coefficient * np.sin(np.deg2rad(alpha_1)) * n * (np.pi * avg_diam_1) ** 2 * (1 - avg_reaction_degree_1) ** 0.5

    blade_length_1 = upper / lower
    blade_length_2 = blade_length_1 + overlapping

    assert np.isclose(avg_diam_1 / blade_length_1, veernost_1, rtol=0.01)

    root_diameter = avg_diam_1 - blade_length_2
    point_zt = gas(P=pz * unit, s=point_0.s)
    full_heat_drop = h0 - point_zt.h
    actual_heat_drop = full_heat_drop * efficiency
    hz = h0 - actual_heat_drop
    point_z = gas(P=pz * unit, h=hz)

    from scipy.optimize import fsolve

    def equation_to_solve(x):
        return x ** 2 + x * root_diameter - avg_diam_1 * blade_length_2 * point_z.v / point_2.v

    blade_length_z = fsolve(equation_to_solve, 0.01)[0]
    

    avg_diam_2 = root_diameter + blade_length_z
    

    def linear_distribution(left, right, x):
        return (right - left) * x + left

    x = np.cumsum(np.ones(n_stages) * 1 / (n_stages - 1)) - 1 / (n_stages - 1)
    diameters = linear_distribution(avg_diam_1, avg_diam_2 , x)
    blade_lengths = linear_distribution(blade_length_2, blade_length_z , x)
    veernosts = diameters / blade_lengths
    reaction_degrees = get_reaction_degree(root_dor=root_reaction_degree, veernost=veernosts)
    u_cf = get_u_cf(dor=reaction_degrees)
    heat_drops = get_heat_drop(diameters, u_cf)
    output_speed_coeff_loss = np.full_like(heat_drops, 0.95)
    output_speed_coeff_loss[0] = 1
    actual_heat_drops = output_speed_coeff_loss * heat_drops
    mean_heat_drop = np.mean(actual_heat_drops)
    reheat_factor = 4.8 * 10 ** (-4) * (1 - efficiency) * full_heat_drop * (n_stages - 1) / n_stages
    full_heat_drop * (1 + reheat_factor) / mean_heat_drop
    bias = full_heat_drop * (1 + reheat_factor) - np.sum(actual_heat_drops)
    bias = bias / n_stages
    new_actual_heat_drop = actual_heat_drops + bias
    return root_diameter,diameters, blade_lengths, veernosts, reaction_degrees, u_cf, new_actual_heat_drop, avg_diam_2, blade_length_z

def plot_distribution(values, ax_name):
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    ax.plot(range(1, 12), values,  marker='o')
    ax.set_xlabel("Номер ступени")
    ax.set_ylabel(ax_name)
    ax.grid()

def plot_heat_drop(point0, d_sr, n, p, H0, inlet_mass_flow, internal_efficiency,n_stages,veernost_1):
    root_diameter,diameters, blade_lengths, veernosts, reaction_degrees, u_cf, new_actual_heat_drop, avg_diam_2,blade_length_z = determination_of_the_number_of_steps(point0, d_sr, n, p, H0, inlet_mass_flow, internal_efficiency,n_stages,veernost_1)
    plot_distribution(diameters, "d, m")
    plot_distribution(blade_lengths, "l, m")
    plot_distribution(veernosts, "Веерность")
    plot_distribution(reaction_degrees, "Степень реактивности")
    plot_distribution(u_cf, "U/Cф")
    plot_distribution(new_actual_heat_drop, "Теплоперепады по ступеням")

def root_diameter_print(point0, d_sr, n, p, H0, inlet_mass_flow, internal_efficiency,n_stages,veernost_1):
    root_diameter,diameters, blade_lengths, veernosts, reaction_degrees, u_cf, new_actual_heat_drop, avg_diam_2,blade_length_z = determination_of_the_number_of_steps(point0, d_sr, n, p, H0, inlet_mass_flow, internal_efficiency,n_stages,veernost_1)
    print ("Корневой диаметр = ",root_diameter,"м")

def calculation_trength(point0, d_sr, n, p, H0, inlet_mass_flow):
    u = speed_u (d_sr,n)
    delta_Hp, delta_Hvc, E0, eff, eff_, delta_eff, point_2_, point_t_konec = calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow)
    mu2, F2, beta2_e, z_2, t2opt, beta2_ust, b2_l2 = specification_working_grid_parameters(point0, d_sr, n, p, H0, inlet_mass_flow)
    el1, F1, alf1_e, e_opt, l1, mu1 ,z1 ,t_1 = correction_params(point0, d_sr, n, p, H0, inlet_mass_flow)
    w_1, beta_1, w2t, l2, a2t, M2t, delta_Hc, point_2_t, point_1_ = calculation_working_grid(point0, d_sr, n, p, H0, inlet_mass_flow)
    name, beta_2_e, beta_1_, t_otn, M2t_, b2, f2, I_2_min, W_2_min, alpha_install = grid_working_selection()
    b2_atl = 25.9
    W2_min_ = W_2_min * math.pow(b2 / b2_atl, 3)
    sigma_bending = (inlet_mass_flow * H0 * 1000 * eff * l2) / (2 * u * z_2 * W2_min_ * e_opt)
    omega = 2 * math.pi * n
    sigma_stretching = 0.5 * 7800 * math.pow(omega, 2) * d_sr * l2
    return W2_min_, sigma_bending, omega, sigma_stretching


def vibra(point0, d_sr, n, p, H0, inlet_mass_flow, internal_efficiency,n_stages,veernost_1):
    name, beta_2_e, beta_1_, t_otn, M2t_, b2, f2, I_2_min, W_2_min, alpha_install = grid_working_selection()
    root_diameter,diameters, blade_lengths, veernosts, reaction_degrees, u_cf, new_actual_heat_drop, avg_diam_2,blade_length_z = determination_of_the_number_of_steps(point0, d_sr, n, p, H0, inlet_mass_flow, internal_efficiency,n_stages,veernost_1)
    mu2, F2, beta2_e, z_2, t2opt, beta2_ust, b2_l2 = specification_working_grid_parameters(point0, d_sr, n, p, H0, inlet_mass_flow)
    B_2 = b2 * 10**-3 * math.sin(math.radians(beta2_ust))
    mm = 1e-3
    m = n_stages
    t = 25 * mm
    beta = beta2_ust
    density = 7800
    E = 2 * (10**11)
    z = z_2
    d = avg_diam_2
    l = blade_length_z
    f = f2 * (10 ** (-4))
    J = I_2_min * (10 ** (-8))
    delta = 5 * mm
    B = B_2
    i = (J / f) ** 0.5
    _lambda = l / i
    psi = 0.9

    def static_frequency(i):
        _m = {
                1: 0.56,
                2: 3.51,
                3: 9.82 
            }
        first = psi * _m[i] / (l ** 2)
        second = ((E * J) / (density * f)) ** 0.5
        return first * second

    f_a0 = static_frequency(1) * 0.8
    f_a1 = static_frequency(1) * 6
    f_b0 = static_frequency(1) * 4.2

    H = 0.12
    J_b = B * (delta ** 3) / 12
    k = (12 * (m - 1) * H * E * J_b * l * np.sin(np.deg2rad(beta)) ** 2) / (m * t * J * E)
    nu = B * delta * t / (f * l)

    B_bandage = 0.5 * ((d/l) - 1) * ((nu+1/2)/(nu+1/3)) + np.sin(np.deg2rad(beta)) ** 2

    def to_dynamic_frequency(f, n=50):
        root = (1 + B_bandage * (n / f) ** 2) ** 0.5
        return f * root

    print(to_dynamic_frequency(f_a0), to_dynamic_frequency(f_a1), to_dynamic_frequency(f_b0))
    def min_max(f, delta=0.05):
        return f * (1-delta) , f * (1 + delta)

    n_line = np.linspace(0, 60)
    min_line, max_line = min_max(to_dynamic_frequency(f_a0, n=n_line))

    def k_line(k, n=n_line):
        return k * n_line
    fig, ax = plt.subplots(1,1,figsize=(15,10))
    ax.plot(n_line, to_dynamic_frequency(f_a0, n=n_line), label='$f_{a0}$')
    ax.fill_between(n_line, y1=min_line, y2=max_line, alpha=0.5)

    ax.plot(n_line, k_line(8), label=f'k={8}')
    ax.plot(n_line, k_line(9), label=f'k={9}')
    ax.plot(n_line, k_line(10), label=f'k={10}')
    ax.plot(n_line, k_line(11), label=f'k={11}')
    ax.plot(n_line, k_line(12), label=f'k={12}')
    ax.plot(n_line, k_line(13), label=f'k={13}')
    ax.set_xlabel("n, rps")
    ax.set_xlabel("f, Hz")
    ax.grid()
    ax.legend()
    ax.set_title("Вибрационная диаграмма");  


def dstrength(point0, d_sr, n, p, H0, inlet_mass_flow, internal_efficiency,n_stages,veernost_1):
    root_diameter,diameters, blade_lengths, veernosts, reaction_degrees, u_cf, new_actual_heat_drop, avg_diam_2,blade_length_z = determination_of_the_number_of_steps(point0, d_sr, n, p, H0, inlet_mass_flow, internal_efficiency,n_stages,veernost_1)

    MPa = 1e6
    kW = 1e3
    MW = 1e6
    nu = 0.3
    r1 = 0 
    r2 = root_diameter / 2 

    sigma_1 = 0
    sigma_2 = 100 * MPa
    density = 7800

    angular_speed = 2 * np.pi * n

    max_stress = 510 * MPa

    def sigma_r(r, r2, sigma_2):
        a = (3 + nu) / 8
        return a * density * (angular_speed ** 2) * (r2**2 - r**2) + sigma_2

    def sigma_theta(r, r2, sigma_2):
        a = (3 + nu) / 8
        b = (1 + 3 * nu) / (3 + nu)
        return a * density * (angular_speed ** 2) * (r2**2 - b * r**2) + sigma_2

    sigma_r(0, r2=r2, sigma_2=sigma_2) / MPa, sigma_r(r2, r2=r2, sigma_2=sigma_2) / MPa
    sigma_theta(0, r2=r2, sigma_2=sigma_2) / MPa, sigma_theta(r2, r2=r2, sigma_2=sigma_2) / MPa

    print("Коэффициент запаса прочности: ", max_stress/sigma_theta(0, r2=r2, sigma_2=sigma_2))

def kritic_rot_speed():
  E = 1.8 * 10**11
  density = 7800

  L = 4
  L_rotor = 4.4
  d = 0.56
  d_0 = 0.1

  rotor_mass = 11941

  EI = E * np.pi * (d ** 4 - d_0 ** 4) / 64
  print('Средняя изгибная жесткость ротора', EI)

  P_11 = (np.pi ** 2 / L ** 2) * (EI / (rotor_mass / L_rotor)) ** 0.5
  print('Первая собственная частота вращения ротора по первому тону колебаний на абсолютно жестких опорах', P_11, 'rad/s')
  #print(P_11, "rad/s")  # rad / s 

  P_12 = 4 * P_11
  print('Первая собственная частота вращения ротора по второму тону колебаний на абсолютно жестких опорах', P_12, 'rad/s')
  #print(P_12, "rad/s")
  #P_12  # rad / s 

  delta_op = 0.5 * 10 ** -9
  C_horizontal = 0.5 * 10 ** 9
  delta_opor = delta_op + 1 / C_horizontal
  delta_opor

  P_21 = (2 / (rotor_mass * delta_opor)) ** 0.5
  print('2-e собственные частоты вращения ротора на упругих опорах', P_21, 'rad/s')
  #print(P_21, "rad/s")
  #P_21  # rad / s

  P_22 = (L / L_rotor) * 3 ** 0.5 * P_21
  print('2-e собственные частоты вращения ротора на упругих опорах', P_22, 'rad/s')
  #print(P_22, "rad/s")  # rad / s

  P_1 = 1 / ((1 / P_11 ** 2) + (1 / P_21 ** 2) ) ** 0.5
  print("Первая критическая частота вращения ротора",P_1,'rad/s', P_1 / (2 * np.pi), "Гц") 
  #P_1 / (2 * np.pi)  # Hz
  P_2 = 1 / ((1 / P_12 ** 2) + (1 / P_22 ** 2) ) ** 0.5
  print("Вторая критическая частота вращения ротора",P_2,'rad/s', P_2 / (2 * np.pi), "Гц")


def data_output1(point0, d_sr, n, p, H0, inlet_mass_flow):
    delta_Hp, delta_Hvc, E0, eff, eff_, delta_eff, point_2_, point_t_konec = calculation_blade_efficiency(point0, d_sr, n, p, H0, inlet_mass_flow)
    eff_oi, N_i, delta_Htr, delta_Hy, delta_Hparc = inside_kpd(point0, d_sr, n, p, H0, inlet_mass_flow)
    ko = {
        'Показатель': ["Потери в рабочей решетки", 
              "Энергия выходной скорости", 
              "Располагаемая энегрия ступени",
              "Относительный лопаточный КПД",
              "Внутренний относительный кпд ступени", 
              "Внутренния мощьность ступени"],
        'Параметр': [ r"$\Delta H_{р} \frac{кДж}{кг}$", r"$\Delta H_{vc} \frac{кДж}{кг}$", r"$E_{0}\frac{кДж}{кг}$",  r"$\eta_{oi}$", r"$\eta_{о.д}$" ,r"$N_{oi}$, кВт" ],
    'Значение': [round(delta_Hp/1000,2), round(delta_Hvc/1000,2), E0/1000, round(eff_oi,3),round(eff,3), round(N_i/1000,1)]
    }

    df = pd.DataFrame(data=ko)
    display(df)


def data_output3(point0, d_sr, n, p, H0, inlet_mass_flow):
    W2_min_, sigma_bending, omega, sigma_stretching = calculation_trength(point0, d_sr, n, p, H0, inlet_mass_flow)
    x = {
     'Показатель': ["Момент сопротивления профиля рабочей лопатки", 
              "Напряжение изгиба лопатки", 
              "угловая скорость рабочего колеса", 
              "Напряжение растяжения лопатки"],       
     'Параметр': [r"$W2_{min}$", r"$\sigma$", r"$\omega$", r"$\sigma$"],
     'Значение': [W2_min_, sigma_bending, omega, sigma_stretching]
        }



    df10 = pd.DataFrame(data=x)
    display(df10)






