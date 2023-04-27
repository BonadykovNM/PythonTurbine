from typing import List, Tuple, Optional
from iapws import IAPWS97
from iapws import IAPWS97 as gas
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd 
import os

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
        """
        Функция поиска теоретического и действительного значения первой точки
        :param float p0: Давление начальной точки в Па
        :param float t0: Температура начальной точки в К
        :param float p_middle: Давление промежуточного перегрева в Па
        :param float internal_efficiency: КПД цилиндров турбины
        :return Tuple[IAPWS97, IAPWS97,IAPWS97]:Параметры теоретического значения первой точки, Параметры действительного значения первой точки
        """ 
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
        plt.plot(ks, y)
        if z > 7:
            plt.plot(coef, f_y,label = 'z = ∞')
        else:
            plt.plot(coef, f_y,label = f'z = {z}')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.show()
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
        print("Коэфициент для нахождения коэфициента", coeff)

        image = mpimg.imread("graf.png")
        plt.axis('off')
        plt.imshow(image)
        plt.show()
        cf = check_graf(z,coeff)
        print (cf)
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
    df1 = pd.DataFrame([(point0[1].P, point1[1].P, point2[1].P, point_middle[1].P),
                        (point0[1].T, point1[1].T, point2[1].T, point_middle[1].T),
                        (point0[1].h, point1[1].h, point2[1].h, point_middle[1].h),
                        (point0[1].s, point1[1].s, point2[1].s, point_middle[1].s)],
                        index=["P, МПа", "T, K", r"h, $\frac{кДж}{кг}$", r"S, $\frac{кДж}{кг * K}$ "], columns=["point 0", "point 1", "point 2", "point middle"])
    return df1

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









