import pandas as pd
import numpy as np
from scipy import interpolate

# Leer CSV sin header
# Asume que la primera columna es X y la segunda es Y
data = pd.read_csv('Default.csv', header=None,sep=';', decimal= ',')

# Asignar nombres a las columnas para claridad
x = data[0].values  # Primera columna = X
y = data[1].values  # Segunda columna = Y

# Crear la función de interpolación
f = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")

# Tu valor X específico
x_target = 1.8
y_estimated = f(x_target)

print(f"El valor Y estimado para X={x_target} es {y_estimated}")