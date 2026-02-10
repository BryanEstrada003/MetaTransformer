import numpy as np
import pandas as pd

# Supongamos que tienes un DataFrame con forma: (muestras * sensores, lecturas)
# Ejemplo: 8 muestras × 3 sensores = 24 filas, cada una con 20 lecturas

# Crear datos de ejemplo (8 muestras, 3 sensores, 20 lecturas)
num_muestras = 8
num_sensores = 3
num_lecturas = 20

# Datos simulados
data = np.random.randn(num_muestras * num_sensores, num_lecturas) * 5 + 10  # Valores ~N(10, 5)

# Convertir a DataFrame
df = pd.DataFrame(data)
df.columns = [f'Lectura_{i+1}' for i in range(num_lecturas)]
df['Muestra'] = np.repeat(range(num_muestras), num_sensores)
df['Sensor'] = ['MQ135', 'MQ4', 'MQ136'] * num_muestras

# Aplicar ruido gaussiano
noise_level = 0.02  # 2% del valor promedio
df_noisy = df.copy()

for i in range(num_lecturas):
    col = f'Lectura_{i+1}'
    # Ruido proporcional al promedio de esa columna
    noise = np.random.normal(0, noise_level * df[col].mean(), len(df))
    df_noisy[col] = df[col] + noise