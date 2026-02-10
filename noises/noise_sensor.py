import numpy as np
import pandas as pd

# Tus datos
data = {
    'MQ135': [6.348, 5.378, 5.08, 5.112, 5.203, 5.087, 7.074, 6.184],
    'MQ4': [0.411, 0.54, 0.468, 0.461, 0.639, 0.428, 0.461, 0.601],
    'MQ136': [0.455, 0.46, 0.42, 0.458, 0.392, 0.36, 0.413, 0.437]
}

df = pd.DataFrame(data)

# Parámetros del ruido
noise_level = 0.01  # 1% del valor de la señal (ajustable)
np.random.seed(42)  # Para reproducibilidad

# Aplicar ruido gaussiano a cada columna
df_noisy = df.apply(lambda col: col + np.random.normal(0, noise_level * col.mean(), len(col)))

print("Datos originales:")
print(df)
print("\nDatos con ruido gaussiano:")
print(df_noisy)