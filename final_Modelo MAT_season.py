import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

excel_path = r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_3_redes neuronales\Datos_mate_metricas_1.xlsx'
sheet = 'Hoja1'

# Leer datos
df = pd.read_excel(excel_path, sheet_name=sheet)

# Obtener estaciones únicas
estaciones = df['Estacion'].dropna().unique()

print('\n== Métricas de validación por estación (modelo matemático)==\n')
for estacion in estaciones:
    tipo_validacion = f'Validación_{estacion}'
    # Seleccionar únicamente la validación de la estación
    mask = df['Tipo'] == tipo_validacion
    if mask.sum() == 0:
        print(f"Estación {estacion}: no hay filas de validación.")
        continue
    
    y_true = df.loc[mask, 'Preal_inv (kW)'].values
    y_pred = df.loc[mask, 'P DT ec1 (kW)'].values

    val_mse = mean_squared_error(y_true, y_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Estación: {estacion}")
    print(f"MSE   : {val_mse:.6f}")
    print(f"RMSE  : {val_rmse:.6f}")
    print(f"MAE   : {val_mae:.6f}")
    print(f"R²    : {r2:.6f}")
    print('-'*40)
