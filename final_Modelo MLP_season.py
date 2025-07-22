import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ------------- Configuración de paths ----------------
excel_path = r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_3_redes neuronales\Datos_mate_metricas_1.xlsx'
sheet = 'Hoja1'

# ------------- Función para mapear fecha a estación -----------
def fecha_a_estacion(fecha):
    mes = fecha.month
    dia = fecha.day
    if ((mes == 12 and dia >= 21) or (mes <= 3 and (mes != 3 or dia < 21))):
        return 'Invierno'
    elif ((mes == 3 and dia >= 21) or (mes < 6) or (mes == 6 and dia < 21)):
        return 'Primavera'
    elif ((mes == 6 and dia >= 21) or (mes < 9) or (mes == 9 and dia < 23)):
        return 'Verano'
    else:
        return 'Otoño'

# ------------- Cargar datos y añadir columna de estación -------
df = pd.read_excel(excel_path, sheet_name=sheet)
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')

df['Estacion'] = df['Fecha'].apply(fecha_a_estacion)
df['Tipo'] = np.nan

# -------------- MLP Architecture -------------
class MLPv2(nn.Module):
    def __init__(self):
        super(MLPv2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layers(x)

# -------------- Por cada estación -------------
for estacion, grupo in df.groupby('Estacion'):
    if grupo.shape[0] < 8:  # Evita conjuntos pequeños que puedan causar error
        print(f"Saltando estación {estacion}: muy pocos datos ({grupo.shape[0]})")
        continue

    idx = grupo.index
    # Features y target
    X = grupo[['Irrad', 'Wind speed(m/s)', 'Temp (ºC)']].values
    y = grupo['Preal_inv (kW)'].values.reshape(-1, 1)
    indices = np.array(grupo.index)

    # Normalización estándar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_tensor, y_tensor, indices, test_size=0.25, random_state=42
    )

    # Marcar Tipo en el DataFrame original
    df.loc[idx_train, 'Tipo'] = f"Entrenamiento_{estacion}"
    df.loc[idx_val, 'Tipo'] = f"Validación_{estacion}"

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = MLPv2()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 10
    counter = 0
    epochs = 200

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).numpy()
        y_true = y_val.numpy()

    val_mse = mean_squared_error(y_true, y_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_true, y_pred)
    r2_mlp = r2_score(y_true, y_pred)

    print(f"\nEstación: {estacion}")
    print(f"MSE   : {val_mse:.6f}")
    print(f"RMSE  : {val_rmse:.6f}")
    print(f"MAE   : {val_mae:.6f}")
    print(f"R²    : {r2_mlp:.6f}")
    print('-'*40)

# -------- Guardar el Excel actualizado -------
df.to_excel(excel_path, sheet_name=sheet, index=False)
print("\nExcel actualizado con columnas 'Estacion' y 'Tipo'.")

