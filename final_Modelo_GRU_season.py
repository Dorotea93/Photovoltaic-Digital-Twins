import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----- Parámetros -----
excel_path = r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_3_redes neuronales\Datos_mate_metricas_1.xlsx'
sheet = 'Hoja1'
sequence_length = 6  # Por ejemplo, 6 pasos de 15 minutos

# Función para asignar estación por fecha
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

# Función para crear secuencias temporales para GRU
def create_sequences(X, y, sequence_length):
    X_sequences = []
    y_sequences = []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    return np.array(X_sequences), np.array(y_sequences)

# Carga y procesamiento inicial
df = pd.read_excel(excel_path, sheet_name=sheet)
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
df['Estacion'] = df['Fecha'].apply(fecha_a_estacion)
df['Tipo'] = np.nan

# Definición de la arquitectura GRU
class GRUModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          dropout=dropout if num_layers > 1 else 0, 
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

# Procesamiento estación por estación
for estacion, grupo in df.groupby('Estacion'):
    if len(grupo) <= sequence_length + 8:
        print(f"Saltando {estacion}: muy pocos datos para secuencias GRU ({len(grupo)})")
        continue

    idx = grupo.index
    X = grupo[['Irrad', 'Wind speed(m/s)', 'Temp (ºC)']].values
    y = grupo['Preal_inv (kW)'].values.reshape(-1, 1)

    # Normalización dentro de la estación
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Creación de secuencias
    X_sequences, y_sequences = create_sequences(X_scaled, y, sequence_length)
    sequence_indices = np.arange(len(X_sequences))

    X_train, X_val, y_train, y_val, idx_train_seq, idx_val_seq = train_test_split(
        X_sequences, y_sequences, sequence_indices, test_size=0.25, random_state=42
    )

    # Identificar índices originales usados en validación
    validation_original_indices = set()
    for seq_idx in idx_val_seq:
        validation_original_indices.update(idx[seq_idx:seq_idx + sequence_length + 1])

    # Marcar el DataFrame con los conjuntos
    df.loc[idx, 'Tipo'] = f"Entrenamiento_{estacion}"
    df.loc[list(validation_original_indices), 'Tipo'] = f"Validación_{estacion}"

    # Convertir a tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Definir modelo GRU
    model = GRUModel(input_size=3, hidden_size=50)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience, counter = 10, 0
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
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t)
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
        y_pred = model(X_val_t).numpy()
        y_true = y_val_t.numpy()

    val_mse = mean_squared_error(y_true, y_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_true, y_pred)
    r2_gru = r2_score(y_true, y_pred)

    print(f"\nEstación: {estacion}")
    print(f"MSE   : {val_mse:.6f}")
    print(f"RMSE  : {val_rmse:.6f}")
    print(f"MAE   : {val_mae:.6f}")
    print(f"R²    : {r2_gru:.6f}")
    print('-'*40)

  

# Guardar Excel actualizado
df.to_excel(excel_path, sheet_name=sheet, index=False)
print("\nExcel actualizado con columnas 'Estacion' y 'Tipo' para GRU estacional.")
