# 1Funciona y registrado - Modelo GRU
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Función para crear secuencias temporales para GRU
def create_sequences(X, y, sequence_length):
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    
    return np.array(X_sequences), np.array(y_sequences)

# Paso 1: Carga y preprocesamiento de datos
data = pd.read_excel(
    r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_3_redes neuronales\Datos_instalacion_6_paneles.xlsx',
    sheet_name='Conjunto Datos'
)

# --- MODIFICACIÓN: Guardar copia del DataFrame original ---
df_original = data.copy()

print(f"Datos cargados: {len(data)} registros")

# Separar variables de entrada y salida
X = data[['Irrad', 'Vel', 'Temp']].values
y = data['Preal'].values.reshape(-1, 1)

# Normalizar los datos de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 2: Crear secuencias temporales (ventana de 6 pasos = 30 minutos)
sequence_length = 6
X_sequences, y_sequences = create_sequences(X_scaled, y, sequence_length)

print(f"Secuencias creadas: {X_sequences.shape} -> {y_sequences.shape}")
print(f"Forma de cada secuencia: {X_sequences.shape[1:]} (6 pasos x 3 variables)")

# --- MODIFICACIÓN: Obtener índices de la división ---
sequence_indices = np.arange(len(X_sequences))
X_train, X_val, y_train, y_val, idx_train_seq, idx_val_seq = train_test_split(
    X_sequences, y_sequences, sequence_indices, test_size=0.25, random_state=42
)

# --- MODIFICACIÓN: Identificar datos originales usados en validación ---
validation_original_indices = set()
for seq_idx in idx_val_seq:
    # Cada secuencia de validación usa datos de los índices originales [seq_idx a seq_idx+sequence_length]
    validation_original_indices.update(range(seq_idx, seq_idx + sequence_length + 1))

# Marcar datos en el DataFrame original
df_original['Tipo'] = 'Entrenamiento'  # Valor por defecto
df_original.loc[list(validation_original_indices), 'Tipo'] = 'Validación'

# Guardar Excel con identificación de datos
output_path = r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_3_redes neuronales\Datos_identificados_GRU.xlsx'
df_original.to_excel(output_path, index=False, sheet_name='Datos con Identificación')
print(f"\nArchivo Excel guardado en: {output_path}")

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

print(f"Datos de entrenamiento: {X_train.shape[0]} secuencias")
print(f"Datos de validación: {X_val.shape[0]} secuencias")

# Crear DataLoader para entrenamiento
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Paso 3: Definición del modelo GRU
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
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        # Tomar solo la última salida de la secuencia
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

model = GRUModel(input_size=3, hidden_size=50)
print(f"Modelo GRU creado con {sum(p.numel() for p in model.parameters())} parámetros")

# Paso 4: Configuración de entrenamiento
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Paso 5: Entrenamiento con Early Stopping
print("\n=== INICIANDO ENTRENAMIENTO GRU ===")
best_val_loss = float('inf')
patience = 10
counter = 0
epochs = 200

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validación
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val)
    
    # Imprimir progreso cada 20 épocas
    if (epoch + 1) % 20 == 0:
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss.item():.6f}')
    
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_gru.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping activado en época {epoch+1}!")
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('best_model_gru.pth'))
print("Mejor modelo GRU cargado")

# Paso 6: Evaluación y comparación
model.eval()
with torch.no_grad():
    y_pred = model(X_val).numpy()
    y_true = y_val.numpy()

val_mse = mean_squared_error(y_true, y_pred)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_true, y_pred)
r2_gru = r2_score(y_true, y_pred)

#Tras obtener y_true, y_pred y r2_model
df_results = pd.DataFrame({
    'Actual': y_true.flatten(),
    'Predicted': y_pred.flatten(),
    'R2': r2_gru
})

# Guardar CSV según el modelo
df_results.to_csv('gru_results.csv', index=False)   # Para GRU

# Métricas del modelo matemático 
math_mse = 0.047805
math_rmse = 0.21864
math_mae = 0.07521
math_r2 = 0.914450

print('\n' + '='*60)
print('RESULTADOS FINALES - MODELO GRU')
print('='*60)
print('Comparación de métricas:')
print('                    Modelo Matemático    Modelo GRU')
print(f'MSE:               {math_mse:.9f}    {val_mse:.9f}')
print(f'RMSE:              {math_rmse:.9f}    {val_rmse:.9f}')
print(f'MAE:               {math_mae:.9f}    {val_mae:.9f}')
print(f'R²:                {math_r2:.9f}    {r2_gru:.9f}')

# Calcular mejora porcentual vs modelo matemático
mejora_rmse = ((math_rmse - val_rmse) / math_rmse) * 100
mejora_r2 = ((r2_gru - math_r2) / math_r2) * 100

print(f'\nMEJORA vs MODELO MATEMÁTICO:')
print(f'RMSE: {mejora_rmse:+.2f}% ({"Mejor" if mejora_rmse > 0 else "Peor"})')
print(f'R²:   {mejora_r2:+.2f}% ({"Mejor" if mejora_r2 > 0 else "Peor"})')

print('\n=== GUARDADO COMPLETADO ===')
print('Archivo guardado: best_model_gru.pth')

# AÑADIR ESTO AL FINAL
import joblib
joblib.dump(scaler, 'scaler_gru.pkl')
print('Scaler guardado: scaler_gru.pkl')

# Tras obtener y_true, y_pred y r2_gru
df_results = pd.DataFrame({
    'Actual': y_true.flatten(),
    'Predicted': y_pred.flatten(),
    'R2': r2_gru
})

# Guardar CSV según el modelo
df_results.to_csv('gru_results.csv', index=False)   # Para GRU
print("Predicciones de validación guardadas en gru_results.csv")
