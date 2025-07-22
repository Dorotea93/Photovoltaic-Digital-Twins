# Funciona y registrado 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
# Paso 1: Carga y preprocesamiento de datos
data = pd.read_excel(
    r'C:\Users\Usuario\Desktop\Trabajo_UNI\Gemelos digitales\Articulos\ART_3_redes neuronales\Datos_instalacion_6_paneles.xlsx',
    sheet_name='Conjunto Datos'
)

# Separar variables de entrada y salida
X = data[['Irrad', 'Vel', 'Temp']].values
y = data['Preal'].values.reshape(-1, 1)

# Normalizar los datos de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertir a tensores de PyTorch
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# División de datos (75% train, 25% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_tensor, y_tensor, test_size=0.25, random_state=42
)

# Crear DataLoader para entrenamiento
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Paso 2: Definición del modelo MLP con 1 capa oculta
class MLPv2(nn.Module):
    def __init__(self):
        super(MLPv2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 128),  # Capa oculta (3 inputs -> 128 neuronas)
            nn.ReLU(),
            nn.Linear(128, 1)   # Capa de salida (128 -> 1 output)
        )
        
    def forward(self, x):
        return self.layers(x)

model = MLPv2()

# Paso 3: Configuración de entrenamiento
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Paso 4: Entrenamiento con Early Stopping
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
    
    # Validación
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val)
    
    print(f'Epoch {epoch+1}/{epochs} - Val Loss: {val_loss.item():.6f}')
    
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activado!")
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('best_model.pth'))

# Paso 5: Evaluación y comparación
model.eval()
with torch.no_grad():
    y_pred = model(X_val).numpy()
    y_true = y_val.numpy()

val_mse = mean_squared_error(y_true, y_pred)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_true, y_pred)
r2_mlp = r2_score(y_true, y_pred)

# Métricas del modelo matemático 
math_mse = 0.047805
math_rmse = 0.21864
math_mae = 0.07521
math_r2 = 0.914450
joblib.dump(scaler, 'scaler.pkl')  # Guarda el scaler

print('\nComparación de métricas:')
print('                    Modelo Matemático    Modelo MLP (1 capa)')
print(f'MSE:               {math_mse:.9f}    {val_mse:.9f}')
print(f'RMSE:              {math_rmse:.9f}    {val_rmse:.9f}')
print(f'MAE:               {math_mae:.9f}    {val_mae:.9f}')
print(f'R²:                {math_r2:.9f}    {r2_mlp:.9f}')
