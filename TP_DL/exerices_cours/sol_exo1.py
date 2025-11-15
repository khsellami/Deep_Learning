import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# 1️⃣ Création du dataset linéaire
np.random.seed(42)
n_points = 100
weight_true = 0.3
bias_true = 0.9

X = np.linspace(0, 10, n_points)
y = weight_true * X + bias_true + np.random.randn(n_points) * 0.1  # bruit

# Split train/test 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Conversion en tensors
X_train = torch.FloatTensor(X_train).unsqueeze(1)  # shape [n,1]
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test).unsqueeze(1)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# Visualisation des données
plt.scatter(X_train, y_train, label='Train')
plt.scatter(X_test, y_test, label='Test')
plt.xlabel('X'); plt.ylabel('y'); plt.legend(); plt.title('Dataset Linéaire')
plt.tight_layout()
plt.savefig('exo1_dataset.png')
print('Saved dataset plot to exo1_dataset.png')

# 2️⃣ Création du modèle personnalisé
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Paramètres initiaux aléatoires
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        return x * self.weight + self.bias

# Instanciation et vérification du state_dict
model = LinearModel()
print(model.state_dict())

# 3️⃣ Définition de la perte et de l'optimiseur
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Boucle d'entraînement
epochs = 300
for epoch in range(1, epochs+1):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        model.eval()
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f'Epoch [{epoch}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 4️⃣ Prédictions et visualisation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

plt.scatter(X_train, y_train, label='Train')
plt.scatter(X_test, y_test, label='Test')
plt.plot(X_test, y_pred, color='red', label='Predictions')
plt.xlabel('X'); plt.ylabel('y'); plt.legend(); plt.title('Prédictions du modèle')
plt.tight_layout()
plt.savefig('exo1_predictions.png')
print('Saved prediction plot to exo1_predictions.png')

# 5️⃣ Sauvegarde et chargement du modèle
torch.save(model.state_dict(), 'linear_model.pth')

# Nouvelle instance et chargement
loaded_model = LinearModel()
loaded_model.load_state_dict(torch.load('linear_model.pth'))
loaded_model.eval()

# Vérification des prédictions
with torch.no_grad():
    y_pred_loaded = loaded_model(X_test)

print("Différence entre prédictions originales et chargées:",
      torch.abs(y_pred - y_pred_loaded).max().item())