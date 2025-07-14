# modelo_entrenamiento.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Cargar dataset
df = pd.read_csv("estudiantes_aprueba_final.csv")
X = df[["horas_estudio", "clases_asistidas", "tareas_entregadas"]]
y = df["aprueba"]

# 2. Normalizar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Implementar modelo
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# 5. Predicciones y métricas
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión del modelo: {accuracy:.2f}")
print(f"Número de iteraciones: {mlp.n_iter_}")

# 6. Guardar modelo y scaler
joblib.dump(mlp, "modelo_guardado.pkl")
joblib.dump(scaler, "scaler_guardado.pkl")
