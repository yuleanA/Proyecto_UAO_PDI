import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from modelo.cnn_arquitectura import ClasificadorVehiculo
import os

# --- CONFIGURACIÓN ---
CARPETA_DATOS = "datos"
CARPETA_MODELO = "modelo"
ARCHIVO_PESOS = os.path.join(CARPETA_MODELO, "pesos_finales.pth")
EPOCAS = 10
BATCH = 32
LEARNING_RATE = 0.001

# --- PREPARACIÓN DE IMÁGENES ---
# Todas las fotos se redimensionan a 128x128 y se normalizan
transformacion = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

print("Cargando imágenes...")
dataset = datasets.ImageFolder(CARPETA_DATOS, transform=transformacion)
clases = dataset.classes
print(f"Clases encontradas: {clases}")
print(f"Total de imágenes: {len(dataset)}")

# 80% entrenamiento, 20% validación
total = len(dataset)
n_entrenamiento = int(total * 0.8)
n_validacion = total - n_entrenamiento
datos_entrenamiento, datos_validacion = torch.utils.data.random_split(
    dataset, [n_entrenamiento, n_validacion]
)

cargador_entrenamiento = DataLoader(datos_entrenamiento, batch_size=BATCH, shuffle=True)
cargador_validacion = DataLoader(datos_validacion, batch_size=BATCH)

# --- MODELO ---
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {dispositivo}")

modelo = ClasificadorVehiculo().to(dispositivo)
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

# --- ENTRENAMIENTO ---
print("\nIniciando entrenamiento...\n")
mejor_precision = 0.0

for epoca in range(EPOCAS):
    # Fase de entrenamiento
    modelo.train()
    perdida_total = 0
    for imagenes, etiquetas in cargador_entrenamiento:
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()

    # Fase de validación
    modelo.eval()
    correctas = 0
    total_val = 0
    with torch.no_grad():
        for imagenes, etiquetas in cargador_validacion:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            salidas = modelo(imagenes)
            _, predicciones = torch.max(salidas, 1)
            total_val += etiquetas.size(0)
            correctas += (predicciones == etiquetas).sum().item()

    precision = 100 * correctas / total_val
    perdida_promedio = perdida_total / len(cargador_entrenamiento)

    print(f"Época {epoca+1}/{EPOCAS} | Pérdida: {perdida_promedio:.4f} | Precisión: {precision:.1f}%")

    # Guardar si es el mejor hasta ahora
    if precision > mejor_precision:
        mejor_precision = precision
        torch.save({
            'model_state_dict': modelo.state_dict(),
            'clases': clases
        }, ARCHIVO_PESOS)
        print(f"  -> Modelo guardado (mejor hasta ahora: {mejor_precision:.1f}%)")

print(f"\nEntrenamiento terminado. Mejor precisión: {mejor_precision:.1f}%")
print(f"Pesos guardados en: {ARCHIVO_PESOS}")