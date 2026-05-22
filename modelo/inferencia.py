import torch
from torchvision import transforms
from PIL import Image
import io
from modelo.cnn_arquitectura import ClasificadorVehiculo
import os

# --- CONFIGURACIÓN ---
ARCHIVO_PESOS = os.path.join(os.path.dirname(__file__), "pesos_finales.pth")

# Misma transformación que en el entrenamiento
transformacion = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Cargar modelo una sola vez al iniciar la app
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = None
clases = None

def cargar_modelo():
    global modelo, clases
    if modelo is None:
        checkpoint = torch.load(ARCHIVO_PESOS, map_location=dispositivo)
        clases = checkpoint['clases']
        modelo = ClasificadorVehiculo().to(dispositivo)
        modelo.load_state_dict(checkpoint['model_state_dict'])
        modelo.eval()

def predecir_vehiculo(imagen_bytes):
    """
    Recibe la imagen de Streamlit (bytes) y retorna (clase, confianza).
    clase    -> "Carro" o "Moto"
    confianza -> número entre 0 y 100
    """
    cargar_modelo()

    # Convertir bytes a imagen PIL
    imagen = Image.open(io.BytesIO(imagen_bytes.read())).convert("RGB")

    # Aplicar transformaciones
    tensor = transformacion(imagen).unsqueeze(0).to(dispositivo)

    # Inferencia
    with torch.no_grad():
        salidas = modelo(tensor)
        probabilidades = torch.softmax(salidas, dim=1)
        confianza, indice = torch.max(probabilidades, 1)

    clase_raw = clases[indice.item()]
    confianza_pct = int(confianza.item() * 100)

    # Traducir nombre de carpeta a etiqueta legible
    etiquetas = {"carros": "Carro", "motos": "Moto"}
    clase = etiquetas.get(clase_raw.lower(), clase_raw.capitalize())

    return clase, confianza_pct