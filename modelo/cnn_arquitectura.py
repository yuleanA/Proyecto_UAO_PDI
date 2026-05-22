import torch
import torch.nn as nn

class ClasificadorVehiculo(nn.Module):
    def __init__(self):
        super(ClasificadorVehiculo, self).__init__()

        # PARTE 1: Extractor de características
        # Estas capas aprenden a detectar formas en la imagen
        # (bordes, ruedas, espejos, etc.)
        self.extractor = nn.Sequential(
            # Bloque 1: detecta bordes y texturas básicas
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 2: detecta formas más complejas
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 3: detecta patrones de alto nivel (silueta del vehículo)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # PARTE 2: Clasificador
        # Toma las características y decide: ¿carro o moto?
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),       # evita que la red "memorice" las fotos
            nn.Linear(256, 2)      # 2 salidas: carro (0) o moto (1)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.clasificador(x)
        return x