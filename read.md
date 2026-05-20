# 🚗 Sistema de Clasificación Vehicular - UAO Campus Inteligente

Este proyecto implementa una solución de Visión por Computadora mediante una Red Neuronal Convolucional (CNN) para clasificar automáticamente los vehículos (Carros vs. Motos) que ingresan a la Universidad Autónoma de Occidente, optimizando la logística de parqueaderos.

## 📂 Arquitectura del Proyecto y Carpetas

El proyecto está dividido bajo el principio de separación de responsabilidades:

* **`app.py`**: Contiene la interfaz gráfica de usuario construida con Streamlit. Maneja el flujo de interacción (Inicio, Resultados y Historial) y gestiona el guardado de datos.
* **`modelo/`**: Carpeta exclusiva para el motor de Inteligencia Artificial. Contiene la arquitectura de la red (`cnn_arquitectura.py`), los pesos entrenados (`.pth`) y el script `inferencia.py` que sirve como puente de comunicación con la interfaz.
* **`base_datos/`**: Almacenamiento local. Contiene `registros.csv`, que actúa como el historial de auditoría de los vehículos ingresados.
* **`requirements.txt`**: Dependencias necesarias para ejecutar el sistema.

## 🚀 Cómo ejecutar el proyecto localmente

1. Instalar las dependencias:
   `pip install -r requirements.txt`
2. Ejecutar la interfaz:
   `streamlit run app.py`