# 🚗 Sistema de Clasificación Vehicular - UAO Campus Inteligente

Este proyecto implementa una solución de Visión por Computadora mediante una Red Neuronal Convolucional (CNN) para clasificar automáticamente los vehículos (Carros vs. Motos) que ingresan a la Universidad Autónoma de Occidente, optimizando la logística de parqueaderos.

## 📂 Arquitectura del Proyecto y Carpetas

El proyecto está dividido bajo el principio de separación de responsabilidades:

* **`app.py`**: Contiene la interfaz gráfica de usuario construida con Streamlit. Maneja el flujo de interacción (Inicio, Resultados y Historial) y gestiona el guardado de datos.
* **`modelo/`**: Carpeta exclusiva para el motor de Inteligencia Artificial. Contiene la arquitectura de la red (`cnn_arquitectura.py`), los pesos entrenados (`.pth`) y el script `inferencia.py` que sirve como puente de comunicación con la interfaz.
* **`base_datos/`**: Almacenamiento local. Contiene `registros.csv`, que actúa como el historial de auditoría de los vehículos ingresados.
* **`requirements.txt`**: Dependencias necesarias para ejecutar el sistema.

## Nota sobre el alcance del prototipo

El documento inicial planteaba tres objetivos específicos: preprocesamiento de imágenes, clasificación vehicular con OCR para lectura de placas, y registro en CSV. El prototipo entregado cumple el primero y el tercero completamente, y cumple parcialmente el segundo — se implementó la clasificación automática del tipo de vehículo (carro vs. moto) mediante una red neuronal convolucional propia entrenada con imágenes reales del campus, pero no se integró el módulo de lectura de placas con Tesseract OCR.
Esta decisión fue tomada conscientemente por limitaciones de tiempo y por integridad académica: Tesseract OCR implica un flujo de procesamiento distinto al trabajado en el curso, y el equipo consideró que integrarlo sin comprenderlo suficientemente comprometía la capacidad de sustentar esa parte del trabajo. La arquitectura del sistema está diseñada para que ese módulo pueda agregarse en una iteración futura sin modificar la estructura existente.
Lo que sí se entregó funciona de extremo a extremo: captura o carga de imagen → preprocesamiento → clasificación con CNN propia entrenada con datos reales → registro en CSV.

## 🚀 Cómo ejecutar el proyecto localmente

1. Instalar las dependencias:
   `pip install -r requirements.txt`
2. Ejecutar la interfaz:
   `streamlit run app.py`

## Alternativa directa

https://deepvision-uao.streamlit.app
