import streamlit as st
import pandas as pd
import datetime
import os
import time

# --- CONFIGURACIÓN DE PÁGINA ---
# Cambiamos el nombre de la pestaña y pusimos layout="centered" para que no se estire feo en PC
st.set_page_config(page_title="DeepVision UAO | Control Vehicular", layout="centered", page_icon="🚘")
ARCHIVO_CSV = "base_datos/registros.csv"

# --- CONEXIÓN CON EL MODELO ---
try:
    from modelo.inferencia import predecir_vehiculo
except ImportError:
    # Función simulada mejorada (ahora simula fallos por fotos borrosas)
    def predecir_vehiculo(imagen):
        import random
        time.sleep(1.5)
        # Simulamos un 20% de probabilidad de que la foto no se entienda
        if random.random() < 0.20:
            return "No Reconocido", 0
        es_carro = random.random() > 0.3
        return "Carro" if es_carro else "Moto", random.randint(85, 99)

# --- INTERFAZ PRINCIPAL ---
st.title("🚘 DeepVision UAO")
st.subheader("Sistema de Auditoría y Clasificación Vehicular")

pantalla_inicio, pantalla_resultado, pantalla_historial = st.tabs(["📸 Inicio", "🎯 Resultado", "📋 Historial"])

with pantalla_inicio:
    st.info("Sistema en línea. Esperando vehículo en portería...")
    
    # La cámara real (eliminamos la imagen falsa)
    imagen_capturada = st.camera_input("Captura de Garita Principal")
    
    imagen_subida = st.file_uploader("O carga una imagen desde tu equipo", type=["jpg", "jpeg", "png"])
    
    imagen_final = imagen_capturada if imagen_capturada is not None else imagen_subida

    if imagen_final is not None:
        if st.button("🔍 PROCESAR VEHÍCULO", type="primary", use_container_width=True):
            
            with st.spinner("Ejecutando Inferencia (CNN)..."):
                clase, confianza = predecir_vehiculo(imagen_final) 
                
                # SISTEMA DE ERROR: Si la red no lo reconoce o la confianza es muy baja
                if clase == "No Reconocido" or confianza < 50:
                    st.error("⚠️ La foto es difícil de reconocer o determinar su tipo. Por favor, asegúrese de que el vehículo esté enfocado e intente tomar otra captura.")
                else:
                    st.session_state['ultimo_resultado'] = {'clase': clase, 'confianza': f"{confianza}%"}
                    st.success("✅ ¡Análisis exitoso! Ve a la pestaña 'Resultado'.")

with pantalla_resultado:
    st.markdown("### Resultado de Detección")
    if 'ultimo_resultado' in st.session_state:
        res = st.session_state['ultimo_resultado']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Clase Detectada", value=res['clase'])
        with col2:
            st.metric(label="Nivel de Confianza", value=res['confianza'])
            
        st.divider()
            
        if st.button("💾 Guardar Registro en Base de Datos", type="primary", use_container_width=True):
            df = pd.read_csv(ARCHIVO_CSV)
            nuevo_id = 1 if len(df) == 0 else df['ID'].max() + 1
            nuevo_registro = pd.DataFrame([{
                'ID': nuevo_id,
                'Fecha': datetime.date.today().strftime("%Y-%m-%d"),
                'Hora': datetime.datetime.now().strftime("%H:%M:%S"),
                'Clase': res['clase'],
                'Confianza': res['confianza']
            }])
            df = pd.concat([df, nuevo_registro], ignore_index=True)
            df.to_csv(ARCHIVO_CSV, index=False)
            st.success("Registro guardado con éxito. Revisa la pestaña Historial.")
            del st.session_state['ultimo_resultado']
    else:
        st.warning("No hay resultados recientes. Toma una foto en la pestaña de Inicio.")

with pantalla_historial:
    st.markdown("### Auditoría Local (CSV)")
    try:
        df_historial = pd.read_csv(ARCHIVO_CSV)
        st.dataframe(df_historial, use_container_width=True)
    except Exception as e:
        st.error("Error al leer la base de datos.")