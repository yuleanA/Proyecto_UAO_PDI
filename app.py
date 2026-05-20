import streamlit as st
import pandas as pd
import datetime
import os
import time
from PIL import Image

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="UAO Campus Inteligente", layout="wide")
ARCHIVO_CSV = "base_datos/registros.csv"

# --- CONEXIÓN CON EL MODELO DE TU COMPAÑERO ---
# Usamos un Try/Except para que tu app funcione AUNQUE tu compañero no haya terminado su parte.
try:
    from modelo.inferencia import predecir_vehiculo
except ImportError:
    # Función simulada (Mock) mientras tu compañero sube su código
    def predecir_vehiculo(imagen):
        import random
        time.sleep(1.5) # Simula el tiempo de procesamiento
        es_carro = random.random() > 0.3
        return "Carro" if es_carro else "Moto", f"{random.randint(85, 99)}%"

# --- INTERFAZ ---
st.title("🚗 UAO Campus Inteligente - Clasificación Vehicular")

# Creamos las 3 pantallas solicitadas
pantalla_inicio, pantalla_resultado, pantalla_historial = st.tabs(["📷 Inicio", "🎯 Resultado", "📋 Historial"])

with pantalla_inicio:
    st.markdown("### Cámara - Portería Principal")
    st.image("https://images.unsplash.com/photo-1494976388531-d1058494cdd8?auto=format&fit=crop&q=80&w=800", caption="Feed en vivo (Simulación)")
    
    if st.button("🔍 PROCESAR VEHÍCULO", type="primary", use_container_width=True):
        # Aquí se llama a la función de la red neuronal
        with st.spinner("Analizando imagen con Deep Learning..."):
            clase, confianza = predecir_vehiculo(None) # En la vida real aquí va la foto
            
            # Guardar el resultado temporalmente en la sesión de Streamlit
            st.session_state['ultimo_resultado'] = {'clase': clase, 'confianza': confianza}
            st.success("¡Análisis completado! Ve a la pestaña 'Resultado'.")

with pantalla_resultado:
    st.markdown("### Resultado de Inferencia")
    if 'ultimo_resultado' in st.session_state:
        res = st.session_state['ultimo_resultado']
        st.info("Bounding Box simulado generado por la red neuronal.")
        
        col1, col2 = st.columns(2)
        with col1:
            # Mostramos imagen según la clase
            img_url = "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?auto=format&fit=crop&q=80&w=400" if res['clase'] == 'Carro' else "https://images.unsplash.com/photo-1558981403-c5f9899a28bc?auto=format&fit=crop&q=80&w=400"
            st.image(img_url, caption=f"Objeto Detectado: {res['clase']}")
        
        with col2:
            st.metric(label="Clase Detectada", value=res['clase'])
            st.metric(label="Nivel de Confianza", value=res['confianza'])
            
            if st.button("✅ Guardar en Historial", type="primary"):
                # Lógica para guardar en el CSV
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
                del st.session_state['ultimo_resultado'] # Limpiar pantalla
    else:
        st.warning("Aún no se ha procesado ningún vehículo. Ve a la pestaña Inicio.")

with pantalla_historial:
    st.markdown("### Base de Datos Local")
    try:
        df_historial = pd.read_csv(ARCHIVO_CSV)
        st.dataframe(df_historial, use_container_width=True)
    except Exception as e:
        st.error("Error al leer la base de datos.")