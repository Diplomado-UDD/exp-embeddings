import streamlit as st
import numpy as np
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv
import os
import plotly.graph_objects as go

# Cargar variables de entorno
load_dotenv()


@st.cache_data(show_spinner=False)
def get_embedding(text: str, _client: OpenAI, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Obtiene el vector embedding de un texto usando la API de OpenAI.
    Usa caché para evitar llamadas repetidas a la API y ahorrar tokens.
    El prefijo _ en _client indica a Streamlit que no use este parámetro para el hash del caché.
    """
    text = text.replace("\n", " ")
    response = _client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula la similitud coseno entre dos vectores.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_most_similar(target_vector: np.ndarray, candidates: Dict[str, np.ndarray]) -> List[tuple]:
    """
    Encuentra las palabras candidatas más similares al vector objetivo.
    """
    similarities = []

    for word, vector in candidates.items():
        sim = cosine_similarity(target_vector, vector)
        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


def create_similarity_chart(resultados: List[tuple], formula: str):
    """
    Crea un gráfico de barras horizontal con los resultados de similitud.
    """
    palabras = [r[0] for r in resultados]
    similitudes = [r[1] for r in resultados]

    # Colores: verde para el más alto, azul para el resto
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(palabras))]

    fig = go.Figure(go.Bar(
        x=similitudes,
        y=palabras,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{s:.4f}' for s in similitudes],
        textposition='auto',
    ))

    fig.update_layout(
        title=f"Similitud con el Resultado ({formula})",
        xaxis_title="Similitud Coseno",
        yaxis_title="Palabra Candidata",
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def main():
    st.set_page_config(
        page_title="Álgebra de Palabras con Embeddings",
        page_icon="🔢",
        layout="wide"
    )

    # Limpiar resultados previos cuando cambian los parámetros
    if 'last_config' not in st.session_state:
        st.session_state.last_config = None

    st.title("🔢 Álgebra de Palabras con Embeddings")

    # Verificar API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ No se encontró OPENAI_API_KEY en el archivo .env")
        st.info("Por favor, crea un archivo .env con tu API key de OpenAI")
        st.code("OPENAI_API_KEY=tu-api-key-aqui", language="bash")
        return

    # Sidebar con configuración
    with st.sidebar:
        st.header("⚙️ Configuración")

        # Ejemplos predefinidos
        st.subheader("Ejemplos Predefinidos")
        ejemplo = st.selectbox(
            "Selecciona un ejemplo",
            ["Madrid - España + Francia",
             "Rey - Hombre + Mujer",
             "Toro - Macho + Hembra",
             "Personalizado"]
        )

        # Configurar según ejemplo
        if ejemplo == "Madrid - España + Francia":
            palabra1_default = "Madrid"
            palabra2_default = "España"
            palabra3_default = "Francia"
            candidatas_default = "París\nBerlín\nRoma\nLondres\nLisboa\nBruselas\nAmsterdam"
        elif ejemplo == "Rey - Hombre + Mujer":
            palabra1_default = "Rey"
            palabra2_default = "Hombre"
            palabra3_default = "Mujer"
            candidatas_default = "Reina\nPríncipe\nPrincesa\nDama\nSeñora\nCaballero\nMonarca"
        elif ejemplo == "Toro - Macho + Hembra":
            palabra1_default = "Toro"
            palabra2_default = "Macho"
            palabra3_default = "Hembra"
            candidatas_default = "Vaca\nCampo\nLeche\nGranja\nAnimal\nHierba"
        else:
            palabra1_default = "Madrid"
            palabra2_default = "España"
            palabra3_default = "Francia"
            candidatas_default = "París\nBerlín\nRoma"

        st.subheader("Palabras Base")
        palabra1 = st.text_input("Primera palabra", value=palabra1_default)
        palabra2 = st.text_input("Restar", value=palabra2_default)
        palabra3 = st.text_input("Sumar", value=palabra3_default)

        st.subheader("Palabras Candidatas")
        candidatas_text = st.text_area(
            "Una por línea",
            value=candidatas_default,
            height=150
        )

        ejecutar = st.button("🚀 Ejecutar Álgebra", type="primary", use_container_width=True)

    # Mostrar encabezado dinámico
    st.markdown(f"### Demostración: {palabra1} - {palabra2} + {palabra3} = ?")

    # Mensaje explicativo dinámico según el ejemplo
    if ejemplo == "Madrid - España + Francia":
        st.markdown("""
        Esta demostración muestra cómo los embeddings capturan **relaciones semánticas** entre palabras.
        El modelo no "sabe" geografía, solo entiende que la relación **Madrid → España** es similar a **París → Francia**.
        """)
    elif ejemplo == "Rey - Hombre + Mujer":
        st.markdown("""
        Esta demostración muestra cómo los embeddings capturan **relaciones semánticas** entre palabras.
        El modelo no "sabe" género, solo entiende que la relación **Rey → Reina** es similar a **Hombre → Mujer**.
        """)
    elif ejemplo == "Toro - Macho + Hembra":
        st.markdown("""
        Esta demostración muestra cómo los embeddings capturan **relaciones semánticas** entre palabras.
        El modelo no "sabe" biología, solo entiende que la relación **Toro → Vaca** es similar a **Macho → Hembra**.
        """)
    else:
        st.markdown("""
        Esta demostración muestra cómo los embeddings capturan **relaciones semánticas** entre palabras.
        Los embeddings aprenden relaciones matemáticas entre conceptos.
        """)

    # Detectar cambios en la configuración
    current_config = f"{palabra1}|{palabra2}|{palabra3}|{candidatas_text}"

    # Área principal
    if ejecutar:
        # Validar que hay palabras candidatas
        palabras_candidatas = [p.strip() for p in candidatas_text.split('\n') if p.strip()]

        if not palabras_candidatas:
            st.error("⚠️ Error: Debes agregar al menos una palabra candidata en la sección 'Una por línea'")
            st.stop()

        if not palabra1.strip() or not palabra2.strip() or not palabra3.strip():
            st.error("⚠️ Error: Las tres palabras base son obligatorias")
            st.stop()

        # Actualizar configuración
        st.session_state.last_config = current_config
        with st.spinner("🔄 Procesando embeddings..."):
            try:
                client = OpenAI(api_key=api_key)

                # Paso 1: Obtener embeddings base
                st.subheader("📍 Paso 1: Obteniendo Embeddings Base")
                col1, col2, col3 = st.columns(3)

                with col1:
                    with st.status(f"Procesando '{palabra1}'...", expanded=True) as status:
                        v_palabra1 = get_embedding(palabra1, client)
                        st.write(f"✅ Vector de {len(v_palabra1)} dimensiones")
                        status.update(label=f"'{palabra1}' completado", state="complete")

                with col2:
                    with st.status(f"Procesando '{palabra2}'...", expanded=True) as status:
                        v_palabra2 = get_embedding(palabra2, client)
                        st.write(f"✅ Vector de {len(v_palabra2)} dimensiones")
                        status.update(label=f"'{palabra2}' completado", state="complete")

                with col3:
                    with st.status(f"Procesando '{palabra3}'...", expanded=True) as status:
                        v_palabra3 = get_embedding(palabra3, client)
                        st.write(f"✅ Vector de {len(v_palabra3)} dimensiones")
                        status.update(label=f"'{palabra3}' completado", state="complete")

                # Paso 2: Álgebra
                st.subheader("🧮 Paso 2: Realizando Álgebra de Vectores")
                st.code(f"v_resultado = {palabra1} - {palabra2} + {palabra3}", language="python")

                v_resultado = v_palabra1 - v_palabra2 + v_palabra3
                st.success(f"✅ Vector resultado calculado ({len(v_resultado)} dimensiones)")

                # Paso 3: Candidatos
                st.subheader("🔍 Paso 3: Obteniendo Embeddings de Candidatos")

                progress_bar = st.progress(0)
                candidatos = {}

                for i, palabra in enumerate(palabras_candidatas):
                    candidatos[palabra] = get_embedding(palabra, client)
                    progress_bar.progress((i + 1) / len(palabras_candidatas))

                st.success(f"✅ {len(candidatos)} palabras candidatas procesadas")

                # Paso 4: Resultados
                st.subheader("🎯 Paso 4: Resultados")
                resultados = find_most_similar(v_resultado, candidatos)

                # Mostrar ganador destacado
                ganador = resultados[0]
                st.success(f"### 🏆 Palabra más cercana: **{ganador[0]}** (similitud: {ganador[1]:.4f})")

                # Gráfico de similitudes
                st.subheader("📊 Ranking de Similitudes")
                fig = create_similarity_chart(resultados, f"{palabra1} - {palabra2} + {palabra3}")
                st.plotly_chart(fig, use_container_width=True)

                # Mostrar vectores y diferencias
                st.subheader("🔬 Análisis Detallado de Vectores")

                # Determinar palabra esperada (primer candidato o específico)
                palabra_esperada = resultados[0][0]  # El ganador por defecto

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 1️⃣ Vector Resultado")
                    st.caption(f"{palabra1} - {palabra2} + {palabra3}")
                    with st.expander("Ver primeras 10 dimensiones"):
                        st.code(str(v_resultado[:10]))

                with col2:
                    st.markdown(f"#### 2️⃣ Embedding de '{palabra_esperada}'")
                    if palabra_esperada in candidatos:
                        v_esperada = candidatos[palabra_esperada]
                        sim_esperada = cosine_similarity(v_resultado, v_esperada)
                        st.metric("Similitud con resultado", f"{sim_esperada:.4f}")
                        with st.expander("Ver primeras 10 dimensiones"):
                            st.code(str(v_esperada[:10]))

                with col3:
                    st.markdown("#### 3️⃣ Diferencia")
                    if palabra_esperada in candidatos:
                        v_esperada = candidatos[palabra_esperada]
                        diferencia = v_resultado - v_esperada
                        distancia_euclidiana = np.linalg.norm(diferencia)
                        st.metric("Distancia Euclidiana", f"{distancia_euclidiana:.4f}")
                        with st.expander("Ver primeras 10 dimensiones"):
                            st.code(str(diferencia[:10]))

                # Comparación de similitudes
                st.markdown("#### 📊 Top 5 Resultados")

                comparacion_data = []
                for i, (palabra, similitud) in enumerate(resultados[:5], 1):
                    comparacion_data.append({
                        "Posición": f"{i}°",
                        "Palabra": palabra,
                        "Similitud Coseno": f"{similitud:.6f}",
                        "Mejor Match": "🏆" if i == 1 else ""
                    })

                st.dataframe(comparacion_data, use_container_width=True, hide_index=True)

                # Tabla de resultados completa
                with st.expander("📊 Ver tabla detallada de todos los resultados"):
                    st.dataframe(
                        {
                            "Posición": list(range(1, len(resultados) + 1)),
                            "Palabra": [r[0] for r in resultados],
                            "Similitud": [f"{r[1]:.6f}" for r in resultados]
                        },
                        use_container_width=True
                    )

                # Conclusión
                st.divider()
                st.subheader("📝 Conclusión")
                st.info(f"""
                **El modelo NO 'sabe' qué es un {ganador[0].lower()}.**

                Solo entiende que la relación **{palabra1} → {ganador[0]}** es similar a **{palabra2} → {palabra3}**.

                Los embeddings capturan **RELACIONES**, no hechos.
                """)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    elif st.session_state.last_config == current_config and st.session_state.last_config is not None:
        # Mostrar mensaje para re-ejecutar si cambió la configuración
        st.warning("⚠️ La configuración ha cambiado. Presiona **Ejecutar Álgebra** para ver los nuevos resultados.")
    else:
        # Mensaje inicial
        st.info("👈 Configura las palabras en el panel lateral y presiona **Ejecutar Álgebra** para comenzar")

        # Mostrar ejemplo visual
        st.subheader("💡 ¿Cómo funciona?")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("### 1️⃣ Palabras → Vectores")
            st.markdown("Cada palabra se convierte en un vector de números (embeddings)")
            st.code("Toro = [0.23, -0.15, 0.89, ...]", language="python")

        with col2:
            st.markdown("### 2️⃣ Álgebra de Vectores")
            st.markdown("Realizamos operaciones matemáticas con los vectores")
            st.code("Resultado = Toro - Macho + Hembra", language="python")

        with col3:
            st.markdown("### 3️⃣ Buscar Similares")
            st.markdown("Encontramos qué palabra candidata está más cerca del resultado")
            st.code("Más cercano: Vaca", language="python")


if __name__ == "__main__":
    main()
