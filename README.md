# Demostración: Álgebra de Palabras con Embeddings

Demo interactiva para enseñar cómo los embeddings capturan relaciones semánticas entre palabras mediante operaciones algebraicas.

## Descripción

Esta aplicación web demuestra el concepto de "álgebra de palabras" usando embeddings de OpenAI. Los estudiantes pueden ver cómo operaciones matemáticas con vectores de palabras capturan relaciones semánticas.

**Idioma**: Esta demo está diseñada para usar **español chileno**. Los ejemplos y resultados están optimizados para este dialecto.

## Características

- Álgebra vectorial con palabras (ej: Madrid - España + Francia = París)
- Visualización de similitudes con gráficos interactivos
- Ejemplos predefinidos listos para usar
- Caché de embeddings para ahorrar tokens de API
- Interfaz web limpia sin terminal

## Ejemplos Incluidos

1. **Madrid - España + Francia** = París (geografía)
2. **Rey - Hombre + Mujer** = Reina (género)
3. **Toro - Macho + Hembra** = Vaca (biología)
4. **Personalizado**: Crea tus propios ejemplos

## Requisitos

- Python 3.12+
- API Key de OpenAI
- uv (gestor de paquetes)

## Instalación

1. Clona el repositorio
2. Instala dependencias:
   ```bash
   uv sync
   ```

3. Configura tu API key de OpenAI:
   ```bash
   cp .env.example .env
   ```

4. Edita `.env` y agrega tu API key:
   ```
   OPENAI_API_KEY=tu-api-key-aqui
   ```

## Uso

Ejecuta la aplicación:

```bash
uv run streamlit run app.py
```

La interfaz web se abrirá automáticamente en tu navegador.

## Cómo Funciona

1. **Paso 1**: Obtiene embeddings (vectores) para las palabras base
2. **Paso 2**: Realiza álgebra vectorial (suma/resta)
3. **Paso 3**: Obtiene embeddings de palabras candidatas
4. **Paso 4**: Calcula similitud coseno para encontrar la palabra más cercana

## Estructura del Proyecto

```
exp-embeddings/
├── app.py              # Aplicación Streamlit principal
├── main.py             # Script CLI (opcional)
├── pyproject.toml      # Dependencias del proyecto
├── .env.example        # Plantilla para variables de entorno
└── README.md           # Este archivo
```

## Dependencias

- `openai>=1.0.0` - API de embeddings
- `numpy>=1.24.0` - Operaciones vectoriales
- `streamlit>=1.28.0` - Interfaz web
- `plotly>=5.17.0` - Gráficos interactivos
- `python-dotenv>=1.0.0` - Carga de variables de entorno

## Notas Pedagógicas

Esta demo está diseñada para mostrar a estudiantes que:

- Los embeddings capturan **relaciones**, no hechos
- El modelo no "sabe" geografía/biología/género
- Solo entiende patrones matemáticos en los datos de entrenamiento
- La relación "capital → país" es consistente matemáticamente

## Optimización de Costos

La aplicación usa caché de Streamlit (`@st.cache_data`) para evitar llamadas repetidas a la API de OpenAI, ahorrando tokens y reduciendo costos.

## Licencia

Proyecto educativo para el Diplomado en Estrategia e Inteligencia Artificial, UDD 2025.
