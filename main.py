import numpy as np
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()


def get_embedding(text: str, client: OpenAI, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Obtiene el vector embedding de un texto usando la API de OpenAI.
    Es como obtener las 'coordenadas GPS' de una palabra en el espacio semántico.
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula la similitud coseno entre dos vectores.
    Retorna un valor entre -1 y 1, donde 1 significa vectores idénticos.
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
    Retorna una lista ordenada de (palabra, similitud).
    """
    similarities = []

    for word, vector in candidates.items():
        sim = cosine_similarity(target_vector, vector)
        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


def main():
    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN: ÁLGEBRA DE PALABRAS CON EMBEDDINGS")
    print("=" * 70)
    print("\nEjemplo: Toro - Macho + Hembra = ?")
    print("(Esperamos que el resultado sea cercano a 'Vaca')\n")

    # Inicializar cliente de OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No se encontró OPENAI_API_KEY en las variables de entorno")
        print("Por favor, ejecuta: export OPENAI_API_KEY='tu-api-key'")
        return

    client = OpenAI(api_key=api_key)

    print("Paso 1: Obteniendo embeddings de las palabras base...")
    print("-" * 70)

    # Obtener embeddings de las palabras base
    v_toro = get_embedding("Toro", client)
    v_macho = get_embedding("Macho", client)
    v_hembra = get_embedding("Hembra", client)

    print(f"✓ 'Toro'   -> Vector de {len(v_toro)} dimensiones")
    print(f"✓ 'Macho'  -> Vector de {len(v_macho)} dimensiones")
    print(f"✓ 'Hembra' -> Vector de {len(v_hembra)} dimensiones")

    print("\nPaso 2: Realizando el álgebra de vectores...")
    print("-" * 70)
    print("Operación: v_resultado = v_toro - v_macho + v_hembra")

    # Realizar el álgebra de vectores
    v_resultado = v_toro - v_macho + v_hembra

    print(f"✓ Resultado -> Vector de {len(v_resultado)} dimensiones")

    print("\nPaso 3: Obteniendo embeddings de palabras candidatas...")
    print("-" * 70)

    # Palabras candidatas
    palabras_candidatas = ["Vaca", "Campo", "Leche", "Granja", "Animal", "Hierba", "Caballo"]

    candidatos = {}
    for palabra in palabras_candidatas:
        candidatos[palabra] = get_embedding(palabra, client)
        print(f"✓ '{palabra}'")

    print("\nPaso 4: Buscando la palabra más similar al resultado...")
    print("-" * 70)

    # Encontrar la palabra más similar
    resultados = find_most_similar(v_resultado, candidatos)

    print("\nRESULTADOS (ordenados por similitud):")
    print("=" * 70)

    for i, (palabra, similitud) in enumerate(resultados, 1):
        barra = "█" * int(similitud * 50)
        print(f"{i}. {palabra:12} | {similitud:.4f} | {barra}")

    print("\n" + "=" * 70)
    print(f"🎯 RESULTADO MÁS CERCANO: {resultados[0][0]}")
    print("=" * 70)

    print("\n📝 CONCLUSIÓN:")
    print("-" * 70)
    print("El modelo NO 'sabe' qué es una vaca.")
    print("Solo entiende que la relación Toro → Vaca es similar a Macho → Hembra.")
    print("Los embeddings capturan RELACIONES, no hechos.\n")


if __name__ == "__main__":
    main()
