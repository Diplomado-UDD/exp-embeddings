2.5: Demo de Código: Viendo los Vectores (20 min)
Diapositiva 41: Demo: demo_clase_2_embeddings.py
Objetivo: Demostrar el "Álgebra de Palabras" (Pilar 2) en vivo.
Diapositiva 42: El Código - Parte 1: Obtener Vectores.
[AQUÍ MUESTRAS TU CÓDIGO: get_embedding]
Explicación (para mánagers): "Vamos a pedirle a la API de OpenAI que nos dé la 'coordenada GPS' (el vector) para tres palabras: 'Toro', 'Macho' y 'Hembra'".
Diapositiva 43: El Código - Parte 2: El "Álgebra".
[AQUÍ MUESTRAS TU CÓDIGO: v_resultado = v_toro - v_macho + v_hembra]
Explicación: "Estamos restando y sumando las coordenadas. El resultado es una nueva 'coordenada GPS' que teóricamente debería aterrizar cerca del concepto 'Vaca'".
Diapositiva 44: El Código - Parte 3: La Búsqueda.
[AQUÍ MUESTRAS TU CÓDIGO: find_most_similar(v_resultado, [lista_de_candidatos...])]
Explicación: "Ahora le preguntamos a un conjunto de vectores candidatos (Vaca, Campo, Leche, etc.) cuál está más cerca (matemáticamente) de esta nueva coordenada que calculamos".
Diapositiva 45: El Resultado en Vivo.
[AQUÍ MUESTRAS LA SALIDA DE TU SCRIPT]
El resultado más cercano es: Vaca
Conclusión: El modelo no "sabe" qué es una vaca. Solo sabe que la relación Toro -> Vaca es la misma que Macho -> Hembra. Entiende de relaciones, no de hechos.
