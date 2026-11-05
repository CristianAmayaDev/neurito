# Neurito 3.0: respuestas de la guía del laberinto ([GUIA_LABERINTO.md](GUIA_LABERINTO.md))

Red **165 → 96 → 4** (ReLU + Softmax) que imita a BFS para elegir ARRIBA, ABAJO, IZQUIERDA o DERECHA. Se ejecuta con `uv run laberinto.py`. Las pruebas pasan con `uv run python -m unittest -v test_laberinto.py` (5 de 5).

**Cómo se midió:** se repitió el ciclo de [laberinto.py](laberinto.py) sin interfaz gráfica, con las mismas funciones de [laberinto_mundo.py](laberinto_mundo.py) y [laberinto_red.py](laberinto_red.py).

- **Datos:** 3576 ejemplos (los 3 mapas tienen 33, 36 y 36 casillas libres). Se usan 2860 para entrenar y 716 para probar, con semilla 42. Ningún par mapa-posición-meta de prueba aparece en entrenamiento.
- **Juego:** cada uno de los 716 retos de prueba se jugó como en la ventana: **45 intentos**, un choque no mueve a Neurito y gasta un intento.
- **Éxito:** porcentaje de retos en que llega a la comida antes de 45 intentos.
- **Ruta/óptima:** pasos usados entre pasos mínimos de BFS, solo en los retos con éxito (1.00 = camino más corto).
- **Atasco en pared:** falló y chocó al menos una vez. **Atasco en ciclo:** falló sin chocar, dando vueltas entre casillas.

### 1. Neurito sin entrenar: ¿por qué se atasca?

| Épocas | Exactitud (prueba) | Probabilidad máxima media | Éxito red pura | Choques por reto | Atasco en pared | Atasco en ciclo |
|---|---|---|---|---|---|---|
| 0 | 27.7 % | 29.7 % | 3.1 % | 36.4 de 45 | 82.3 % | 14.7 % |

**Conclusión:** con pesos aleatorios la red adivina: 27.7 % es casi el 25 % del azar con 4 salidas, y Softmax reparte casi igual (la salida más alta vale en promedio 29.7 %). Además, la red elige ARRIBA o ABAJO en el 87 % de los casos.

Se atasca por una razón concreta: **la red es determinista y no tiene memoria**. Si elige una dirección con pared, Neurito no se mueve. Entonces la entrada (mapa, posición, meta) es exactamente la misma, la salida también, y vuelve a chocar contra la misma pared hasta gastar los 45 intentos. Por eso hace 36 choques por reto. En el 14.7 % restante no choca, pero va y vuelve entre dos casillas para siempre. Durante el juego la red no aprende nada; solo aprende cuando se oprimen 1, 2 o 3.

### 2. Entrenar 100, 200 y 400 épocas

96 neuronas ocultas, tasa 0.45, lotes de 256:

| Épocas | Pérdida | Exactitud (entrenamiento) | Exactitud (prueba) | Éxito red pura | Choques por reto | Atasco en ciclo | Éxito asistido | Ruta/óptima (asistido) |
|---|---|---|---|---|---|---|---|---|
| 0 | — | 25.3 % | 27.7 % | 3.1 % | 36.37 | 14.7 % | 87.8 % | 2.93 |
| 10 | 0.6822 | 76.2 % | 73.0 % | 23.9 % | 14.19 | 43.0 % | 98.5 % | 1.86 |
| 50 | 0.2756 | 89.2 % | 82.3 % | 36.3 % | 3.47 | 55.6 % | 98.3 % | 1.49 |
| **100** | 0.1352 | 94.1 % | 84.9 % | 51.3 % | 0.91 | 46.6 % | 100 % | 1.35 |
| **200** | 0.0376 | 99.9 % | 90.1 % | 81.3 % | 0.31 | 18.0 % | 100 % | 1.12 |
| **400** | 0.0107 | 100 % | 91.5 % | 82.7 % | 0.19 | 16.9 % | 100 % | 1.09 |

**Conclusión:**
- **Pérdida:** baja todo el tiempo (1.37 → 0.135 → 0.038 → 0.011).
- **Precisión:** la de entrenamiento llega a 100 % en 200 épocas, pero la de prueba se queda cerca de **91 %**. La red se memoriza los ejemplos que vio, y los ~9 puntos de diferencia son lo que no logra generalizar a pares posición-meta nuevos.
- **Movimientos:** el gran salto ocurre entre 100 y 200 épocas (51 % → 81 % de éxito en red pura). De 200 a 400 la pérdida baja 3.5 veces, pero el éxito solo sube 1.4 puntos. Igual que en Neurito 2.0, **menos pérdida no siempre es mejor comportamiento**.
- Cuando la red pura llega, lo hace por el **camino más corto** (ruta/óptima ≈ 1.01), porque copia a BFS. Al entrenar, los choques casi desaparecen y los fracasos pasan a ser **ciclos**: dos casillas que se señalan una a la otra.
- **Precisión ≠ laberintos completados:** 91.5 % de decisiones correctas da solo 82.7 % de retos completados. Un reto necesita varias decisiones seguidas y basta una equivocada para salirse de la ruta o entrar en un ciclo.

### 3. Red pura frente a navegación asistida

| Épocas | Modo | Éxito | Choques por reto | Ruta/óptima |
|---|---|---|---|---|
| 0 | Red pura | 3.1 % | 36.37 | 1.00 |
| 0 | Asistido | **87.8 %** | 0 | 2.93 |
| 400 | Red pura | 82.7 % | 0.19 | 1.01 |
| 400 | Asistido | **100 %** | 0 | 1.09 |

¿Qué hace cada parte?

| Parte | Quién la aporta | Dónde está |
|---|---|---|
| Probabilidad de cada dirección según mapa, posición y meta | **El modelo** (aprendido de BFS) | `red.probabilidades` |
| Nunca elegir una dirección con pared | **Regla** (no se aprendió) | `valido(...)` en `elegir_accion` |
| Restar 0.35 por cada visita previa a la casilla vecina | **Regla** (memoria externa) | `visitas` en `elegir_accion` |

**Conclusión:**
- El dato más revelador es que **asistido sin entrenar ya consigue 87.8 %**. Los laberintos son pequeños, y "no chocar + preferir casillas nuevas" basta para explorarlos casi siempre, aunque con rutas casi 3 veces más largas que la óptima. Ese 87.8 % es mérito de las reglas, no de la red.
- La red aporta la **dirección correcta**: con 400 épocas la ruta asistida baja de 2.93 a 1.09 veces la óptima y el éxito sube a 100 %.
- En **red pura todo es mérito (o culpa) del modelo**: 82.7 %. La regla de paredes elimina los choques, y la de visitas rompe los ciclos que la red sola no puede romper porque no recuerda dónde estuvo.
- Para evaluar qué tanto aprendió Neurito hay que mirar la **red pura**. El modo asistido mide red + reglas.

### 4. Neuronas ocultas: 16, 32, 96 y 128

400 épocas, tasa 0.45. Promedio de 4 semillas (1, 2, 3 y 42); entre paréntesis, el peor y el mejor caso.

| Ocultas | Exactitud (prueba) | Éxito red pura | Éxito asistido |
|---|---|---|---|
| 16 | 88.4 % (87.2 a 89.5) | 77.5 % (74.3 a 79.5) | 100 % |
| 32 | 90.2 % (89.2 a 91.3) | 82.6 % (78.4 a 84.9) | 100 % |
| 96 | 91.0 % (90.2 a 91.5) | 83.1 % (82.4 a 84.1) | 100 % |
| 128 | 90.8 % (90.4 a 91.3) | 83.6 % (82.1 a 85.3) | 100 % |

**Conclusión:**
- Las cuatro redes llegan a 99.4–100 % en entrenamiento: todas tienen capacidad para memorizar los 2860 ejemplos.
- **16 neuronas** generaliza peor: unos 3 puntos menos en prueba y 6 menos en red pura.
- Desde **32** la mejora es pequeña, y entre 96 y 128 la diferencia es menor que la variación entre semillas. Lo que limita a la red no es el tamaño, sino lo que recibe: posición y meta como casillas sueltas, sin ver las paredes (ver «Alcance y limitaciones» en la guía).
- Con 96 la red es la más estable entre semillas, y por eso es un buen valor por defecto.

### 5. Tasa de aprendizaje: ¿converge con 0.45?

96 neuronas ocultas, 400 épocas. "Subidas" cuenta cuántas épocas terminaron con más pérdida que la anterior.

| Tasa | Pérdida ép. 1 | Ép. 10 | Ép. 100 | Ép. 400 | Subidas | Exactitud (prueba) | Éxito red pura | Resultado |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 1.394 | 1.379 | 1.247 | 0.682 | 0 | 73.9 % | 20.4 % | Converge, pero muy lento |
| 0.1 | 1.388 | 1.258 | 0.444 | 0.149 | 22 | 87.2 % | 60.2 % | Le faltan épocas |
| **0.45** | 1.370 | 0.682 | 0.135 | **0.0107** | 85 | 91.5 % | 82.7 % | **Converge** (base) |
| 1.0 | 1.342 | 0.551 | 0.027 | 0.0028 | 75 | 93.3 % | 85.5 % | Converge más rápido |
| 2.0 | 1.312 | 0.482 | 0.007 | 0.0011 | 20 | 93.3 % | 85.6 % | Converge más rápido |
| 3.0 | — | — | — | 0.0007 | — | 93.9 % | — | Todavía converge |
| 5.0 | 1.515 | 1.446 | 1.177 | 1.187 | 193 | 30.3 % | 4.5 % | **No converge** |

**Conclusión:**
- **Sí, 0.45 converge.** La pérdida baja de 1.37 a 0.011. Hay 85 pequeñas subidas, normales en descenso con mini-lotes porque cada lote da un gradiente con ruido, pero la tendencia siempre es hacia abajo.
- **0.01** apenas se mueve: en 400 épocas queda como la base en 10 épocas.
- Entre **1.0 y 3.0** converge todavía más rápido y gana ~2 puntos en prueba. En este problema 0.45 es una opción prudente, no la más rápida.
- Con **5.0** el entrenamiento se rompe: la pérdida sube casi la mitad de las veces (193 de 399), llega a 1.93 (peor que al inicio) y la red vuelve a jugar como sin entrenar (4.5 % de éxito).

### 6. Cambiar de mapa

Red base (96 ocultas, 400 épocas). Los tres mapas se entrenan juntos, pero los retos de prueba son pares posición-meta nuevos.

| Mapa | Ejemplos (entr. / prueba) | Exactitud (entrenamiento) | Exactitud (prueba) | Éxito red pura | Choques por reto | Éxito asistido | Ruta/óptima (asistido) |
|---|---|---|---|---|---|---|---|
| 1 | 846 / 210 | 100 % | 92.4 % | 85.7 % | 0.00 | 100 % | 1.07 |
| 2 | 1009 / 251 | 100 % | 90.0 % | 78.5 % | 0.36 | 100 % | 1.13 |
| 3 | 1005 / 255 | 100 % | 92.2 % | 84.3 % | 0.18 | 100 % | 1.08 |

**Conclusión:**
- El cambio de mapa no confunde a la red, porque **conoce el número de mapa** (una de sus 165 entradas). Los tres quedan en 100 % de entrenamiento y entre 90 % y 92 % de prueba.
- El **mapa 2** es el más difícil. Tiene más pasillos que forman vueltas, así que muchas casillas tienen dos caminos igual de cortos y BFS desempata con un orden fijo (ARRIBA, ABAJO, IZQUIERDA, DERECHA). Esa regla de desempate es más difícil de aprender que un pasillo con una sola salida.
- Esto demuestra que la red **generaliza a pares posición-meta nuevos dentro de mapas conocidos**. **No** demuestra que resuelva un mapa nuevo: con un cuarto mapa la red no sabría dónde están sus paredes, porque nunca las recibe como entrada.

### 7. DFS, BFS y A* como profesor

Se cambió solo la función que genera las etiquetas. Los pares de entrenamiento y prueba, la red (96 ocultas, tasa 0.45, 400 épocas) y la forma de jugar se dejaron iguales. Los archivos del proyecto no se modificaron.

- **BFS:** el profesor actual. Recorre desde la comida por capas y da siempre el camino más corto.
- **DFS:** recorre desde la comida yendo lo más profundo posible. Cada casilla apunta hacia la casilla desde la que DFS llegó. Siempre lleva a la comida, pero no por el camino más corto.
- **A\*:** desde cada posición busca la comida con heurística Manhattan (distancia en casillas ignorando paredes). La etiqueta es el primer paso del camino.

| Profesor | Ruta del profesor / óptima | Etiquetas iguales a BFS | Pérdida | Exactitud (prueba) | Éxito red pura | Ruta/óptima (red pura) | Éxito asistido (45 intentos) |
|---|---|---|---|---|---|---|---|
| DFS | 2.24 | 49.2 % | 0.0145 | 90.5 % | **69.3 %** | **2.16** | 90.9 % (99.9 % con 120) |
| BFS | 1.00 | 100 % | 0.0107 | 91.5 % | 82.7 % | 1.01 | 100 % |
| A* | 1.00 | 96.4 % | 0.0110 | 92.6 % | 86.3 % | 1.00 | 100 % |

**Conclusión:**
- **DFS empeora a Neurito**, aunque la exactitud casi no cambia (90.5 % frente a 91.5 %). La red imita a DFS igual de bien, pero DFS enseña rutas 2.24 veces más largas. Neurito las copia (2.16 veces la óptima) y fracasa más: 69.3 % en red pura. En modo asistido, 9 % de los retos no alcanzan a terminar en 45 intentos; con 120 intentos llega en 99.9 %.
- **A\* da casi lo mismo que BFS.** Con heurística Manhattan también encuentra caminos óptimos. Solo cambia el 3.6 % de las etiquetas: casillas con dos caminos igual de cortos, donde cada algoritmo desempata distinto. La pequeña mejora (86.3 % frente a 82.7 % en red pura) sale de una sola semilla y se debe a ese desempate, no a rutas más cortas.
- **Lección principal:** en aprendizaje por imitación **el alumno no supera al profesor**. La exactitud mide qué tan bien la red copia al profesor, no si el profesor es bueno. Un profesor malo produce un alumno igual de "exacto" pero peor en el juego.
- A* es más eficiente que BFS cuando se busca un solo camino en mapas grandes. Aquí, con mapas de 9×9 y una tabla completa de etiquetas, BFS desde la meta es lo más simple, porque en un solo recorrido calcula la respuesta para todas las posiciones.

### Resumen

- La red aprende una **tabla de decisiones** que imita a BFS. No razona sobre paredes ni recuerda por dónde pasó.
- **Exactitud de prueba ≈ 91 %**, pero eso equivale a **≈ 83 % de laberintos** completados en red pura, porque los errores se acumulan en una ruta.
- Las **reglas del modo asistido** (paredes y visitas) cubren lo que la red no puede hacer sola. Por sí solas ya resuelven estos mapas pequeños, aunque con rutas largas.
