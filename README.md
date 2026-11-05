# Neurito: mi cerebro artificial

Mascota virtual que decide qué hacer (**comer**, **dormir** o **jugar**) con una red neuronal hecha a mano en NumPy y mostrada con pygame.

- **Red:** 3 entradas (hambre, energía, aburrimiento) → capa oculta ReLU → 3 salidas Softmax.
- **Entrenamiento:** 3000 ejemplos sintéticos y descenso de gradiente por lotes completos.
- **Configuración base:** 6 neuronas ocultas, 1200 épocas, tasa de aprendizaje 0.1.
- **Resultado base:** pérdida final 0.1225 y precisión de 98.90 %.

## Cómo ejecutar

```bash
uv run python main.py          # la mascota interactiva
uv run python experimentos.py  # los experimentos de abajo, sin interfaz
```

En [main.py](main.py) puedes cambiar `HIDDEN_SIZE`, `EPOCHS` y `LEARNING_RATE` para repetir los experimentos.

---

## Experimentos

### 1. Dos necesidades fijas, variar la tercera

Valores fijos: hambre = 0.5, energía = 0.7 (cansancio = 0.3), aburrimiento = 0.2.

| Necesidad variada | Antes del cambio | Punto de cambio | Después |
|---|---|---|---|
| Hambre | 0.00 a 0.25 → DORMIR | **0.30** → COMER (47.4 % vs 46.5 %) | COMER 85.5 % en 0.40, cerca de 100 % en 0.95 |
| Energía | 0.00 a 0.50 → DORMIR | **0.55** → COMER (71.7 %) | COMER se mantiene |
| Aburrimiento | 0.00 a 0.50 → COMER | **0.55** → JUGAR (70.3 %) | JUGAR cerca de 100 % en 0.95 |

**Conclusión:** la acción cambia justo cuando la necesidad variada supera a la mayor de las otras dos. La red aprendió la frontera de la política (argmax), y cerca de esa frontera las probabilidades quedan cerca de 50/50.

### 2. Prioridades casi iguales (Softmax)

| Entrada [hambre, energía, aburrimiento] | COMER | DORMIR | JUGAR |
|---|---|---|---|
| [0.5, 0.5, 0.5] todo empatado | 35.2 % | 33.4 % | 31.3 % |
| [0.52, 0.5, 0.5] hambre apenas mayor | **44.3 %** | 28.9 % | 26.8 % |
| [0.5, 0.48, 0.5] cansancio apenas mayor | 30.6 % | **42.3 %** | 27.1 % |
| [0.5, 0.5, 0.52] aburrimiento apenas mayor | 30.7 % | 29.0 % | **40.3 %** |
| [0.9, 0.1, 0.9] empate alto | 35.9 % | 33.2 % | 30.9 % |
| [0.9, 0.5, 0.3] hambre claramente mayor | **99.9 %** | 0.1 % | 0.0 % |

**Conclusión:** con prioridades casi iguales Softmax reparte la probabilidad casi por igual (~33 % cada una), es decir, la red está insegura. Una diferencia de solo 0.02 ya mueve unos 10 puntos hacia la acción ganadora. Cuando hay una diferencia clara, la probabilidad se concentra en una sola acción.

### 3. Tamaño de la capa oculta

1200 épocas, tasa de aprendizaje 0.1:

| Neuronas | Pérdida final | Precisión |
|---|---|---|
| 3 | 0.1618 | 96.50 % |
| 6 | 0.1225 | 98.90 % |
| 10 | 0.1262 | 98.80 % |
| 20 | 0.1220 | 99.50 % |

**Conclusión:** con 3 neuronas la red tiene poca capacidad y queda un poco peor. De 6 en adelante la mejora es pequeña: el problema es sencillo, así que más neuronas casi no ayudan y cuestan más cómputo.

### 4. Épocas y tasa de aprendizaje

Pérdida medida al inicio, a 1/4, a la mitad y al final del entrenamiento:

| Épocas | Tasa | Pérdida | Precisión | Resultado |
|---|---|---|---|---|
| 200 | 0.1 | 1.074 → 0.936 → 0.806 → 0.498 | 92.90 % | Faltan épocas: todavía estaba aprendiendo |
| 1200 | 0.1 | 1.074 → 0.339 → 0.194 → 0.122 | 98.90 % | Configuración base |
| 5000 | 0.1 | 1.074 → 0.119 → 0.079 → 0.053 | 99.60 % | Mejora, pero cada vez menos: se estanca |
| 1200 | 0.01 | 1.074 → 0.985 → 0.912 → 0.738 | 83.00 % | Tasa muy baja: aprende demasiado lento |
| 1200 | 0.5 | 1.074 → 0.107 → 0.071 → 0.048 | 99.60 % | Aprende más rápido y mejor |
| 1200 | 1.0 | 1.074 → 0.172 → 0.089 → 0.045 | 98.90 % | Todavía estable |
| 1200 | 5.0 | 1.074 → 1.098 → 1.098 → 1.098 | 34.90 % | **Inestable:** las neuronas ReLU mueren y la red adivina |
| 1200 | 20.0 | 1.074 → 9.017 → 9.017 → 6.162 | 34.90 % | **Diverge:** la pérdida sube en vez de bajar |

**Conclusión:**
- Más épocas mejoran el resultado hasta que la ganancia se vuelve muy pequeña (estancamiento).
- Una tasa muy baja aprende lento.
- Una tasa moderada (0.1 a 1.0) funciona bien.
- Una tasa muy alta (5 o más) vuelve el entrenamiento inestable o lo hace diverger.

### 5. ¿Qué requeriría una cuarta acción?

Ejemplo: la acción **BAÑARSE**, ligada a una necesidad nueva, la **suciedad**.

1. **Datos:** agregar una cuarta columna de entrada (suciedad entre 0 y 1), así `X` pasa a tener forma `(N, 4)`.
2. **Etiquetas:** extender la política a `argmax(hambre, 1 - energía, aburrimiento, suciedad)`, con etiquetas de 0 a 3.
3. **Red:** `W1` pasa a medir 4 × ocultas, `W2` ocultas × 4 y `b2` 1 × 4, para tener 4 salidas en Softmax.
4. **Clases balanceadas:** con los datos generados quedan COMER = 750, DORMIR = 773, JUGAR = 749 y BAÑARSE = 728 ejemplos.
5. **Interfaz:** una barra nueva, teclas para subir y bajar la suciedad, los textos en `TEXT` y las animaciones (`bañarse_1.png`, `bañarse_2.png`).

Aunque la acción no dependa de una necesidad nueva, igual hay que definir con claridad cuándo es la correcta y etiquetar ejemplos de esa situación.

---

## Explica tus resultados

### ¿En qué se diferencian pesos y sesgos?

Los **pesos** (`W1`, `W2`) multiplican cada entrada y dicen cuánto influye cada necesidad en cada neurona: dan la inclinación.

Los **sesgos** (`b1`, `b2`) se suman al final y no dependen de la entrada: desplazan el umbral. Permiten que una neurona se active (o que una acción gane) aunque las entradas sean cero, o que necesite más para activarse.

En una recta `y = w·x + b`, `w` es la pendiente y `b` es el punto de corte.

### ¿Por qué el cansancio es 1 − energía?

La red recibe la energía, pero la necesidad de dormir crece cuando la energía **baja**. Con `cansancio = 1 − energía` las tres necesidades quedan en la misma dirección (más alto = más urgente) y en la misma escala de 0 a 1. Así se pueden comparar directamente con `argmax`: energía 0.2 equivale a cansancio 0.8.

### ¿Qué hace ReLU y qué hace Softmax?

- **ReLU** (`max(0, x)`) deja pasar los valores positivos y convierte los negativos en 0. Agrega no linealidad. Sin ella, todas las capas juntas equivaldrían a una sola transformación lineal y la red no podría aprender fronteras como "la mayor de tres necesidades".
- **Softmax** convierte los puntajes finales de la red en probabilidades positivas que suman 100 %. La acción con mayor probabilidad es la elegida, y el porcentaje muestra cuán segura está la red, como se ve en el experimento 2.

### ¿Por qué una precisión alta puede ser engañosa?

La precisión es un promedio que esconde **dónde** falla el modelo:

| Casos de prueba | Precisión |
|---|---|
| Todos | 98.70 % |
| Casi empatados (margen < 0.05, n = 135) | **90.37 %** |
| Claros (n = 865) | 100.00 % |



---

# Neurito 2.0: aprende a saltar (respuestas de [GUIA_SALTO.md](GUIA_SALTO.md))

Red **3 → 12 → 1** (ReLU + sigmoide) que decide cuándo saltar. Se ejecuta con `uv run saltar.py`.

**Cómo se midió:** se repitió el ciclo de [saltar.py](saltar.py) sin interfaz gráfica. Se usaron los mismos datos (5000 de entrenamiento y 1500 de prueba, semilla 42) y la misma física de [salto_juego.py](salto_juego.py). Cada prueba de juego dura **9000 fotogramas (5 minutos a 30 FPS)** en modo automático.

### 1. Juego sin entrenar

| Épocas | Precisión (prueba) | Esquivados | Choques | Saltos |
|---|---|---|---|---|
| 0 | 44.93 % | 0 | 59 | 0 |

**Conclusión:** con pesos aleatorios la probabilidad de salto nunca llega a 0.5, así que Neurito **nunca salta** y choca con todos los obstáculos. La precisión del 44.93 % aparece porque responde siempre «esperar» y acierta solo los ejemplos negativos.

### 2. Entrenar de cien en cien épocas

| Épocas | Pérdida | Precisión (prueba) | Esquivados | Choques |
|---|---|---|---|---|
| 100 | 0.2601 | 91.20 % | 67 | 5 |
| 200 | 0.2306 | 91.87 % | 72 | 1 |
| 300 | 0.2066 | 92.33 % | 74 | **0** |
| 500 | 0.1368 | 94.13 % | 74 | 0 |
| 700 | 0.0858 | 96.47 % | 74 | 0 |
| 1000 | 0.0574 | 97.80 % | 74 | 0 |
| 2000 | 0.0306 | 98.80 % | 74 | 0 |

**Conclusión:** la pérdida baja todo el tiempo y la precisión sube. El juego, en cambio, deja de mejorar desde la época **300**: ya no choca, aunque la precisión siga en 92 %. Desde ese punto, los ejemplos que la red mejora son casos del borde de la ventana que no cambian el resultado en el juego. Mejor pérdida no siempre significa mejor comportamiento.

### 3. Neuronas ocultas: 3, 12 y 24

| Ocultas | Épocas | Pérdida | Precisión (entrenamiento) | Precisión (prueba) | Choques |
|---|---|---|---|---|---|
| 3 | 1000 | 0.2060 | 93.66 % | 93.20 % | 0 |
| 3 | 3000 | 0.1978 | 94.04 % | 93.67 % | 0 |
| 12 | 1000 | 0.0574 | 98.04 % | 97.80 % | 0 |
| 12 | 3000 | 0.0219 | 99.28 % | 99.13 % | 0 |
| 24 | 1000 | 0.0502 | 98.28 % | 97.73 % | 0 |
| 24 | 3000 | 0.0207 | 99.40 % | 99.27 % | 0 |

**Conclusión:**
- Con **3 neuronas** la red se estanca cerca del 94 %. Le falta capacidad para formar una ventana con dos bordes («ni muy lejos ni muy cerca»).
- **12 y 24** quedan casi iguales. Duplicar las neuronas apenas mejora.
- La **generalización es buena en los tres casos**: entre entrenamiento y prueba hay menos de 0.5 puntos de diferencia, así que no hay sobreajuste. Los datos son sintéticos y salen de la misma regla, por eso la prueba se parece mucho al entrenamiento.
- Los tres tamaños esquivan todo en el juego. La ventana física es más ancha que la ventana de las etiquetas, así que no hace falta clasificar perfecto.

### 4. Tasa de aprendizaje: 0.01, 0.8 y 1.2

Pérdida inicial en los tres casos: 0.7099.

| Tasa | Épocas | Pérdida | Precisión (prueba) | Esquivados | Choques |
|---|---|---|---|---|---|
| 0.01 | 100 | 0.6449 | 90.73 % | 58 | 12 |
| 0.01 | 1000 | 0.4302 | 86.53 % | **0** | **56** |
| 0.01 | 3000 | 0.3111 | 88.60 % | **0** | **55** |
| 0.8 | 100 | 0.2601 | 91.20 % | 67 | 5 |
| 0.8 | 1000 | 0.0574 | 97.80 % | 74 | 0 |
| 1.2 | 100 | 0.2560 | 89.27 % | 57 | 13 |
| 1.2 | 300 | 0.1689 | 91.40 % | 71 | 2 |
| 1.2 | 1000 | 0.0392 | 98.27 % | 74 | 0 |
| 1.2 | 3000 | 0.0157 | 99.60 % | 74 | 0 |

**Conclusión:**
- **0.01 converge muy lento.** Después de 3000 épocas la pérdida (0.31) sigue peor que la de 0.8 en 100 épocas. Lo más importante: con **88.6 % de precisión choca en todos los obstáculos**. La red todavía marca «saltar» con tiempos de 18 a 23 fotogramas. Todos sus errores son falsos positivos (510 en entrenamiento, 0 falsos negativos), así que **salta demasiado pronto** y aterriza antes de pasar el obstáculo. Además, después de cada choque la pista vuelve a velocidad 5.5, así que repite el mismo error siempre.
- **0.8 (la base)** converge rápido y de forma estable.
- **1.2** empieza un poco más irregular (más choques en las primeras 300 épocas). Después converge **más rápido** que 0.8 y llega a una pérdida menor. Con este problema 1.2 todavía no diverge; una tasa mucho mayor sí podría volver inestable el entrenamiento.

### Ventana física frente a ventana de entrenamiento

Para entender los resultados se simuló un salto en cada tiempo estimado y velocidad posible:

| Velocidad (px/fotograma) | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 |
|---|---|---|---|---|---|
| Saltos que esquivan (tiempo en fotogramas) | 4.5 a 15 | 4.5 a 18 | 5 a 20.5 | 4.5 a 22 | 5 a 23.5 |

Las etiquetas piden saltar entre **7 y 13**, una ventana que cabe dentro de la física en todas las velocidades. Por eso la red base no choca aunque no clasifique perfecto. Con 0.01, en cambio, la red salta en 18 a 23 fotogramas, por fuera del límite físico. En resumen, **la precisión cuenta errores, pero no dice si son errores graves para el juego**.

---

# Neurito 3.0: aprende a encontrar comida en un laberinto

Las respuestas a los experimentos de [GUIA_LABERINTO.md](GUIA_LABERINTO.md) están en [RESPUESTAS_LABERINTO.md](RESPUESTAS_LABERINTO.md).
