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

Además:
- Se mide contra una **política sintética** que nosotros inventamos. La red puede copiar perfectamente una regla que no representa bien a una mascota real.
- Con **clases desbalanceadas** sería peor. Si 95 % de los ejemplos fueran COMER, un modelo que siempre dice COMER tendría 95 % de precisión sin aprender nada. Aquí, como las clases están balanceadas, ese modelo acertaría solo 32.90 %.
- No dice nada sobre la **confianza** de cada predicción ni sobre datos fuera del rango 0 a 1.
