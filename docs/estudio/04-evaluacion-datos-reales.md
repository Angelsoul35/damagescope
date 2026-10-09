# 4. Evaluación en fotos reales: sobreajuste, validación cruzada y ROC-AUC

Los capítulos 1 y 2 usan paneles **sintéticos**: rápidos y controlados, ideales para probar
el flujo completo. Este capítulo da el siguiente paso: medir **el mismo detector del
proyecto** (features HOG + random forest, sin cambiarlo) sobre **fotos reales de carros**, y
responder una pregunta de fondo: *¿el modelo de verdad generaliza, o solo memorizó?*

El código está en `damagescope/real_data.py` (cargar/descargar) y
`damagescope/real_benchmark.py` (medir). Corre el benchmark con:

```bash
python -m damagescope.real_benchmark
```

---

## 4.1 Los datos reales

* **Fuente:** `ikuldeep1/vehicle-damage-fraud-image-balanced` (Hugging Face Hub): 12.729
  fotos reales de carros, balanceadas. La etiqueta es `0 = sin daño`, `1 = con daño`
  (verificado a ojo sobre muestras de cada clase antes de construir el cargador).
* **Subconjunto vendorizado:** 200 imágenes por clase (400 en total, ~2,6 MB) bajo
  `datasets/`, para que el benchmark corra **offline y reproducible**. Las imágenes se toman
  de offsets repartidos por todo el rango de cada clase, así varían fondos, ángulos, colores
  y tipos de daño; no son un bloque contiguo.
* **Tamaño:** cada imagen se reduce a 128 px (el tamaño al que el extractor HOG del proyecto
  reescala de todos modos), así no se pierde nada de lo que el detector ve.

> **Dato clave: este dato NO es sintético.** Un panel sintético es limpio y predecible; una
> foto real trae fondo, iluminación, suciedad, reflejos y choques de muchos tipos. Por eso
> aquí **no** esperamos un 1.0 perfecto: un resultado fuerte y *bien ajustado* por encima de
> la línea base del azar (0.5) es exactamente lo correcto.

---

## 4.2 Las *features*: qué se mide en la imagen y por qué

El detector no mira los píxeles crudos; primero describe la imagen con **HOG** (Histogram of
Oriented Gradients; ver capítulo 1). La tubería (*pipeline*) es exactamente la del proyecto:

1. **A gris y a 128×128** (`to_gray_float`): el daño se nota por su *forma* (bordes), no por
   el color, así que trabajar en gris reduce ruido y dimensiones.
2. **HOG** (`extract_features`): 8 orientaciones, celdas de 16×16, bloques de 2×2 →
   **1.568 dimensiones** por imagen. HOG cuenta hacia dónde apuntan los bordes en cada celda;
   un choque, un rayón o una abolladura crean bordes fuertes y desordenados que un panel sano
   no tiene.
3. **`StandardScaler`**: cada dimensión se centra y se escala (media 0, desviación 1), para
   que ninguna domine por su escala.
4. **`RandomForestClassifier`** (200 árboles): muchos árboles de decisión que votan; robusto,
   da probabilidades y no necesita GPU.

> **Por qué HOG y no los píxeles.** 1.568 números que resumen la *estructura de bordes*
> separan mejor dañado/sano, con pocos datos, que 128×128 = 16.384 píxeles crudos llenos de
> variación irrelevante (color de la pintura, fondo). Menos dimensiones con más señal =
> menos riesgo de sobreajuste.

---

## 4.3 Sobreajuste (*overfitting*) vs. subajuste (*underfitting*)

Son las dos formas de fallar al generalizar:

* **Subajuste (*underfitting*):** el modelo es demasiado simple o la señal es muy débil. Le
  va **mal hasta en los datos de entrenamiento**. Síntoma: *accuracy* y ROC-AUC bajos en
  **todo** (train, CV, test).
* **Sobreajuste (*overfitting*):** el modelo **memoriza** el entrenamiento (incluido su
  ruido) y no generaliza. Síntoma: puntajes **altos en train** pero **bajos en datos no
  vistos** (CV / test). Lo delata la **brecha** (*gap*):

$$\text{brecha} = \text{métrica}_{\text{train}} - \text{métrica}_{\text{validación}}$$

Una brecha grande = sobreajuste; una brecha pequeña con puntajes altos = buen ajuste.

> **Trampa del Random Forest (muy importante).** Un random forest alcanza casi **1.0 de
> ROC-AUC en SU PROPIO entrenamiento por construcción**: los árboles crecen hasta separar
> cada ejemplo. Eso **no** es sobreajuste por sí solo, y **no** se arregla "capando" el
> modelo. La forma correcta de juzgarlo es mirar si **CV y held-out concuerdan** y están muy
> por encima del azar. Por eso el veredicto de `real_benchmark.py` **ignora** la brecha
> train→CV como señal de sobreajuste en un random forest, y decide por el **acuerdo
> CV / held-out**.

El veredicto (`classify_fit`) resume esto:

| Veredicto | Condición |
|---|---|
| **well-fit** | CV y test ≥ 0.65, de acuerdo entre sí (\|CV−test\| ≤ 0.15) y folds estables (std ≤ 0.12) |
| **overfit** | train alto (≥ 0.65) pero CV bajo (< 0.65): memoriza, no generaliza |
| **underfit** | train bajo (< 0.65): ni siquiera ajusta |

---

## 4.4 El método: validación cruzada + conjunto de prueba reservado

Medir en los mismos datos con los que entrenas infla el resultado. Por eso separamos:

1. **Held-out (reservado):** `train_test_split` estratificado aparta un **25 %** (100 de 400
   imágenes) que el modelo **nunca** ve durante la selección. Es la prueba final en datos
   frescos. *Estratificado* = mantiene la proporción 50/50 de clases en cada lado.
2. **Validación cruzada de 5 particiones (*5-fold stratified CV*)** sobre el 75 % restante
   (300 imágenes): se parte en 5 pliegues; se entrena en 4 y se evalúa en el que sobra; se
   rota 5 veces. Cada imagen se evalúa **una vez como no vista**. Esto da una estimación
   estable y, con la **desviación entre pliegues**, una idea de cuánto varía.

> **Por qué AMBOS.** La CV estima bien y usa todos los datos de entrenamiento; el held-out
> es un árbitro independiente que confirma la CV sobre datos que no tocaron ni la CV ni la
> selección. Si **concuerdan**, el resultado es creíble.

---

## 4.5 ROC-AUC vs. *accuracy* (con fórmulas)

De la matriz de confusión (VP = verdaderos positivos, FP = falsos positivos, VN = verdaderos
negativos, FN = falsos negativos):

$$\text{accuracy} = \frac{VP + VN}{VP + VN + FP + FN}
\qquad
\text{precision} = \frac{VP}{VP + FP}$$

$$\text{recall (TPR)} = \frac{VP}{VP + FN}
\qquad
\text{FPR} = \frac{FP}{FP + VN}
\qquad
F_1 = 2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision} + \text{recall}}$$

* **Accuracy** depende de un **umbral** fijo (aquí 0.5) y, con clases balanceadas, es fácil
  de leer — pero oculta *cómo* de seguras son las decisiones.
* **ROC-AUC** es el área bajo la curva ROC (TPR contra FPR al barrer **todos** los umbrales).
  Tiene una interpretación probabilística preciosa:

$$\text{ROC-AUC} = P\big(\,\text{score}(\text{positivo aleatorio}) > \text{score}(\text{negativo aleatorio})\,\big)$$

Es decir: la probabilidad de que el modelo le dé **mayor puntaje a una foto con daño que a
una sin daño**, tomadas al azar. **0.5 = azar**, **1.0 = perfecto**. No depende del umbral y
no se deja engañar por el desbalance de clases. Por eso es la métrica principal del
benchmark, y la misma con la que se compararía un modelo profundo futuro.

---

## 4.6 Ejemplo trabajado con los números reales

Salida real de `python -m damagescope.real_benchmark` sobre las 400 fotos:

```
cross-val    : accuracy 0.717  ROC-AUC 0.770  F1 0.708
  per-fold AUC: [0.863, 0.733, 0.813, 0.719, 0.706]  std 0.061
held-out test: accuracy 0.800  ROC-AUC 0.867  F1 0.792
training set : accuracy 1.000  ROC-AUC 1.000
gaps         : train-vs-CV AUC +0.230  CV-vs-test AUC 0.097
baseline     : chance ROC-AUC 0.50  (shuffled-label CV AUC 0.461)
verdict      : WELL-FIT
```

**Matriz de confusión real del held-out** (100 imágenes, 50 con daño y 50 sin daño, umbral
0.5):

|  | Predicho: daño | Predicho: sin daño |
|---|---|---|
| **Real: daño** | VP = 38 | FN = 12 |
| **Real: sin daño** | FP = 8 | VN = 42 |

Reemplazando en las fórmulas:

$$\text{accuracy} = \frac{38 + 42}{100} = 0.80
\qquad
\text{precision} = \frac{38}{38 + 8} = 0.826$$

$$\text{recall} = \frac{38}{38 + 12} = 0.76
\qquad
F_1 = 2\cdot\frac{0.826 \cdot 0.76}{0.826 + 0.76} = 0.792$$

**Cómo leerlo:**

* **Muy por encima del azar.** CV (0.770) y held-out (0.867) le sacan mucho a la línea base
  0.50. En fotos reales variadas, es un resultado fuerte.
* **Bien ajustado, no memorizado.** Train AUC = 1.000 es la "trampa del random forest" de
  §4.3; lo que importa es que CV y held-out **concuerdan** (\|0.770 − 0.867\| = 0.097, dentro
  de la tolerancia 0.15) y que los pliegues son estables (std 0.061). Veredicto: **well-fit**.
* **La brecha train→CV (+0.230) es esperada**, no un defecto: es cómo funciona un bosque.

---

## 4.7 La prueba de la etiqueta barajada (*shuffled-label sanity check*)

¿Y si el 0.77 fuera un espejismo por una **fuga de datos** (*leakage*) o por azar? Hay una
prueba rigurosa y contundente: **barajar las etiquetas** (romper la relación imagen↔etiqueta)
y repetir la validación cruzada. Si la señal fuera falsa, el puntaje seguiría alto; si es
real, debe **desplomarse al azar**.

Resultado real: con etiquetas barajadas, la **ROC-AUC de CV cae a 0.461 ≈ 0.50**. Confirmado:
el detector aprende el **contenido real del daño**, no un atajo. El test
`test_shuffling_labels_collapses_auc` fija esto por escrito (exige que la AUC barajada quede
por debajo de 0.60 y bien por debajo de la AUC real).

> **Por qué es tan fuerte.** Es una verificación *constructiva*: no afirma "no hay leakage",
> lo **demuestra** rompiendo la señal y mostrando que el desempeño muere con ella.

---

## 4.8 Qué sigue (extender, no reemplazar)

Estos números son la **base verificable y sin fuga de datos** del proyecto sobre fotos
reales. Un modelo profundo (una CNN de detección o segmentación) entrenado con muchas fotos
etiquetadas **extiende** esto: sube la *accuracy* en fotos muy ruidosas, agrega segmentación
por píxel y un costo basado en datos. Y como todo se mide con la **misma ROC-AUC**, ese
modelo nuevo se compara de frente contra este punto de partida riguroso.

## Reto para ti

* Sube `per_class` en `fetch_damage` a 500 por clase y mira cómo cambian la ROC-AUC y la
  brecha train→CV. ¿La CV se vuelve más estable (menor std entre pliegues)?
* Cambia HOG por *features* de color (histogramas RGB) y compara la ROC-AUC. ¿La forma
  (bordes) o el color separan mejor el daño?
* Añade una curva ROC real (TPR vs. FPR) al reporte y ubica el umbral 0.5 sobre ella.
