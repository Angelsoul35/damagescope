# 3. Preguntas de entrevista

## Nivel junior / media

**¿Qué hace DamageScope en una frase?**
Detecta daños en fotos de paneles de vehículos, los ubica con cajas, estima severidad y costo,
y exporta el modelo a ONNX para correr en CPU.

**¿Qué es HOG y por qué lo usas?**
Histogram of Oriented Gradients: describe la imagen por las direcciones de sus bordes. Sirve
porque un rayón o abollón agrega bordes fuertes que un panel liso no tiene, así que separa bien
dañado de limpio.

**¿Por qué Random Forest y no una red neuronal?**
Porque quiero algo rápido, explicable y reproducible que corra sin GPU. Es una base rigurosa,
sin fuga de datos (*leakage-free*) y medible —validada además sobre fotos reales de carros—;
una red profunda es el siguiente paso que la **extiende**.

## Nivel media / senior

**¿Cómo ubicas el daño sin un detector entrenado?**
Con el residual: un panel limpio es casi igual a su versión desenfocada; el daño es de alta
frecuencia y resalta. Umbralizo el residual, agrupo en regiones conectadas y saco una caja por
región. Mido con IoU contra las cajas reales.

**¿Qué es IoU y por qué importa?**
Intersección sobre unión de dos cajas; mide qué tan bien coincide la caja predicha con la real.
Es la métrica estándar de localización/detección.

**¿Cómo evitas que un reflejo se confunda con daño?**
Los reflejos son suaves, así que el residual los ignora. El generador los agrega a propósito
para que la evaluación sea justa.

**¿Para qué exportas a ONNX?**
Para servir liviano en CPU sin cargar scikit-learn, y para poder correr el modelo en otros
runtimes. Una prueba verifica que la salida ONNX coincide con scikit-learn.

## Nivel senior / diseño

**¿Qué le falta para producción?**
Un modelo profundo (CNN de detección o segmentación) entrenado con fotos reales etiquetadas;
manejo de fondo, iluminación y suciedad; más tipos de daño; y un estimador de costo real. La
arquitectura (detectar, ubicar, medir, servir, exportar) ya está lista para eso.

**¿Cómo sabrías que un modelo nuevo es mejor?**
Con las mismas métricas: ROC-AUC para detección e IoU para localización, sobre el mismo
conjunto. La línea base clásica da el número a superar.

## Reto para ti

* Cambia el Random Forest por un CNN pequeño y compara ROC-AUC e IoU.
* Agrega segmentación (máscara por píxel) y mide IoU de máscara, no solo de caja.
* Entrena con imágenes con fondo y evalúa cuánto cae la localización por residual.
