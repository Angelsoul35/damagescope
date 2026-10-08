# 1. Visión por computador: detección

## El problema

Dada la foto de un panel de carro, ¿hay daño o no? Es un problema de **clasificación de
imágenes**. Lo resolvemos con **visión clásica**: extraer características de la imagen y
entrenar un clasificador, sin red neuronal profunda.

## HOG (Histogram of Oriented Gradients)

HOG describe una imagen por las **direcciones de sus bordes** en celdas pequeñas. En cada
celda cuenta hacia dónde apuntan los gradientes (los cambios de intensidad).

> **Por qué HOG sirve aquí.** Un rayón o un abollón agregan **bordes fuertes y orientados**
> (la línea del rayón, el borde del abollón). Un panel liso no los tiene. Entonces los
> vectores HOG de un panel dañado y uno limpio son muy distintos, y un clasificador los
> separa fácil.

En `features.py` usamos `skimage.feature.hog` con 8 orientaciones y celdas de 16×16.

## El clasificador: Random Forest

En `model.py`. Un **bosque aleatorio** (muchos árboles de decisión que votan) sobre las
características HOG decide dañado/limpio. Lo elegimos porque es preciso en estos datos, rápido,
y **se exporta limpio a ONNX** (capítulo 2).

> **Concepto — Random Forest.** Entrena muchos árboles sobre subconjuntos de los datos y
> promedia sus votos. Es robusto, no necesita mucho ajuste y da probabilidades.

## Las métricas

Sobre un conjunto de prueba separado reportamos:

* **Accuracy:** porcentaje de aciertos.
* **Precision:** de lo que marqué como dañado, ¿cuánto lo estaba?
* **Recall:** de lo dañado, ¿cuánto detecté?
* **ROC-AUC:** capacidad de ordenar dañado vs. limpio (0.5 azar, 1.0 perfecto).

En el conjunto sintético: accuracy ~0.85, ROC-AUC ~0.91.

## El dato es sintético (y controlado)

En `generate.py` generamos paneles: base lisa con gradiente, algo de ruido y **reflejos**
suaves (que el modelo NO debe confundir con daño), y para los dañados agregamos rayones y
abollones. Guardamos la **máscara** y las **cajas** reales del daño, para poder medir la
detección y la localización contra la verdad.
