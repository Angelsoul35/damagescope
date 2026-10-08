# 2. Localización, métricas y ONNX

## Ubicar el daño sin un detector entrenado

Un detector de objetos entrenado necesita muchas cajas etiquetadas. Tenemos una señal más
simple: **un panel limpio es liso**.

En `localize.py`:

1. Desenfocamos la imagen (gaussiano) para estimar el panel liso.
2. Calculamos el **residual** = |imagen − imagen desenfocada|. El daño, que es de alta
   frecuencia, resalta; lo liso casi desaparece.
3. Umbralizamos (media + k·desviación), cerramos huecos (morfología), agrupamos los píxeles
   brillantes en **regiones conectadas** y sacamos una **caja** por región sobre un área
   mínima.

> **Por qué funciona.** Los reflejos son suaves, así que el residual los ignora. Por eso el
> generador los agrega: para que la prueba sea justa. (Ver
> [ADR-0003](../adr/0003-residual-localization.md).)

## Medir la localización: IoU

> **Concepto — IoU (Intersection over Union).** Mide cuánto se solapan dos cajas: área de la
> intersección dividida por el área de la unión. 1.0 = perfectas, 0.0 = no se tocan.

Para cada caja real tomamos el mejor IoU contra cualquier caja predicha y promediamos. Como
el generador guarda las cajas reales, este número es real. En el conjunto sintético: IoU ~0.80.

## Severidad y costo

En `severity.py`. La severidad sale de la **fracción de área dañada** y el número de regiones
(none / low / medium / high). El costo es un modelo sintético simple (base por región + término
por área) — para el demo, no un estimado real.

## ONNX para CPU

> **Concepto — ONNX.** Un formato estándar para modelos de ML. Un modelo exportado a ONNX se
> corre con `onnxruntime` en CPU (o en otros lenguajes/runtimes) sin cargar todo el stack de
> entrenamiento.

En `model.py` exportamos el scaler + random forest a ONNX con `skl2onnx`. Al servir, cargamos
el `.onnx` con `onnxruntime`; no necesitamos scikit-learn. El HOG sigue en Python. Una prueba
verifica que la salida ONNX **coincide** con scikit-learn. (Ver
[ADR-0004](../adr/0004-onnx-export.md).)
