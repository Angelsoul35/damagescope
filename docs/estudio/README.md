# Guía de estudio — DamageScope

Esta guía está en español y explica el proyecto para que lo estudies y lo sustentes en una
entrevista. El código y la documentación técnica están en inglés; aquí va el *por qué*.

## Orden de lectura

1. [Visión por computador: detección](01-deteccion-hog.md) — cómo se representa una imagen con
   HOG y cómo se clasifica daño vs. sin daño.
2. [Localización, métricas y ONNX](02-localizacion-y-onnx.md) — cómo se ubican los daños, cómo
   se mide con IoU y por qué exportar a ONNX.
3. [Preguntas de entrevista](03-preguntas-de-entrevista.md) — preguntas típicas con
   respuestas.

## Cómo correrlo mientras estudias

```bash
make install
make demo     # entrena, inspecciona un panel dañado y uno limpio, mide IoU
make test     # 19 pruebas
make export   # exporta el modelo a ONNX
```

El concepto central: **detectar daños en fotos de vehículos, ubicarlos con una caja, estimar
severidad y costo, y servir el modelo en CPU (ONNX)** — con visión clásica, honesta y medible.
