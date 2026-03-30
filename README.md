# Prediccion de Series Temporales con RNN | Time Series Prediction with RNN

Proyecto educativo en Python y TensorFlow para predecir el siguiente valor de una serie temporal senoidal usando una red neuronal recurrente simple.

Educational Python and TensorFlow project to predict the next value of a sinusoidal time series using a simple recurrent neural network.

## ES - Espanol

### Descripcion
Este proyecto implementa una RNN basica con `SimpleRNN` para aprender el patron de una onda senoidal.
Se comparan diferentes tamanos de ventana de entrenamiento para observar su impacto en el error final (MSE).

### Caracteristicas
- Generacion de datos sinteticos con una funcion senoidal.
- Preparacion de datos en ventanas deslizantes.
- Entrenamiento de modelo recurrente con TensorFlow/Keras.
- Evaluacion de rendimiento con MSE.
- Comparacion de resultados para varias ventanas: 5, 15 y 30.

### Estructura del proyecto
- `rnn_prediccion.py`: script principal con generacion de datos, entrenamiento y evaluacion.

### Requisitos
- Python 3.9 o superior
- TensorFlow
- NumPy
- Matplotlib

### Instalacion
1. Clona el repositorio.
2. (Opcional) Crea y activa un entorno virtual.
3. Instala dependencias:

```bash
pip install tensorflow numpy matplotlib
```

### Uso
Ejecuta el script principal:

```bash
python rnn_prediccion.py
```

### Salida esperada
El programa entrenara 3 modelos (una por tamano de ventana) y mostrara un resumen con el error MSE final por configuracion.

Ejemplo de salida:

```text
--- Entrenando modelo con Ventana de Entrenamiento = 5 ---
Tasa de Error (MSE) final para ventana 5: 0.000123
...
RESUMEN DE TASAS DE ERROR (MSE)
Ventana de 5 pasos previos -> Error: 0.000123
Ventana de 15 pasos previos -> Error: 0.000089
Ventana de 30 pasos previos -> Error: 0.000074
```

### Conceptos clave
- Ventana de entrenamiento: cantidad de pasos temporales previos usados para predecir el siguiente valor.
- MSE (Mean Squared Error): metrica de error promedio al cuadrado; menor es mejor.

### Posibles mejoras
- Agregar graficas de prediccion vs valor real.
- Probar capas `LSTM` o `GRU`.
- Guardar el mejor modelo entrenado.
- Introducir series temporales reales (ventas, sensores, etc.).

---

## EN - English

### Description
This project implements a basic RNN using `SimpleRNN` to learn the pattern of a sinusoidal wave.
It compares different training window sizes to analyze their effect on final error (MSE).

### Features
- Synthetic data generation with a sine function.
- Sliding-window dataset preparation.
- Recurrent model training with TensorFlow/Keras.
- Performance evaluation using MSE.
- Window-size comparison: 5, 15, and 30.

### Project structure
- `rnn_prediccion.py`: main script for data generation, training, and evaluation.

### Requirements
- Python 3.9+
- TensorFlow
- NumPy
- Matplotlib

### Installation
1. Clone this repository.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install tensorflow numpy matplotlib
```

### Usage
Run the main script:

```bash
python rnn_prediccion.py
```

### Expected output
The program trains 3 models (one per window size) and prints a summary with final MSE per configuration.

Example output:

```text
--- Training model with window size = 5 ---
Final Error Rate (MSE) for window 5: 0.000123
...
ERROR RATE SUMMARY (MSE)
Window 5 previous steps -> Error: 0.000123
Window 15 previous steps -> Error: 0.000089
Window 30 previous steps -> Error: 0.000074
```

### Key concepts
- Training window: number of previous time steps used to predict the next value.
- MSE (Mean Squared Error): average squared prediction error; lower is better.

### Potential improvements
- Add prediction-vs-ground-truth plots.
- Try `LSTM` or `GRU` layers.
- Save the best trained model.
- Use real-world time series (sales, sensor data, etc.).

---

## License
This project is intended for educational use.
