import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt


# 1. GENERACIÓN DE DATOS Y PREPARACIÓN

def generar_datos_senoidales(secuencia_total=1000):
    """Genera una onda senoidal simulando una serie temporal."""
    t = np.linspace(0, 50, secuencia_total)
    datos = np.sin(t)
    return datos

def preparar_ventanas(datos, tamano_ventana):
    """
    Divide los datos en ventanas de entrenamiento (X) y sus etiquetas (Y).
    El 'tamano_ventana' define cuántos pasos previos se usan para predecir el siguiente.
    """
    X, Y = [], []
    for i in range(len(datos) - tamano_ventana):
        X.append(datos[i:(i + tamano_ventana)])
        Y.append(datos[i + tamano_ventana])
    
    # Keras requiere formato (muestras, pasos_de_tiempo, caracteristicas)
    return np.array(X).reshape(-1, tamano_ventana, 1), np.array(Y)


# 2. CONSTRUCCIÓN Y ENTRENAMIENTO DE LA RED

def entrenar_evaluar_rnn(tamano_ventana, epochs=20):
    """Crea, entrena y evalúa una RNN básica dado un tamaño de ventana."""
    print(f"\n--- Entrenando modelo con Ventana de Entrenamiento = {tamano_ventana} ---")
    
    # Preparar datos
    datos_crudos = generar_datos_senoidales()
    X, Y = preparar_ventanas(datos_crudos, tamano_ventana)
    
    # Dividir en entrenamiento (80%) y prueba (20%)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Construir el modelo Recurrente
    modelo = Sequential([
        SimpleRNN(16, input_shape=(tamano_ventana, 1), activation='tanh'),
        Dense(1) # Salida lineal 
    ])
    
    # Compilar usando Error Cuadrático Medio (MSE) como tasa de error
    modelo.compile(optimizer='adam', loss='mse')
    
    # Entrenar el modelo
    historial = modelo.fit(X_train, Y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, Y_test))
    
    # Evaluar la tasa de error 
    tasa_error = modelo.evaluate(X_test, Y_test, verbose=0)
    print(f"Tasa de Error (MSE) final para ventana {tamano_ventana}: {tasa_error:.6f}")
    
    return tasa_error


# 3. EJECUCIÓN PRINCIPAL Y COMPARACIÓN

if __name__ == "__main__":

    # Evaluamos 3 escenarios de ventanas diferentes según lo solicitado en la actividad
    ventanas = [5, 15, 30]
    errores = []

    for v in ventanas:
        error = entrenar_evaluar_rnn(tamano_ventana=v)
        errores.append(error)

    # resultados 
    print("\n==========================================")
    print("RESUMEN DE TASAS DE ERROR (MSE)")
    print("==========================================")
    for i in range(len(ventanas)):
        print(f"Ventana de {ventanas[i]} pasos previos -> Error: {errores[i]:.6f}")