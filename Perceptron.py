import os
import numpy as np
import tkinter as tk
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

xIteraciones = []
yErrores = []

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, mensaje):
        self.text_widget.insert(tk.END, mensaje)
        self.text_widget.see(tk.END)  

    def flush(self):
        pass

inputs = None
outputs = None
w = None
b = None

def limpiar_grafica(xIteraciones, yErrores):
    xIteraciones.clear()
    yErrores.clear()

    global line, ax, grafica, text_area

    line.set_xdata([])
    line.set_ydata([])
    ax.relim()
    ax.autoscale_view()
    grafica.draw()
    text_area.delete("1.0", tk.END)

def abrirArchivo():
    global inputs, outputs
    filepath = tk.filedialog.askopenfilename(title="Seleccionar archivo", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], initialdir=".")
    matriz = np.loadtxt(filepath, delimiter=',', skiprows=1)
    inputs = matriz[:,:-1]
    outputs = matriz[:,-1]
    print(inputs)
    print(outputs)

def cargarPesosUmbrales():
    global w, b
    try:
        filepath = tk.filedialog.askopenfilename(
            title="Seleccionar archivo de pesos y umbrales",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="."
        )
        with open(filepath, 'r') as f:
            lineas = f.readlines()
            w = np.fromstring(lineas[0].strip(), sep=',')  # primera línea → pesos
            b = float(lineas[1].strip())                   # segunda línea → bias

        print(f"Pesos cargados:\n{w}")        
        print(f"Bias cargado: {b}")
    except Exception as e:
        print(f"Error al cargar pesos y umbrales: {e}")


#Función de activación escalón
def activation_function(s):
    return 1 if s >= 0 else 0

def predict(input, weights, bias):
    s = np.dot(input, weights) + bias
    return activation_function(s)

def obtener_parametros(inpTasaDeAprendizaje, inpIteraciones, inpMaxError):
    global learning_rate, iterations, max_error, w, b
    try:
        learning_rate = float(inpTasaDeAprendizaje.get())
        iterations = int(inpIteraciones.get())
        max_error = float(inpMaxError.get())

        #print(f"Parámetros actualizados: Tasa de Aprendizaje={learning_rate}, Iteraciones={iterations}, Máximo Error={max_error}")
    except ValueError:
        print("Por favor, ingrese valores válidos para los parámetros.")
        return
    
    w,b = train(inputs, outputs, learning_rate, iterations)
    print(f"Pesos ajustados:\n{w}")
    print(f"Bias ajustado: {b}")

    almacenar_pesos_umbrales(w, b)

def almacenar_pesos_umbrales(weights, bias, carpeta_destino="pesos_umbrales"):
    # Preguntar si desea guardar
    respuesta = tk.messagebox.askyesno("Guardar Pesos y Umbrales", 
                                    "¿Desea guardar los pesos y umbrales actuales?")
    if not respuesta:
        print("Guardado cancelado por el usuario.")
        return

    # Crear carpeta si no existe
    os.makedirs(carpeta_destino, exist_ok=True)

    # Crear nombre único con fecha y hora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"pesos_umbrales_{timestamp}.csv"
    ruta_archivo = os.path.join(carpeta_destino, nombre_archivo)

    # Guardar datos
    with open(ruta_archivo, 'w') as f:
        f.write(','.join(map(str, weights)) + '\n')
        f.write(str(float(bias[0])) + '\n')

    print(f"Pesos y umbrales almacenados en {ruta_archivo}")

def limpiar_grafica():
    xIteraciones.clear()
    yErrores.clear()
    line.set_xdata([])
    line.set_ydata([])
    ax.relim()
    ax.autoscale_view()
    grafica.draw()
    text_area.delete("1.0", tk.END)

def train(inputs, output, learning_rate, iterations):        
    
    limpiar_grafica()
    
    if w is None or b is None or len(w) != inputs.shape[1]:
        print("Entrenando desde pesos y bias aleatorios.")
        #inicializar pesos y bias aleatoriamente    
        weights = np.random.rand(len(inputs[0])) #Un Peso por cada entrada
        bias = np.random.rand(1) #Un Bias por cada salida
    else:
        print("Entrenando desde pesos y bias cargados.")
        weights = w
        bias = b

    print(f"Pesos iniciales:\n{weights}")
    print(f"Bias inicial: {bias}")

    #Por cada iteración
    for iteration in range(iterations):
        total_error = 0
        print(f"Iteración {iteration+1}", end=" ")

        #Mientras el error sea mayor al máximo permitido

        #Leemos las entradas y salidas para realizar el ajuste de pesos y bias
        for input, output in zip(inputs, outputs):
            #print(f"Input: {input}, Output esperado: {output}", end=" ")

            #Calculamos la salida de la función soma
            s = np.dot(input, weights) + bias
            y_pred = activation_function(s)

            #Calculamos el error
            error = output - y_pred
            total_error += abs(error)

            #print(f"Salida predicha: {y_pred}, Error: {error}")

            #Ajustamos los pesos y bias
            delta_w = learning_rate * error * input
            weights += delta_w

            delta_b = learning_rate * error
            bias += delta_b
        
        average_error = total_error / len(inputs)
        print(f"Error promedio: {average_error}")

        xIteraciones.append(iteration + 1)
        yErrores.append(average_error)

        global line, ax, grafica

        line.set_xdata(xIteraciones)
        line.set_ydata(yErrores)
        ax.relim()
        ax.autoscale_view()        
        grafica.draw()
        root.update()

        # Verificamos si el error promedio es menor o igual al máximo permitido

        if average_error <= max_error:
            print("Error mínimo alcanzado, deteniendo entrenamiento.")
            break

    return weights, bias

#Ventana principal de entrenamiento
def crear_ventana(xIteraciones, yErrores, ConsoleRedirector, abrirArchivo, obtener_parametros):

    global inpTasaDeAprendizaje, inpIteraciones, inpMaxError
    global line, ax, grafica, text_area

    root = tk.Tk()
    fig, ax = plt.subplots()
    line, = ax.plot(xIteraciones, yErrores, 'b-')
    root.title("Perceptron")
    root.geometry("1150x500")

    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=0)
    root.rowconfigure(2, weight=0)
    root.rowconfigure(3, weight=0)
    root.rowconfigure(4, weight=0)
    root.columnconfigure(0, weight=0)
    root.columnconfigure(1, weight=0)
    root.columnconfigure(2, weight=0)
    root.columnconfigure(3, weight=0)
    root.columnconfigure(4, weight=0)

    botonCargarDS = tk.Button(root, text="Cargar Dataset", command=abrirArchivo)
    botonCargarDS.grid(column=0, row=0, sticky=tk.NW, padx=15, pady=15)

    botonCargarWU = tk.Button(root, text="Cargar Pesos y Umbrales", command=cargarPesosUmbrales)
    botonCargarWU.grid(column=1, row=0, sticky=tk.NW, padx=15, pady=15)

    lblTasaDeAprendizaje = tk.Label(root, text="Tasa de Aprendizaje:")
    lblTasaDeAprendizaje.grid(column=0, row=1, sticky=tk.N, padx=15, pady=15)

    lblIteraciones = tk.Label(root, text="Número de Iteraciones:")
    lblIteraciones.grid(column=0, row=2, sticky=tk.W, padx=15, pady=15)

    lblMaxError = tk.Label(root, text="Máximo Error:")
    lblMaxError.grid(column=0, row=3, sticky=tk.W, padx=15, pady=15)

    btnEntrenar = tk.Button(root, text="Entrenar", 
                            command=lambda: [obtener_parametros(inpTasaDeAprendizaje, 
                                                                inpIteraciones, 
                                                                inpMaxError)]
                            )
    btnEntrenar.grid(column=0, row=4, sticky=tk.W, padx=15, pady=15)

    inpTasaDeAprendizaje = tk.Entry(root)
    inpTasaDeAprendizaje.grid(column=1, row=1, sticky=tk.W, padx=15, pady=15)

    inpIteraciones = tk.Entry(root)
    inpIteraciones.grid(column=1, row=2, sticky=tk.W, padx=15, pady=15)

    inpMaxError = tk.Entry(root)
    inpMaxError.grid(column=1, row=3, sticky=tk.W, padx=15, pady=15)

    btnSimular = tk.Button(root, text="Simular", command=Simulacion)
    btnSimular.grid(column=0, row=4, sticky=tk.E, padx=15, pady=15)

    text_area = tk.Text(root, wrap="word", height=25, width=40)
    text_area.place(x=350, y=15)

    grafica = FigureCanvasTkAgg(fig, master=root)
    grafica.get_tk_widget().place(x=700, y=15, width=400, height=400)    

    sys.stdout = ConsoleRedirector(text_area)

    return root, line, ax, grafica, text_area

#Ventana de simulación
def Simulacion():
    ventana = tk.Tk()
    ventana.title("Simulación del Perceptron")
    ventana.geometry("550x250")

    ventana.rowconfigure(0, weight=0)
    ventana.rowconfigure(1, weight=0)
    ventana.rowconfigure(2, weight=0)
    ventana.columnconfigure(0, weight=0)
    ventana.columnconfigure(1, weight=0)
    ventana.columnconfigure(2, weight=0)
    ventana.columnconfigure(3, weight=0)

    def ejecutar_simulacion():
        global w, b
        entradas = list(map(float, inpX.get().split(",")))
        salida = predict(entradas, w, b)
        textAreaSalida.delete("1.0", tk.END)  # desde la línea 1, carácter 0, hasta el final
        textAreaSalida.insert(tk.END, str(salida))

    lblPatron1 = tk.Label(ventana, text="Patrón")
    lblPatron1.grid(column=0, row=0, sticky=tk.W, padx=15, pady=15)

    inpX = tk.Entry(ventana, textvariable="Digite las entradas separadas por comas")
    inpX.grid(column=0, row=1, sticky=tk.W, padx=15, pady=15)

    btnSimular = tk.Button(ventana, text="Simular", command=ejecutar_simulacion)
    btnSimular.grid(column=0, row=2, sticky=tk.W, padx=15, pady=15)

    btnCargarWU = tk.Button(ventana, text="Cargar Pesos y Umbrales", command=cargarPesosUmbrales)
    btnCargarWU.grid(column=0, row=3, sticky=tk.W, padx=15, pady=15)

    lblSalida = tk.Label(ventana, text="Salida")
    lblSalida.grid(column=3, row=0, sticky=tk.W, padx=15, pady=15)

    textAreaSalida = tk.Text(ventana, wrap="word", height=10, width=40)
    textAreaSalida.place(x=200, y=50)

    return ventana

root, line, ax, grafica, text_area = crear_ventana(
    xIteraciones, yErrores, ConsoleRedirector, abrirArchivo, obtener_parametros
)

root.mainloop()