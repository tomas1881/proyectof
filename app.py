from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "clave_secreta"

# === CARGAR MODELOS Y ESCALADORES ===
modelo_logistica = joblib.load("modelos/modelo_logistica.pkl")
modelo_mlp = joblib.load("modelos/modelo_mlp.pkl")
modelo_svm = joblib.load("modelos/modelo_svm.pkl")
escalador_mlp = joblib.load("modelos/escalador_mlp.pkl")
escalador_fcm = joblib.load("modelos/escalador_fcm.pkl")
pesos_fcm = joblib.load("modelos/pesos_fcm.pkl")

# === COLUMNAS DE ENTRADA ===
columnas_modelo = [f"C{i}" for i in range(1, 31)]

# === FUNCIÓN PARA FCM ===
def predecir_fcm(X, pesos):
    activacion = np.dot(X, pesos)
    return (activacion > 0.5).astype(int)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/form_individual', methods=['GET', 'POST'])
def form_individual():
    campos = [
        {"nombre": "Edad de la madre (años)", "placeholder": "Ej: 28", "descripcion": "Edad actual de la madre"},
        {"nombre": "Índice de Masa Corporal (IMC)", "placeholder": "Ej: 24.5", "descripcion": "IMC actual"},
        {"nombre": "Edad gestacional al parto (semanas)", "placeholder": "Ej: 38", "descripcion": "Semanas de embarazo al parto"},
        {"nombre": "Gravidez", "placeholder": "Ej: 2", "descripcion": "Número de embarazos"},
        {"nombre": "Paridad", "placeholder": "Ej: 1", "descripcion": "Número de partos previos"},
        {"nombre": "Síntoma inicial (0:edema, 1:hipertensión, 2:FGR)", "placeholder": "Ej: 1", "descripcion": "Síntoma inicial observado"},
        {"nombre": "Edad gestacional al inicio del síntoma", "placeholder": "Ej: 32", "descripcion": "Semanas al inicio del síntoma"},
        {"nombre": "Días desde síntoma inicial al parto", "placeholder": "Ej: 14", "descripcion": "Intervalo en días"},
        {"nombre": "Edad gestacional de hipertensión", "placeholder": "Ej: 33", "descripcion": "Semanas de inicio de hipertensión"},
        {"nombre": "Días desde hipertensión al parto", "placeholder": "Ej: 10", "descripcion": "Intervalo en días"},
        {"nombre": "Edad gestacional de edema", "placeholder": "Ej: 31", "descripcion": "Semanas de inicio de edema"},
        {"nombre": "Días desde edema al parto", "placeholder": "Ej: 9", "descripcion": "Intervalo en días"},
        {"nombre": "Edad gestacional de proteinuria", "placeholder": "Ej: 34", "descripcion": "Semanas de inicio de proteinuria"},
        {"nombre": "Días desde proteinuria al parto", "placeholder": "Ej: 7", "descripcion": "Intervalo en días"},
        {"nombre": "Tratamiento expectante (0/1)", "placeholder": "Ej: 1", "descripcion": "Si recibió tratamiento expectante"},
        {"nombre": "Terapia antihipertensiva previa (0/1)", "placeholder": "Ej: 0", "descripcion": "Terapia antes de hospitalización"},
        {"nombre": "Antecedentes (0: No, 1: Hipertensión, 2: SOP)", "placeholder": "Ej: 1", "descripcion": "Condiciones médicas anteriores"},
        {"nombre": "Presión arterial sistólica máxima", "placeholder": "Ej: 120", "descripcion": "Máxima sistólica (mmHg)"},
        {"nombre": "Presión arterial diastólica máxima", "placeholder": "Ej: 80", "descripcion": "Máxima diastólica (mmHg)"},
        {"nombre": "Motivo del parto (0-5)", "placeholder": "Ej: 1", "descripcion": "Código de motivo del parto"},
        {"nombre": "Modo de parto (0: Cesárea, 1: Parto)", "placeholder": "Ej: 0", "descripcion": "Tipo de parto"},
        {"nombre": "BNP máximo", "placeholder": "Ej: 300", "descripcion": "Nivel máximo de BNP"},
        {"nombre": "Creatinina máxima", "placeholder": "Ej: 0.9", "descripcion": "Valor máximo de creatinina"},
        {"nombre": "Ácido úrico máximo", "placeholder": "Ej: 5.5", "descripcion": "Nivel máximo de ácido úrico"},
        {"nombre": "Proteinuria máxima", "placeholder": "Ej: 300", "descripcion": "Máximo nivel de proteína en orina"},
        {"nombre": "Proteína total máxima", "placeholder": "Ej: 6.5", "descripcion": "Proteína total (g/dL)"},
        {"nombre": "Albúmina máxima", "placeholder": "Ej: 3.5", "descripcion": "Nivel de albúmina"},
        {"nombre": "ALT (alanina aminotransferasa) máxima", "placeholder": "Ej: 35", "descripcion": "Enzima ALT"},
        {"nombre": "AST (aspartato aminotransferasa) máxima", "placeholder": "Ej: 30", "descripcion": "Enzima AST"},
        {"nombre": "Plaquetas máximas", "placeholder": "Ej: 250", "descripcion": "Conteo plaquetario"}
    ]

    valores = {f"input{i+1}": "" for i in range(30)}
    resultado = None
    modelo_usado = None

    if request.method == 'POST':
        try:
            datos = []
            for i in range(1, 31):
                val = request.form.get(f"input{i}", "")
                valores[f"input{i}"] = val
                datos.append(float(val))
            modelo_seleccionado = request.form.get("modelo")
            df = pd.DataFrame([datos], columns=columnas_modelo)

            if modelo_seleccionado == "logistica":
                pred = modelo_logistica.predict(df)[0]
                resultado = pred
                modelo_usado = "Regresión Logística"
            elif modelo_seleccionado == "mlp":
                df_escalado = escalador_mlp.transform(df)
                pred = modelo_mlp.predict(df_escalado)[0]
                resultado = pred
                modelo_usado = "Red Neuronal (MLP)"
            elif modelo_seleccionado == "svm":
                pred = modelo_svm.predict(df)[0]
                resultado = pred
                modelo_usado = "Máquina SVM"
            elif modelo_seleccionado == "fcm":
                df_escalado = escalador_fcm.transform(df)
                pred = predecir_fcm(df_escalado, pesos_fcm)[0]
                resultado = pred
                modelo_usado = "Mapa Cognitivo Difuso (FCM)"
        except Exception as e:
            resultado = f"Error: {str(e)}"
            modelo_usado = "Error en el procesamiento"

    return render_template("form_individual.html", campos=campos, valores=valores, resultado=resultado, modelo_usado=modelo_usado)

@app.route('/form_lote')
def form_lote():
    return render_template("form_lote.html")

@app.route('/lote', methods=['POST'])
def lote():
    try:
        archivo = request.files['archivo']
        modelo_seleccionado = request.form["modelo"]

        nombre_archivo = archivo.filename
        if not nombre_archivo.endswith((".xlsx", ".xls", ".csv")):
            return "Error: Solo se permiten archivos .xlsx, .xls o .csv"

        if nombre_archivo.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)

        X = df[columnas_modelo]
        y = df['C31']

        resultados = {}
        matrices = {}

        modelos_config = [
            ("Logística", modelo_logistica, "logistica", X),
            ("MLP", modelo_mlp, "mlp", escalador_mlp.transform(X)),
            ("SVM", modelo_svm, "svm", X)
        ]

        for nombre, modelo, clave, entrada in modelos_config:
            pred = modelo.predict(entrada)
            acc = accuracy_score(y, pred)
            cm = confusion_matrix(y, pred)
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d')
            plt.title(f"Matriz de Confusión - {nombre}")
            path_img = f"static/cm_{clave}.png"
            plt.savefig(path_img)
            resultados[nombre] = acc
            matrices[nombre] = path_img

        X_fcm = escalador_fcm.transform(X)
        pred_fcm = predecir_fcm(X_fcm, pesos_fcm)
        acc_fcm = accuracy_score(y, pred_fcm)
        cm_fcm = confusion_matrix(y, pred_fcm)
        plt.figure()
        sns.heatmap(cm_fcm, annot=True, fmt='d')
        plt.title("Matriz de Confusión - FCM")
        path_fcm = "static/cm_fcm.png"
        plt.savefig(path_fcm)
        resultados["FCM"] = acc_fcm
        matrices["FCM"] = path_fcm

        modelo_nombre_map = {
            "logistica": "Logística",
            "mlp": "MLP",
            "svm": "SVM",
            "fcm": "FCM"
        }

        modelo_usuario = modelo_nombre_map[modelo_seleccionado]
        exactitud = resultados[modelo_usuario]

        return render_template(
            "resultado_lote.html",
            resultados=resultados,
            matrices=matrices,
            modelo_usuario=modelo_usuario,
            modelo_usado=modelo_seleccionado,
            exactitud=exactitud
        )

    except Exception as e:
        return f"Error en predicción por lote: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
