import streamlit as st
import pickle

# Cargar el modelo y el DictVectorizer
with open('models/students-model-svm.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Rendimiento Estudiantil (SVM)")

# Formulario para introducir datos del alumno
st.header("Introduce los datos del alumno:")

#Diccionarios para los selectbox y luego para el diccionario del alumno
#Tiempo de estudio
studytime_op = {
    "< 2 horas": 1,
    "2 a 5 horas": 2,
    "5 a 10 horas": 3,
    "> 10 horas": 4
}
# Salir con amigos 
goout_op = {
    "Muy bajo": 1,
    "Bajo": 2,
    "Medio": 3,
    "Alto": 4,
    "Muy alto": 5
}
# Educación de la madre/padre 
edu_op = {
    "Ninguna": 0,
    "Educación Primaria (4º grado)": 1,
    "5º a 9º grado": 2,
    "Educación Secundaria": 3,
    "Educación Superior": 4
}
# Tiempo para llegar al instituto/colegio
travel_op = {
    "< 15 min.": 1,
    "15 a 30 min.": 2,
    "30 min. a 1 hora": 3,
    "> 1 hora": 4
}

higher = st.selectbox("Quiere cursar educación superior", ["yes", "no"])
mjob = st.selectbox("Trabajo de la madre", ["at_home", "health", "other", "services", "teacher"])
failures = st.number_input("Suspensos previos (0-4)", min_value=0, max_value=4, step=1)
age = st.number_input("Edad", min_value=15, max_value=22, step=1)
absences = st.number_input("Faltas", min_value=0, step=1)

studytime = st.selectbox("Tiempo de estudio semanal", list(studytime_op.keys()))
goout = st.selectbox("Frecuencia de salir con amigos", list(goout_op.keys()))
medu = st.selectbox("Nivel educativo de la Madre", list(edu_op.keys()))
fedu = st.selectbox("Nivel educativo del Padre", list(edu_op.keys()))
traveltime = st.selectbox("Tiempo para ir de casa a la escuela", list(travel_op.keys()))


# Botón de predicción
if st.button("Predecir"):
    # Crear un diccionario con los datos del alumno
    student_data = {
        "failures": failures,
        "age": age,
        "absences": absences,
        "studytime": studytime_op[studytime],
        "goout": goout_op[goout],
        "medu": edu_op[medu],
        "fedu": edu_op[fedu],
        "traveltime": travel_op[traveltime],
        "higher": higher,
        "mjob": mjob
    }

    # Transformar los datos del estudiante y predecir
    X_student = dv.transform([student_data])
    y_pred_proba = model.predict_proba(X_student)[0][1]

   # Mostrar resultado
    st.subheader("Resultado:")
    if y_pred_proba < 0.5:
        st.error(f"Probabilidad de suspender: {(1-y_pred_proba):.2f}")
    else:
        st.success(f"Probabilidad de APROBAR: {y_pred_proba:.2f}")