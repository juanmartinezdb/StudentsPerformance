import streamlit as st
import pickle

# Cargar el modelo y el DictVectorizer
with open('models/students-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Rendimiento Estudiantil")

# Formulario para introducir datos del cliente
st.header("Introduce los datos del alumno:")
gender = st.selectbox("Género", ["female", "male"])
race_ethnicity = st.selectbox("Grupo Étnico", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox("Nivel de educación de los padres", [
    "some high school", "high school", "some college", 
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Tipo de almuerzo", ["standard", "free/reduced"])
test_preparation_course = st.selectbox("Curso de preparación", ["none", "completed"])

# Botón de predicción
if st.button("Predecir"):
    # Crear un diccionario con los datos del cliente
    en_bruto_data = {
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "parental_level_of_education": parental_level_of_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course
    }

    # Definimos la limpieza igual que en el notebook (esto me dio error porque no me los anotaba igual, asi las etiquetas
    #para marcar se ven bien y luego lo traducimos a lo que entiende el modelo)
    def limpiar_texto(texto):
        return texto.lower().replace(' ', '_').replace('/', '_').replace("'", '_')

    # Creamos un nuevo diccionario con los datos limpios
    student_data = {k: limpiar_texto(v) for k, v in en_bruto_data.items()}

    # Transformar los datos del estudiante y predecir
    X_student = dv.transform([student_data])
    y_pred_proba = model.predict_proba(X_student)[0][1]

   # Mostrar resultado
    st.subheader("Resultado:")
    if y_pred_proba < 0.5:
        st.error(f"Probabilidad de suspender: {(1-y_pred_proba):.2f}")
    else:
        st.success(f"Probabilidad de APROBAR: {y_pred_proba:.2f}")