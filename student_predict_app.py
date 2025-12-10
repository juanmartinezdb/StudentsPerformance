import pickle
from flask import Flask, jsonify, request
from student_predict_service import predict_single

app = Flask('student-predict')


with open('models/students-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    student = request.get_json()
    passed, prediction = predict_single(student, dv, model)

    result = {
        'passed': bool(passed),
        'pass_probability': float(prediction),
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=9696)