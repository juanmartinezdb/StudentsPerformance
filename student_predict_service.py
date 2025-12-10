def predict_single(student, dv, model):
    x = dv.transform([student])
    y_pred = model.predict_proba(x)[:, 1]
    return (y_pred[0] >= 0.5, y_pred[0])