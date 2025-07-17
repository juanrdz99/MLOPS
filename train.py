# train.py
import mlflow
import mlflow.sklearn
import joblib
from mlflow.models.signature import infer_signature
from functions import load_data, preprocess, train_model, evaluate_model

if __name__ == '__main__':
    # 1. Cargar datos
    X_train, X_test, y_train, y_test = load_data('data/breast_cancer.csv')
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

    # 2. Configurar experimento
    mlflow.set_experiment('breast_cancer_classification')
    with mlflow.start_run() as run:
        params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        mlflow.log_params(params)

        # 3. Entrenar modelo
        model = train_model(X_train_s, y_train, **params)

        # 4. Evaluar y registrar métricas
        metrics = evaluate_model(model, X_test_s, y_test)
        mlflow.log_metrics(metrics)

        # 5. Registrar modelo en MLflow con firma e input_example
        signature = infer_signature(X_train_s, model.predict(X_train_s))
        mlflow.sklearn.log_model(
            sk_model=model,
            name='random_forest_model',
            signature=signature,
            input_example=X_test_s[:5]
        )

        # 6. Exportar scaler y modelo a archivos pickle para la API
        joblib.dump(scaler, 'standard_scaler.pkl')
        joblib.dump(model, 'random_forest_model.pkl')

        # 7. Registrar los pickles como artefactos secundarios en MLflow
        mlflow.log_artifact('standard_scaler.pkl', artifact_path='preprocessor')
        mlflow.log_artifact('random_forest_model.pkl', artifact_path='model_pickle')

    print('Entrenamiento y registro completados.')
