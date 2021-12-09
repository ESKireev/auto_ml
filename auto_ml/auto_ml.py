import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from pathlib import Path
import click
import pandas as pd
from logger import get_logger

logger = get_logger()


class Metric:

    custom_metrics = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
        'roc_auc': roc_auc_score,
    }

    def __init__(self, name: str):
        self.name = name
        self._metric = self.custom_metrics[name]

    def __call__(self, y_true, y_pred):
        return self._metric(y_true, y_pred)


class TrainerModels:

    basic_models = {
        KNeighborsClassifier: {
            'n_neighbors': [3, 5, 11],
            'weights': ['uniform', 'distance'],
        },
        LogisticRegression: {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        },
        RandomForestClassifier: {
            'n_estimators': [10, 100, 1000],
            'max_features': ['sqrt', 'log2'],
        },
    }

    def __init__(
            self,
            target_features_path: Path,
            predictors_path: Path,
            metric: str,
            output_path: Path
    ):
        self._X = pd.read_csv(target_features_path, sep=';').values
        self._y = pd.read_csv(predictors_path, sep=';').values
        self._metric = Metric(metric)
        self._output_path = output_path
        self._best_score = 0.0
        self._best_model = None

    def _save_best_model(self):
        with open(self._output_path / 'model.pickle', 'wb') as file:
            pickle.dump(self._best_model, file)

    def _print_result(self):
        logger.info(f"Best model: {self._best_model}")
        logger.info(f"Best {self._metric.name} score: {self._best_score}")

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(self._X,
                                                            self._y,
                                                            test_size=0.25)
        y_train, y_test = y_train.ravel(), y_test.ravel()

        best_models_parameters = {}
        for model in self.basic_models:
            grid_search = GridSearchCV(
                estimator=model(),
                param_grid=self.basic_models.get(model),
                n_jobs=-1,
                cv=5,
                scoring=self._metric.name,
            )
            grid_search.fit(X_train, y_train)

            best_models_parameters[model] = grid_search.best_params_

        for model in best_models_parameters:
            clf = model(**best_models_parameters.get(model)).fit(
                X_train, y_train)
            y_pred = clf.predict(X_test)

            score = self._metric(y_test, y_pred)

            if score > self._best_score:
                self._best_score = score
                self._best_model = clf

        self._save_best_model()
        self._print_result()


@click.command()
@click.argument('target_features_path', type=click.Path(exists=True))
@click.argument('predictors_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
@click.argument('metric', type=click.Choice(list(Metric.custom_metrics.keys())))
def main(target_features_path, predictors_path, metric, output_path):
    target_features_path = Path(target_features_path)
    predictors_path = Path(predictors_path)
    output_path = Path(output_path)
    trainer = TrainerModels(target_features_path=target_features_path,
                            predictors_path=predictors_path,
                            metric=metric,
                            output_path=output_path)
    trainer.run()


if __name__ == '__main__':
    main()
