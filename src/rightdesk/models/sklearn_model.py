"""TF-IDF vocabulary/IDF and classifier are fitted exclusively on training text."""

import warnings
from pathlib import Path

import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from rightdesk.models.base import RoutingModel, training_fingerprint
from rightdesk.schemas import ModelPrediction, TicketInput, WorkflowConfig


class TfidfLogisticModel(RoutingModel):
    def __init__(self, workflow: WorkflowConfig, seed: int = 42):
        super().__init__(workflow)
        self.seed = seed
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                        max_features=30000,
                        lowercase=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed),
                ),
            ]
        )

    def fit(self, tickets: list[TicketInput], targets: list[str]) -> None:
        self.check_training(tickets, targets)
        if len(set(targets)) < 2:
            raise ValueError("Logistic regression requires at least two observed queues")
        self.fitted = False
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("error", ConvergenceWarning)
            self.pipeline.fit([ticket.text for ticket in tickets], targets)
        self.revision = (
            f"tfidf-logistic-v1:{training_fingerprint(tickets, targets)}:seed-{self.seed}"
        )
        self.metadata = {
            "training_cases": len(tickets),
            "seed": self.seed,
            "tfidf": {
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "min_df": 1,
                "max_features": 30000,
                "lowercase": True,
            },
            "classifier": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
            "vocabulary_size": len(self.pipeline["tfidf"].vocabulary_),
            "classes": self.pipeline.classes_.tolist(),
            "iterations": self.pipeline["classifier"].n_iter_.tolist(),
            "probability_semantics": "uncalibrated logistic regression",
        }
        self.fitted = True

    def predict_one(self, ticket: TicketInput) -> ModelPrediction:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        values = self.pipeline.predict_proba([ticket.text])[0]
        probabilities = dict.fromkeys(self.queues, 0.0)
        probabilities.update(
            {
                str(queue): float(value)
                for queue, value in zip(self.pipeline.classes_, values, strict=True)
            }
        )
        return self.result(ticket, queue_probabilities=probabilities)

    def save(self, directory: Path) -> None:
        super().save(directory)
        joblib.dump(self.pipeline, directory / "pipeline.joblib")
