"""Training-prior baseline: predicts the most common training queue."""

from collections import Counter

from rightdesk.models.base import RoutingModel, training_fingerprint
from rightdesk.schemas import ModelPrediction, TicketInput


class MajorityModel(RoutingModel):
    def fit(self, tickets: list[TicketInput], targets: list[str]) -> None:
        self.check_training(tickets, targets)
        counts = Counter(targets)
        self.probabilities = {queue: counts[queue] / len(targets) for queue in self.queues}
        self.revision = f"majority-v1:{training_fingerprint(tickets, targets)}"
        self.metadata = {
            "probability_semantics": "empirical training priors",
            "probabilities": self.probabilities,
            "training_cases": len(tickets),
        }
        self.fitted = True

    def predict_one(self, ticket: TicketInput) -> ModelPrediction:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.result(ticket, queue_probabilities=self.probabilities)
