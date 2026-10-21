"""Load once per worker; bridge queue-compatible workflows explicitly for observation."""

import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from rightdesk.data.importers.abcd import checksum
from rightdesk.evaluation.calibration import TemperatureCalibrator
from rightdesk.evaluation.calibration_predictions import checked_run, frozen_baselines
from rightdesk.models.laya_model import LayaModel
from rightdesk.policy.clarification import ClarificationConfig, IntakePolicy
from rightdesk.policy.routing import RoutingSettings
from rightdesk.schemas import ModelPrediction, WorkflowConfig


class FrozenPredictor:
    def __init__(
        self,
        snapshot,
        source: Path,
        calibration: Path,
        adapter: Path,
        names=("majority", "rules", "tfidf_logistic", "laya"),
    ):
        if (
            not names
            or len(set(names)) != len(names)
            or not set(names) <= {"majority", "rules", "tfidf_logistic", "laya"}
        ):
            raise ValueError("Choose distinct supported shadow models")
        self.names = tuple(names)
        run, calibrated = checked_run(source), checked_run(calibration)
        if run["evaluation_split"] != "development" or calibrated["phase"] != 7:
            raise ValueError("Shadow worker needs frozen development models and calibration")
        if calibrated["source_run_sha256"] != checksum(source / "run.json"):
            raise ValueError("Frozen source/calibration mismatch")
        if checksum(adapter) != run["laya_config_sha256"]:
            raise ValueError("Frozen adapter changed")
        self.source_workflow = WorkflowConfig.model_validate(run["workflow"])
        primary = WorkflowConfig.model_validate(snapshot["workflow"])
        if primary.workflow_id != self.source_workflow.workflow_id or {
            q.id: q.description for q in primary.queues
        } != {q.id: q.description for q in self.source_workflow.queues}:
            raise ValueError("Workflow bridge requires identical queue IDs and descriptions")
        if calibrated["workflow"] != run["workflow"]:
            raise ValueError("Calibration workflow mismatch")
        self.models = frozen_baselines(source, self.source_workflow, run)
        if "laya" in names:
            self.models["laya"] = LayaModel(self.source_workflow, adapter)
        self.workflows, self.policies, self.calibrators = {}, {}, {}
        self.metadata = {
            "source_run_sha256": checksum(source / "run.json"),
            "calibration_run_sha256": checksum(calibration / "run.json"),
            "adapter_sha256": checksum(adapter),
            "models": list(names),
            "source_workflow_version": self.source_workflow.version,
            "observed_workflow_version": primary.version,
            "bridge": "identical queue semantics; new dialogue context is unvalidated",
            "routing": "disabled for live observation; no certification transfers across workflows",
            "clarification": "experimental proposals only; no question is sent",
            "code_sha256": {
                p.name: checksum(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "policies": {},
        }
        for name in names:
            kind = "native" if name == "rules" else "temperature"
            payload = json.loads((calibration / f"{name}_{kind}" / "policy.json").read_text())
            settings = RoutingSettings.model_validate({**payload["settings"], "enabled": False})
            if name != "rules":
                self.calibrators[name] = TemperatureCalibrator.model_validate_json(
                    (calibration / f"{name}_calibrator.json").read_text()
                )
            data = primary.model_dump()
            data["policy"].update(
                version=f"shadow_v1:{name}",
                routing_threshold=None,
                calibration_version=settings.calibration_version,
                sufficient_information_threshold=None,
            )
            workflow = WorkflowConfig.model_validate(data)
            self.workflows[name] = workflow
            self.policies[name] = IntakePolicy(
                workflow, settings, ClarificationConfig.model_validate(snapshot["clarification"])
            )
            self.metadata["policies"][name] = {
                "workflow": workflow.model_dump(),
                "settings": settings.model_dump(),
            }
            if self.models[name].revision != settings.model_revision:
                raise ValueError("Shadow model revision mismatch")

    def predict(self, name, ticket, asked):
        if ticket.language not in (None, "en"):
            return None, self.policies[name].decide(ticket, None, asked)
        with threadpool_limits(limits=1):
            if name == "laya":
                self.models[name].load()
            prediction = self.models[name].predict_one(ticket)
        if name in self.calibrators:
            prediction = self.calibrators[name].transform(prediction, self.source_workflow)
        data = prediction.model_dump()
        data["workflow_version"] = self.workflows[name].version
        prediction = ModelPrediction.for_workflow(data, self.workflows[name])
        return prediction, self.policies[name].decide(ticket, prediction, asked)

    def failure(self, name, ticket):
        return self.policies[name].decide(ticket, None, [])
