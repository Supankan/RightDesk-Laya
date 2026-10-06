# Architecture

The CLI and Streamlit dashboard will call shared application services. Model
adapters produce predictions; application policy produces proposed actions.

```text
TicketInput + WorkflowConfig
             |
      Model adapter
             |
      ModelPrediction
             |
     Application policy
             |
      PolicyDecision ----> Shadow log and evaluation
                                  ^
                            ReviewedLabel
```

Long experiment runs will use the CLI and write results under `artifacts/`.
The dashboard will read those results and capture reviewer annotations in
SQLite. Filtering results should not trigger model inference again.

Start with rules, a conventional classifier, and Laya behind one interface.
Clarification selection requires a separate evaluation. Shadow operation
records proposals while the incumbent process continues to handle tickets.

HTTP integration can be added around the services when live shadow intake is
needed. The current scaffold contains no serving or execution implementation.
