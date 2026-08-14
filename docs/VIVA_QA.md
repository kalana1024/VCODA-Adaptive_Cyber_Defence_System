# V-CODA Viva Questions and Suggested Answers

## Why is V-CODA hybrid?

Different models solve different problems. Supervised models recognise labelled attacks, anomaly models identify unusual traffic, deep models learn nonlinear representations or sequences, CTI and ATT&CK provide context, while a deterministic policy verifies response safety.

## Why use XGBoost as the primary supervised model?

NF-UQ-NIDS-v2 is structured flow data. XGBoost is a strong tabular baseline, supports nonlinear interactions, class weighting, probability output, efficient histogram training, and SHAP explanations. It is compared against other models rather than assumed best.

## Why not use only a CNN?

A CNN is justified when local order has meaning. Arbitrary feature-column order is not time. V-CODA uses CNN/CNN-LSTM/Transformer only when trustworthy flow sequences can be grouped and ordered.

## How do you prevent data leakage?

Inspection identifies labels, dataset origin, IP identifiers, flow IDs, constants, high-cardinality strings, and post-event fields. Preprocessing is fitted on training only, calibration and fusion use validation only, and final evaluation uses the held-out test source.

## What is zero-day detection here?

V-CODA does not guarantee zero-day detection. Benign-only anomaly models may flag behaviour different from learned normal traffic. Such an alert is an anomaly requiring investigation, not proof of a zero-day attack.

## How are ensemble weights selected?

V-CODA aligns model predictions by row ID on validation data. It either trains a logistic stacking model or searches constrained soft-voting weights to optimise the configured validation objective. Test data is not used to choose weights.

## How is MITRE ATT&CK used?

Predicted categories and behavioural evidence are mapped to probable ATT&CK candidates through maintainable YAML rules. Confidence and justification are returned. Exact techniques are not claimed when evidence is insufficient.

## What makes the system verifiable?

Every result contains model versions, checksums, preprocessing metadata, input coverage, probabilities, anomaly score, risk inputs, explanation method, MITRE justification, response decision, drift state, and a chained audit hash.

## What does self-healing mean?

It means operational recovery: detect corrupt models, quarantine them, roll back to a checksum-valid version, verify audit integrity, identify stale services, and reverse expired temporary actions. It does not mean unsupervised autonomous code modification.

## Why is automatic retraining disabled?

Unverified online labels can poison the model. Drift triggers preservation and review. A candidate model must be trained, validated, registered, and explicitly promoted.

## Why separate response policy from AI?

A probabilistic detector should not have unrestricted system authority. The policy engine independently checks thresholds, corroboration, allowlists, protected assets, confirmation, cooldown, and rollback.

## What are the main limitations?

Benchmark traffic differs from an organisation's live network. Temporal models need trustworthy sequence metadata. PCAP extraction may not reproduce every NF-UQ feature. CTI quality depends on sources. Anomaly alerts can be false positives. Windows capture and firewall behaviour depend on local permissions and drivers.
