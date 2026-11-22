# Control summaries

Aggregated outputs of the additional controls (the `reproduce/control_*.py` scripts).
They back the control tables of the thesis appendix "Additional Controls for
Game-Based Attribution" and the corresponding main-text paragraphs. Per-image caches
and attribution maps are not shipped; the summaries contain the means, paired
differences, intervals and the provenance blocks written by the scripts.

| File | Control | Related scripts |
|---|---|---|
| `sharpening/sharpening_p15_validation_matched.json` | Fixed power sharpening, validation-matched comparison | `control_sharpening.py`, `control_report.py` |
| `full_auc/summary.json` | Full spatial-pixel deletion and insertion curves | `control_auc.py`, `control_auc_report.py` |
| `fulltest_robustness_summary.json` | ViT-B/16 risk coefficient versus zero risk on the 565 test images | `control_robustness.py`, `control_fulltest_robustness_report.py` |
| `vgg_gate_matched_summary.json` | VGG-16 gate-matched risk decomposition | `control_robustness.py`, `control_vgg_gate_matched_report.py` |
| `robustness_protocol_panel_summary.json` | Noise-protocol panel (distribution, scale, draw count) | `control_robustness_panel.py` |
| `manifests/robustness_*.json` | Evaluated rows of the robustness controls | `control_plan.py`, `control_report.py` |
| `manifests/vgg_gate_matched_rows.json` | Row of the gate-matched control with the probabilistic gate forced at zero risk | `control_robustness.py`, `control_vgg_gate_matched_report.py` |
| `sg_pool_all_metrics/summary.json` | Stopping Game original-forward max-pooling control | `control_sg_pool_all_metrics.py`, `control_sg_pool_table.py` |
| `sg_validation_audit.json`, `sg_numerical_audit.json` | Signed Stopping Game reconstruction against autograd | `control_sg_debug.py`, `control_sg_identity.py` |
| `rg_recovery_vgg16.json` | Routing Game versus Zennit alpha-beta LRP on VGG-16 | `control_rg_recovery.py` |
| `rg_recovery_vit_end_to_end.json`, `rg_recovery_vit.json` | Routing Game recovery on ViT-B/16, end to end and per attention block | `control_rg_vit_end_to_end.py`, `control_rg_attention_recovery.py` |
| `paired_prediction_performance/summary.json` | Prediction audit of the checkpoint-aligned randomization control | `control_prediction_performance.py` |
| `corrected_vit_randomization.json` | Checkpoint-aligned map and trajectory diagnostics | `control_diagnostics.py`, `control_randomization_tables.py` |
| `local_hellinger_controls.json` | Local Hellinger response to weight perturbations | `control_diagnostics.py`, `control_local_hellinger_table.py` |
| `localization_power_curves/protocol.md` | Frozen protocol of the cache-only power curves | `control_localization_power_curves.py` |

The controls ran with torch 2.5.1 and torchvision 0.20.1, as recorded in the
provenance blocks; `requirements-lock.txt` describes the environment of the main
evaluation. The provenance blocks record script and source hashes of the working copy that
produced each run. Later changes to this code are limited to the handling of empty
routing streams, routing temperatures below 0.1, default paths and comments; none of them
alters the evaluated configurations.
