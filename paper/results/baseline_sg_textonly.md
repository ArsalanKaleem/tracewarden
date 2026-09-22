# streamguard on test (threshold 0.5000)

| metric | value | 95% CI |
|---|---|---|
| traj/precision | 0.9438 | |
| traj/recall | 0.9535 | |
| traj/f1 | 0.9487 | [0.936, 0.960] |
| step_f1/benign | 0.9758 | |
| step_f1/injection_point | 0.8900 | |
| step_f1/hijacked | 0.9625 | |
| step_f1/failed_injection | 0.8126 | |
| step_f1/macro | 0.9102 | [0.897, 0.923] |
| injection_em | 0.8142 | |
| hijack_iou | 0.9317 | [0.917, 0.950] |
| flag_rate/attacked | 0.9535 | |
| flag_rate/benign | 0.0362 | |
| flag_rate/failed_attack | 0.0505 | |
| flag_rate/hard_negative | 0.0686 | |
| recall/delayed_execution | 0.8120 | |
| recall/full_hijack | 1.0000 | |
| recall/partial_hijack | 0.9327 | |
| prevent/blocked_before_execution | 0.9721 | |
| prevent/first_hijack_caught | 0.9523 | |
| prevent/benign_interruption | 0.0465 | |