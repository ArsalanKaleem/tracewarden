# streamguard on test (threshold 0.5000)

| metric | value | 95% CI |
|---|---|---|
| traj/precision | 0.9581 | |
| traj/recall | 0.9729 | |
| traj/f1 | 0.9654 | [0.957, 0.974] |
| step_f1/benign | 0.9763 | |
| step_f1/injection_point | 0.8762 | |
| step_f1/hijacked | 0.9680 | |
| step_f1/failed_injection | 0.8460 | |
| step_f1/macro | 0.9166 | [0.904, 0.926] |
| injection_em | 0.7948 | |
| hijack_iou | 0.9475 | [0.933, 0.959] |
| flag_rate/attacked | 0.9729 | |
| flag_rate/benign | 0.0324 | |
| flag_rate/failed_attack | 0.0046 | |
| flag_rate/hard_negative | 0.0735 | |
| recall/delayed_execution | 0.9316 | |
| recall/full_hijack | 0.9978 | |
| recall/partial_hijack | 0.9423 | |
| prevent/blocked_before_execution | 0.9818 | |
| prevent/first_hijack_caught | 0.9703 | |
| prevent/benign_interruption | 0.0348 | |