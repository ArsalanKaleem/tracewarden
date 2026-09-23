# streamguard on test (threshold 0.5000)

| metric | value | 95% CI |
|---|---|---|
| traj/precision | 0.9595 | |
| traj/recall | 0.9781 | |
| traj/f1 | 0.9687 | [0.959, 0.976] |
| step_f1/benign | 0.9824 | |
| step_f1/injection_point | 0.9010 | |
| step_f1/hijacked | 0.9706 | |
| step_f1/failed_injection | 0.9102 | |
| step_f1/macro | 0.9411 | [0.931, 0.950] |
| injection_em | 0.8606 | |
| hijack_iou | 0.9605 | [0.948, 0.971] |
| flag_rate/attacked | 0.9781 | |
| flag_rate/benign | 0.0400 | |
| flag_rate/failed_attack | 0.0000 | |
| flag_rate/hard_negative | 0.0539 | |
| recall/delayed_execution | 0.9145 | |
| recall/full_hijack | 1.0000 | |
| recall/partial_hijack | 0.9663 | |
| prevent/blocked_before_execution | 0.9886 | |
| prevent/first_hijack_caught | 0.9768 | |
| prevent/benign_interruption | 0.0338 | |