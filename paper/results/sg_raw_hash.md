# streamguard on test (threshold 0.5000)

| metric | value | 95% CI |
|---|---|---|
| traj/precision | 0.9831 | |
| traj/recall | 0.9768 | |
| traj/f1 | 0.9799 | [0.972, 0.987] |
| step_f1/benign | 0.9891 | |
| step_f1/injection_point | 0.9403 | |
| step_f1/hijacked | 0.9812 | |
| step_f1/failed_injection | 0.9469 | |
| step_f1/macro | 0.9644 | [0.956, 0.971] |
| injection_em | 0.8916 | |
| hijack_iou | 0.9590 | [0.946, 0.971] |
| flag_rate/attacked | 0.9768 | |
| flag_rate/benign | 0.0038 | |
| flag_rate/failed_attack | 0.0183 | |
| flag_rate/hard_negative | 0.0343 | |
| recall/delayed_execution | 0.8803 | |
| recall/full_hijack | 1.0000 | |
| recall/partial_hijack | 0.9808 | |
| prevent/blocked_before_execution | 0.9812 | |
| prevent/first_hijack_caught | 0.9755 | |
| prevent/benign_interruption | 0.0137 | |