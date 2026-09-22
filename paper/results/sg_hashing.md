# streamguard on test (threshold 0.8144)

| metric | value | 95% CI |
|---|---|---|
| traj/precision | 0.9934 | |
| traj/recall | 0.9755 | |
| traj/f1 | 0.9844 | [0.978, 0.990] |
| step_f1/benign | 0.9880 | |
| step_f1/injection_point | 0.9400 | |
| step_f1/hijacked | 0.9863 | |
| step_f1/failed_injection | 0.9137 | |
| step_f1/macro | 0.9570 | [0.950, 0.964] |
| injection_em | 0.8826 | |
| hijack_iou | 0.9608 | [0.948, 0.972] |
| flag_rate/attacked | 0.9755 | |
| flag_rate/benign | 0.0000 | |
| flag_rate/failed_attack | 0.0000 | |
| flag_rate/hard_negative | 0.0245 | |
| recall/delayed_execution | 0.8718 | |
| recall/full_hijack | 1.0000 | |
| recall/partial_hijack | 0.9808 | |
| prevent/blocked_before_execution | 0.9852 | |
| prevent/first_hijack_caught | 0.9729 | |
| prevent/benign_interruption | 0.0053 | |
| conformal/benign_hold_or_block | 0.0053 | |
| conformal/benign_block | 0.0000 | |
| conformal/attack_caught | 0.9755 | |
| conformal/attack_blocked | 0.8916 | |