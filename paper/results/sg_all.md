# streamguard on test (threshold 0.5000)

| metric | value | 95% CI |
|---|---|---|
| traj/precision | 0.9782 | |
| traj/recall | 0.9832 | |
| traj/f1 | 0.9807 | [0.974, 0.987] |
| step_f1/benign | 0.9871 | |
| step_f1/injection_point | 0.9352 | |
| step_f1/hijacked | 0.9825 | |
| step_f1/failed_injection | 0.9161 | |
| step_f1/macro | 0.9552 | [0.947, 0.963] |
| injection_em | 0.8839 | |
| hijack_iou | 0.9645 | [0.952, 0.975] |
| flag_rate/attacked | 0.9832 | |
| flag_rate/benign | 0.0038 | |
| flag_rate/failed_attack | 0.0183 | |
| flag_rate/hard_negative | 0.0539 | |
| recall/delayed_execution | 0.9060 | |
| recall/full_hijack | 1.0000 | |
| recall/partial_hijack | 0.9904 | |
| prevent/blocked_before_execution | 0.9892 | |
| prevent/first_hijack_caught | 0.9806 | |
| prevent/benign_interruption | 0.0180 | |
| conformal/benign_hold_or_block | 0.0053 | |
| conformal/benign_block | 0.0000 | |
| conformal/attack_caught | 0.9755 | |
| conformal/attack_blocked | 0.8916 | |