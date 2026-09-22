# streamguard on test (threshold 0.5000)

| metric | value | 95% CI |
|---|---|---|
| traj/precision | 0.9719 | |
| traj/recall | 0.9819 | |
| traj/f1 | 0.9769 | [0.970, 0.985] |
| step_f1/benign | 0.9864 | |
| step_f1/injection_point | 0.9356 | |
| step_f1/hijacked | 0.9819 | |
| step_f1/failed_injection | 0.9064 | |
| step_f1/macro | 0.9526 | [0.944, 0.961] |
| injection_em | 0.8877 | |
| hijack_iou | 0.9658 | [0.957, 0.976] |
| flag_rate/attacked | 0.9819 | |
| flag_rate/benign | 0.0095 | |
| flag_rate/failed_attack | 0.0183 | |
| flag_rate/hard_negative | 0.0637 | |
| recall/delayed_execution | 0.8974 | |
| recall/full_hijack | 1.0000 | |
| recall/partial_hijack | 0.9904 | |
| prevent/blocked_before_execution | 0.9892 | |
| prevent/first_hijack_caught | 0.9806 | |
| prevent/benign_interruption | 0.0232 | |
| conformal/benign_hold_or_block | 0.0095 | |
| conformal/benign_block | 0.0000 | |
| conformal/attack_caught | 0.9755 | |
| conformal/attack_blocked | 0.8903 | |