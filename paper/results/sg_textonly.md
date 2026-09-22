# streamguard on test (threshold 0.5000)

| metric | value | 95% CI |
|---|---|---|
| traj/precision | 0.9681 | |
| traj/recall | 0.9781 | |
| traj/f1 | 0.9730 | [0.965, 0.981] |
| step_f1/benign | 0.9848 | |
| step_f1/injection_point | 0.9310 | |
| step_f1/hijacked | 0.9776 | |
| step_f1/failed_injection | 0.9068 | |
| step_f1/macro | 0.9501 | [0.942, 0.959] |
| injection_em | 0.8787 | |
| hijack_iou | 0.9592 | [0.946, 0.971] |
| flag_rate/attacked | 0.9781 | |
| flag_rate/benign | 0.0190 | |
| flag_rate/failed_attack | 0.0138 | |
| flag_rate/hard_negative | 0.0588 | |
| recall/delayed_execution | 0.8803 | |
| recall/full_hijack | 0.9978 | |
| recall/partial_hijack | 0.9904 | |
| prevent/blocked_before_execution | 0.9835 | |
| prevent/first_hijack_caught | 0.9768 | |
| prevent/benign_interruption | 0.0264 | |