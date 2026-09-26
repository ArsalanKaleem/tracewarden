| test set | n | traj/precision | traj/recall | traj/f1 | step_f1/macro | injection_em | hijack_iou | prevent/blocked_before_execution | prevent/first_hijack_caught | prevent/benign_interruption | conformal/benign_hold_or_block |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agentdrift | 1722 | 0.993 | 0.975 | 0.984 | 0.957 | 0.883 | 0.961 | 0.985 | 0.973 | 0.005 | 0.005 |
| real-A-7B | 450 | 0.468 | 0.259 | 0.333 | 0.263 | 0.029 | 0.098 | 0.106 | 0.137 | 0.132 | 0.132 |
| real-B-14B | 340 | 0.604 | 0.359 | 0.451 | 0.282 | 0.052 | 0.142 | 0.180 | 0.216 | 0.193 | 0.193 |
| multistep | 232 | 0.278 | 0.238 | 0.256 | 0.268 | 0.024 | 0.151 | 0.190 | 0.119 | 0.137 | 0.137 |
