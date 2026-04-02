"""
plot_curve.py
─────────────
Parses your training log and plots a proper performance curve.
Run:  python3 plot_curve.py
Saves → performance_curve.png
"""

import re
import matplotlib.pyplot as plt

# ── Paste your full training log here ────────────────────────────────────────
TRAINING_LOG = """
Step      0 | lr 6.00e-07 | Train Loss: 8.5291 | Acc: 0.0508 | Top5: 0.2344 | ScoreErr: 0.2165 || Val Loss: 8.5406 | Acc: 0.0586 | Top5: 0.1992 | ScoreErr: 0.1838
Step    100 | lr 6.06e-05 | Train Loss: 7.1481 | Acc: 0.1016 | Top5: 0.2422 | ScoreErr: 0.0822 || Val Loss: 7.1531 | Acc: 0.0664 | Top5: 0.2578 | ScoreErr: 0.0782
Step    200 | lr 1.21e-04 | Train Loss: 6.2871 | Acc: 0.1367 | Top5: 0.2891 | ScoreErr: 0.0351 || Val Loss: 6.2541 | Acc: 0.1133 | Top5: 0.3086 | ScoreErr: 0.0455
Step    300 | lr 1.81e-04 | Train Loss: 6.1320 | Acc: 0.1133 | Top5: 0.3125 | ScoreErr: 0.0493 || Val Loss: 6.1767 | Acc: 0.1055 | Top5: 0.3516 | ScoreErr: 0.0567
Step    400 | lr 2.41e-04 | Train Loss: 6.3396 | Acc: 0.0820 | Top5: 0.2734 | ScoreErr: 0.0716 || Val Loss: 6.2277 | Acc: 0.1250 | Top5: 0.3086 | ScoreErr: 0.0410
Step    500 | lr 3.00e-04 | Train Loss: 6.0457 | Acc: 0.1328 | Top5: 0.3711 | ScoreErr: 0.0483 || Val Loss: 6.1407 | Acc: 0.0938 | Top5: 0.3086 | ScoreErr: 0.0500
Step    600 | lr 3.00e-04 | Train Loss: 5.9020 | Acc: 0.1094 | Top5: 0.3555 | ScoreErr: 0.0337 || Val Loss: 5.9817 | Acc: 0.1328 | Top5: 0.3633 | ScoreErr: 0.0459
Step    700 | lr 3.00e-04 | Train Loss: 5.9421 | Acc: 0.1055 | Top5: 0.3516 | ScoreErr: 0.0423 || Val Loss: 5.8983 | Acc: 0.0977 | Top5: 0.3164 | ScoreErr: 0.0298
Step    800 | lr 3.00e-04 | Train Loss: 5.8536 | Acc: 0.1328 | Top5: 0.3398 | ScoreErr: 0.0381 || Val Loss: 5.9526 | Acc: 0.1094 | Top5: 0.3477 | ScoreErr: 0.0670
Step    900 | lr 3.00e-04 | Train Loss: 5.8002 | Acc: 0.0898 | Top5: 0.3320 | ScoreErr: 0.0500 || Val Loss: 5.9095 | Acc: 0.1094 | Top5: 0.3711 | ScoreErr: 0.0577
Step   1000 | lr 3.00e-04 | Train Loss: 5.6839 | Acc: 0.1250 | Top5: 0.3555 | ScoreErr: 0.0431 || Val Loss: 6.0164 | Acc: 0.1172 | Top5: 0.3125 | ScoreErr: 0.0530
Step   1100 | lr 3.00e-04 | Train Loss: 5.7720 | Acc: 0.1055 | Top5: 0.3633 | ScoreErr: 0.0387 || Val Loss: 5.7002 | Acc: 0.1094 | Top5: 0.3516 | ScoreErr: 0.0425
Step   1200 | lr 3.00e-04 | Train Loss: 5.6884 | Acc: 0.1250 | Top5: 0.3711 | ScoreErr: 0.0348 || Val Loss: 5.7693 | Acc: 0.1289 | Top5: 0.3398 | ScoreErr: 0.0421
Step   1300 | lr 3.00e-04 | Train Loss: 5.7935 | Acc: 0.1133 | Top5: 0.3516 | ScoreErr: 0.0384 || Val Loss: 5.7199 | Acc: 0.1719 | Top5: 0.3945 | ScoreErr: 0.0401
Step   1400 | lr 3.00e-04 | Train Loss: 5.6853 | Acc: 0.1328 | Top5: 0.3242 | ScoreErr: 0.0443 || Val Loss: 5.6347 | Acc: 0.1289 | Top5: 0.3477 | ScoreErr: 0.0383
Step   1500 | lr 2.99e-04 | Train Loss: 5.6334 | Acc: 0.1406 | Top5: 0.3711 | ScoreErr: 0.0410 || Val Loss: 5.5652 | Acc: 0.1406 | Top5: 0.3477 | ScoreErr: 0.0455
Step   1600 | lr 2.99e-04 | Train Loss: 5.7426 | Acc: 0.1172 | Top5: 0.3477 | ScoreErr: 0.0467 || Val Loss: 5.4702 | Acc: 0.0703 | Top5: 0.3242 | ScoreErr: 0.0400
Step   1700 | lr 2.99e-04 | Train Loss: 5.7245 | Acc: 0.1406 | Top5: 0.3672 | ScoreErr: 0.0490 || Val Loss: 5.8156 | Acc: 0.0781 | Top5: 0.3672 | ScoreErr: 0.0340
Step   1800 | lr 2.99e-04 | Train Loss: 5.4143 | Acc: 0.1406 | Top5: 0.3711 | ScoreErr: 0.0457 || Val Loss: 5.7442 | Acc: 0.1484 | Top5: 0.3438 | ScoreErr: 0.0586
Step   1900 | lr 2.99e-04 | Train Loss: 5.6415 | Acc: 0.1406 | Top5: 0.3555 | ScoreErr: 0.0432 || Val Loss: 5.5386 | Acc: 0.0938 | Top5: 0.3555 | ScoreErr: 0.0480
Step   2000 | lr 2.99e-04 | Train Loss: 5.5256 | Acc: 0.1641 | Top5: 0.3906 | ScoreErr: 0.0371 || Val Loss: 5.4830 | Acc: 0.1055 | Top5: 0.3516 | ScoreErr: 0.0461
Step   2100 | lr 2.98e-04 | Train Loss: 5.5396 | Acc: 0.1133 | Top5: 0.3242 | ScoreErr: 0.0258 || Val Loss: 5.6571 | Acc: 0.1172 | Top5: 0.3555 | ScoreErr: 0.0363
Step   2200 | lr 2.98e-04 | Train Loss: 5.3914 | Acc: 0.1641 | Top5: 0.3945 | ScoreErr: 0.0418 || Val Loss: 5.8564 | Acc: 0.1094 | Top5: 0.3125 | ScoreErr: 0.0472
Step   2300 | lr 2.98e-04 | Train Loss: 5.3316 | Acc: 0.1055 | Top5: 0.4062 | ScoreErr: 0.0314 || Val Loss: 5.4461 | Acc: 0.1289 | Top5: 0.3906 | ScoreErr: 0.0403
Step   2400 | lr 2.98e-04 | Train Loss: 5.3703 | Acc: 0.1367 | Top5: 0.4062 | ScoreErr: 0.0323 || Val Loss: 5.7402 | Acc: 0.0742 | Top5: 0.3086 | ScoreErr: 0.0472
Step   2500 | lr 2.98e-04 | Train Loss: 5.4591 | Acc: 0.1445 | Top5: 0.3477 | ScoreErr: 0.0353 || Val Loss: 5.5812 | Acc: 0.1523 | Top5: 0.3867 | ScoreErr: 0.0368
Step   2600 | lr 2.97e-04 | Train Loss: 5.4701 | Acc: 0.1172 | Top5: 0.3984 | ScoreErr: 0.0360 || Val Loss: 5.4944 | Acc: 0.1250 | Top5: 0.3398 | ScoreErr: 0.0460
Step   2700 | lr 2.97e-04 | Train Loss: 5.1928 | Acc: 0.1484 | Top5: 0.3633 | ScoreErr: 0.0534 || Val Loss: 5.4387 | Acc: 0.1641 | Top5: 0.3555 | ScoreErr: 0.0388
Step   2800 | lr 2.97e-04 | Train Loss: 5.4558 | Acc: 0.1562 | Top5: 0.3750 | ScoreErr: 0.0516 || Val Loss: 5.5498 | Acc: 0.1367 | Top5: 0.3438 | ScoreErr: 0.0484
Step   2900 | lr 2.97e-04 | Train Loss: 5.3788 | Acc: 0.1094 | Top5: 0.3398 | ScoreErr: 0.0447 || Val Loss: 5.3780 | Acc: 0.1133 | Top5: 0.3555 | ScoreErr: 0.0398
Step   3000 | lr 2.96e-04 | Train Loss: 5.2291 | Acc: 0.1445 | Top5: 0.3984 | ScoreErr: 0.0381 || Val Loss: 5.5243 | Acc: 0.1133 | Top5: 0.3281 | ScoreErr: 0.0465
Step   3100 | lr 2.96e-04 | Train Loss: 5.2938 | Acc: 0.1406 | Top5: 0.4414 | ScoreErr: 0.0413 || Val Loss: 5.5476 | Acc: 0.0898 | Top5: 0.3242 | ScoreErr: 0.0303
Step   3200 | lr 2.96e-04 | Train Loss: 5.1686 | Acc: 0.1289 | Top5: 0.3750 | ScoreErr: 0.0439 || Val Loss: 5.2303 | Acc: 0.1289 | Top5: 0.4141 | ScoreErr: 0.0386
Step   3300 | lr 2.95e-04 | Train Loss: 5.2766 | Acc: 0.1797 | Top5: 0.4062 | ScoreErr: 0.0441 || Val Loss: 5.2922 | Acc: 0.1484 | Top5: 0.4414 | ScoreErr: 0.0402
Step   3400 | lr 2.95e-04 | Train Loss: 5.1885 | Acc: 0.1289 | Top5: 0.3828 | ScoreErr: 0.0334 || Val Loss: 5.2373 | Acc: 0.1094 | Top5: 0.4102 | ScoreErr: 0.0506
Step   3500 | lr 2.95e-04 | Train Loss: 5.0490 | Acc: 0.1172 | Top5: 0.4141 | ScoreErr: 0.0404 || Val Loss: 5.0586 | Acc: 0.1602 | Top5: 0.3984 | ScoreErr: 0.0449
Step   3600 | lr 2.94e-04 | Train Loss: 5.2390 | Acc: 0.1719 | Top5: 0.3828 | ScoreErr: 0.0368 || Val Loss: 5.2793 | Acc: 0.1445 | Top5: 0.4180 | ScoreErr: 0.0356
Step   3700 | lr 2.94e-04 | Train Loss: 5.1955 | Acc: 0.1328 | Top5: 0.4102 | ScoreErr: 0.0377 || Val Loss: 5.2549 | Acc: 0.1641 | Top5: 0.4297 | ScoreErr: 0.0406
Step   3800 | lr 2.94e-04 | Train Loss: 5.2459 | Acc: 0.1055 | Top5: 0.3516 | ScoreErr: 0.0406 || Val Loss: 5.3761 | Acc: 0.1562 | Top5: 0.4023 | ScoreErr: 0.0589
Step   3900 | lr 2.93e-04 | Train Loss: 4.8859 | Acc: 0.1680 | Top5: 0.4336 | ScoreErr: 0.0463 || Val Loss: 5.1647 | Acc: 0.1602 | Top5: 0.3945 | ScoreErr: 0.0467
Step   4000 | lr 2.93e-04 | Train Loss: 5.2283 | Acc: 0.1250 | Top5: 0.3867 | ScoreErr: 0.0492 || Val Loss: 5.1450 | Acc: 0.1484 | Top5: 0.4648 | ScoreErr: 0.0453
Step   4100 | lr 2.92e-04 | Train Loss: 4.9369 | Acc: 0.1836 | Top5: 0.4375 | ScoreErr: 0.0363 || Val Loss: 5.1300 | Acc: 0.1328 | Top5: 0.4102 | ScoreErr: 0.0536
Step   4200 | lr 2.92e-04 | Train Loss: 5.0726 | Acc: 0.1523 | Top5: 0.3906 | ScoreErr: 0.0571 || Val Loss: 5.0696 | Acc: 0.1562 | Top5: 0.3789 | ScoreErr: 0.0569
Step   4300 | lr 2.91e-04 | Train Loss: 4.9817 | Acc: 0.1562 | Top5: 0.4336 | ScoreErr: 0.0371 || Val Loss: 5.1570 | Acc: 0.1172 | Top5: 0.3789 | ScoreErr: 0.0555
Step   4400 | lr 2.91e-04 | Train Loss: 4.9430 | Acc: 0.1602 | Top5: 0.3945 | ScoreErr: 0.0365 || Val Loss: 5.0154 | Acc: 0.1445 | Top5: 0.4023 | ScoreErr: 0.0581
Step   4500 | lr 2.90e-04 | Train Loss: 5.0033 | Acc: 0.1211 | Top5: 0.4062 | ScoreErr: 0.0444 || Val Loss: 4.9500 | Acc: 0.1406 | Top5: 0.4414 | ScoreErr: 0.0444
Step   4600 | lr 2.90e-04 | Train Loss: 4.8821 | Acc: 0.1641 | Top5: 0.4336 | ScoreErr: 0.0395 || Val Loss: 5.0879 | Acc: 0.1406 | Top5: 0.3828 | ScoreErr: 0.0359
Step   4700 | lr 2.90e-04 | Train Loss: 4.7977 | Acc: 0.1484 | Top5: 0.4297 | ScoreErr: 0.0472 || Val Loss: 5.0053 | Acc: 0.1367 | Top5: 0.4258 | ScoreErr: 0.0535
Step   4800 | lr 2.89e-04 | Train Loss: 4.8371 | Acc: 0.2031 | Top5: 0.4648 | ScoreErr: 0.0368 || Val Loss: 5.1060 | Acc: 0.1250 | Top5: 0.3828 | ScoreErr: 0.0433
Step   4900 | lr 2.89e-04 | Train Loss: 4.6667 | Acc: 0.2188 | Top5: 0.4688 | ScoreErr: 0.0428 || Val Loss: 4.6287 | Acc: 0.1719 | Top5: 0.4688 | ScoreErr: 0.0418
Step   5000 | lr 2.88e-04 | Train Loss: 4.7898 | Acc: 0.1523 | Top5: 0.3945 | ScoreErr: 0.0295 || Val Loss: 4.8857 | Acc: 0.1602 | Top5: 0.4180 | ScoreErr: 0.0406
Step   5100 | lr 2.87e-04 | Train Loss: 4.6568 | Acc: 0.1602 | Top5: 0.4219 | ScoreErr: 0.0469 || Val Loss: 5.1796 | Acc: 0.1328 | Top5: 0.4023 | ScoreErr: 0.0503
Step   5200 | lr 2.87e-04 | Train Loss: 4.6318 | Acc: 0.1758 | Top5: 0.4414 | ScoreErr: 0.0423 || Val Loss: 5.0625 | Acc: 0.1562 | Top5: 0.3945 | ScoreErr: 0.0424
Step   5300 | lr 2.86e-04 | Train Loss: 4.7810 | Acc: 0.1562 | Top5: 0.4258 | ScoreErr: 0.0482 || Val Loss: 5.0142 | Acc: 0.1797 | Top5: 0.4297 | ScoreErr: 0.0569
Step   5400 | lr 2.86e-04 | Train Loss: 4.7298 | Acc: 0.1758 | Top5: 0.4062 | ScoreErr: 0.0340 || Val Loss: 4.9286 | Acc: 0.1719 | Top5: 0.4023 | ScoreErr: 0.0466
Step   5500 | lr 2.85e-04 | Train Loss: 4.4474 | Acc: 0.1797 | Top5: 0.4727 | ScoreErr: 0.0319 || Val Loss: 4.7402 | Acc: 0.1797 | Top5: 0.4531 | ScoreErr: 0.0388
Step   5600 | lr 2.85e-04 | Train Loss: 4.4144 | Acc: 0.2070 | Top5: 0.4883 | ScoreErr: 0.0455 || Val Loss: 4.9561 | Acc: 0.1562 | Top5: 0.4570 | ScoreErr: 0.0501
Step   5700 | lr 2.84e-04 | Train Loss: 4.6865 | Acc: 0.1250 | Top5: 0.4609 | ScoreErr: 0.0615 || Val Loss: 4.5968 | Acc: 0.1992 | Top5: 0.4531 | ScoreErr: 0.0473
Step   5800 | lr 2.83e-04 | Train Loss: 4.7924 | Acc: 0.1992 | Top5: 0.4102 | ScoreErr: 0.0386 || Val Loss: 4.7220 | Acc: 0.1719 | Top5: 0.3906 | ScoreErr: 0.0395
Step   5900 | lr 2.83e-04 | Train Loss: 4.6526 | Acc: 0.1484 | Top5: 0.4180 | ScoreErr: 0.0336 || Val Loss: 4.8926 | Acc: 0.1367 | Top5: 0.3633 | ScoreErr: 0.0617
Step   6000 | lr 2.82e-04 | Train Loss: 4.5940 | Acc: 0.1836 | Top5: 0.4141 | ScoreErr: 0.0315 || Val Loss: 4.8124 | Acc: 0.1523 | Top5: 0.4648 | ScoreErr: 0.0497
Step   6100 | lr 2.82e-04 | Train Loss: 4.6565 | Acc: 0.1484 | Top5: 0.4219 | ScoreErr: 0.0503 || Val Loss: 4.5751 | Acc: 0.1875 | Top5: 0.4531 | ScoreErr: 0.0539
Step   6200 | lr 2.81e-04 | Train Loss: 4.6892 | Acc: 0.1719 | Top5: 0.4570 | ScoreErr: 0.0496 || Val Loss: 4.9155 | Acc: 0.1406 | Top5: 0.3906 | ScoreErr: 0.0475
Step   6300 | lr 2.80e-04 | Train Loss: 4.7008 | Acc: 0.1445 | Top5: 0.4375 | ScoreErr: 0.0326 || Val Loss: 4.7376 | Acc: 0.1641 | Top5: 0.4570 | ScoreErr: 0.0389
Step   6400 | lr 2.80e-04 | Train Loss: 4.5003 | Acc: 0.1797 | Top5: 0.4414 | ScoreErr: 0.0532 || Val Loss: 4.6882 | Acc: 0.1562 | Top5: 0.4023 | ScoreErr: 0.0518
Step   6500 | lr 2.79e-04 | Train Loss: 4.5596 | Acc: 0.1602 | Top5: 0.4492 | ScoreErr: 0.0494 || Val Loss: 4.5414 | Acc: 0.1758 | Top5: 0.4609 | ScoreErr: 0.0633
Step   6600 | lr 2.78e-04 | Train Loss: 4.5465 | Acc: 0.1641 | Top5: 0.3945 | ScoreErr: 0.0386 || Val Loss: 4.7029 | Acc: 0.1211 | Top5: 0.3789 | ScoreErr: 0.0504
Step   6700 | lr 2.77e-04 | Train Loss: 4.3908 | Acc: 0.1953 | Top5: 0.4805 | ScoreErr: 0.0431 || Val Loss: 4.7714 | Acc: 0.1992 | Top5: 0.4414 | ScoreErr: 0.0392
Step   6800 | lr 2.77e-04 | Train Loss: 4.6350 | Acc: 0.1680 | Top5: 0.4531 | ScoreErr: 0.0301 || Val Loss: 4.7560 | Acc: 0.1523 | Top5: 0.4336 | ScoreErr: 0.0583
Step   6900 | lr 2.76e-04 | Train Loss: 4.2786 | Acc: 0.1992 | Top5: 0.5273 | ScoreErr: 0.0483 || Val Loss: 4.5197 | Acc: 0.1719 | Top5: 0.4727 | ScoreErr: 0.0447
Step   7000 | lr 2.75e-04 | Train Loss: 4.5142 | Acc: 0.1914 | Top5: 0.4727 | ScoreErr: 0.0507 || Val Loss: 4.6102 | Acc: 0.1797 | Top5: 0.4688 | ScoreErr: 0.0364
Step   7100 | lr 2.75e-04 | Train Loss: 4.3709 | Acc: 0.1992 | Top5: 0.4766 | ScoreErr: 0.0446 || Val Loss: 4.7343 | Acc: 0.1680 | Top5: 0.4141 | ScoreErr: 0.0513
Step   7200 | lr 2.74e-04 | Train Loss: 4.4707 | Acc: 0.1680 | Top5: 0.4688 | ScoreErr: 0.0386 || Val Loss: 4.6165 | Acc: 0.1797 | Top5: 0.4570 | ScoreErr: 0.0429
Step   7300 | lr 2.73e-04 | Train Loss: 4.3736 | Acc: 0.1836 | Top5: 0.4883 | ScoreErr: 0.0416 || Val Loss: 4.6909 | Acc: 0.1562 | Top5: 0.4336 | ScoreErr: 0.0575
Step   7400 | lr 2.72e-04 | Train Loss: 4.2586 | Acc: 0.1875 | Top5: 0.4844 | ScoreErr: 0.0452 || Val Loss: 4.5062 | Acc: 0.1953 | Top5: 0.4688 | ScoreErr: 0.0368
Step   7500 | lr 2.72e-04 | Train Loss: 4.4127 | Acc: 0.1836 | Top5: 0.4805 | ScoreErr: 0.0404 || Val Loss: 4.6357 | Acc: 0.1602 | Top5: 0.4570 | ScoreErr: 0.0461
Step   7600 | lr 2.71e-04 | Train Loss: 4.1680 | Acc: 0.1875 | Top5: 0.4844 | ScoreErr: 0.0315 || Val Loss: 4.5985 | Acc: 0.1445 | Top5: 0.3945 | ScoreErr: 0.0410
Step   7700 | lr 2.70e-04 | Train Loss: 4.3047 | Acc: 0.2227 | Top5: 0.5039 | ScoreErr: 0.0376 || Val Loss: 4.5785 | Acc: 0.1602 | Top5: 0.4062 | ScoreErr: 0.0466
Step   7800 | lr 2.69e-04 | Train Loss: 4.3855 | Acc: 0.1914 | Top5: 0.4453 | ScoreErr: 0.0433 || Val Loss: 4.3727 | Acc: 0.1680 | Top5: 0.4883 | ScoreErr: 0.0466
Step   7900 | lr 2.68e-04 | Train Loss: 4.2753 | Acc: 0.1992 | Top5: 0.4844 | ScoreErr: 0.0645 || Val Loss: 4.4543 | Acc: 0.1953 | Top5: 0.4766 | ScoreErr: 0.0656
Step   8000 | lr 2.67e-04 | Train Loss: 4.1357 | Acc: 0.2109 | Top5: 0.4961 | ScoreErr: 0.0460 || Val Loss: 4.6884 | Acc: 0.1484 | Top5: 0.4414 | ScoreErr: 0.0477
Step   8100 | lr 2.67e-04 | Train Loss: 4.2343 | Acc: 0.1992 | Top5: 0.4883 | ScoreErr: 0.0425 || Val Loss: 4.3880 | Acc: 0.1797 | Top5: 0.4453 | ScoreErr: 0.0458
Step   8200 | lr 2.66e-04 | Train Loss: 4.4166 | Acc: 0.1523 | Top5: 0.4453 | ScoreErr: 0.0436 || Val Loss: 4.4427 | Acc: 0.1641 | Top5: 0.4258 | ScoreErr: 0.0459
Step   8300 | lr 2.65e-04 | Train Loss: 4.0773 | Acc: 0.2109 | Top5: 0.5273 | ScoreErr: 0.0633 || Val Loss: 4.4987 | Acc: 0.1992 | Top5: 0.4766 | ScoreErr: 0.0556
Step   8400 | lr 2.64e-04 | Train Loss: 4.1055 | Acc: 0.1602 | Top5: 0.4883 | ScoreErr: 0.0356 || Val Loss: 4.3640 | Acc: 0.1719 | Top5: 0.4727 | ScoreErr: 0.0546
Step   8500 | lr 2.63e-04 | Train Loss: 4.1115 | Acc: 0.2070 | Top5: 0.5078 | ScoreErr: 0.0324 || Val Loss: 4.1772 | Acc: 0.1914 | Top5: 0.4766 | ScoreErr: 0.0346
Step   8600 | lr 2.62e-04 | Train Loss: 4.0013 | Acc: 0.2461 | Top5: 0.5039 | ScoreErr: 0.0538 || Val Loss: 4.2772 | Acc: 0.1797 | Top5: 0.4844 | ScoreErr: 0.0513
Step   8700 | lr 2.61e-04 | Train Loss: 4.0940 | Acc: 0.1758 | Top5: 0.5039 | ScoreErr: 0.0316 || Val Loss: 4.1569 | Acc: 0.2070 | Top5: 0.4844 | ScoreErr: 0.0338
Step   8800 | lr 2.61e-04 | Train Loss: 3.9896 | Acc: 0.2422 | Top5: 0.5078 | ScoreErr: 0.0376 || Val Loss: 4.2317 | Acc: 0.1562 | Top5: 0.4922 | ScoreErr: 0.0529
Step   8900 | lr 2.60e-04 | Train Loss: 4.0433 | Acc: 0.1836 | Top5: 0.4844 | ScoreErr: 0.0350 || Val Loss: 4.3743 | Acc: 0.1836 | Top5: 0.4688 | ScoreErr: 0.0424
Step   9000 | lr 2.59e-04 | Train Loss: 4.0015 | Acc: 0.2227 | Top5: 0.4844 | ScoreErr: 0.0396 || Val Loss: 3.9924 | Acc: 0.2344 | Top5: 0.5078 | ScoreErr: 0.0362
Step   9100 | lr 2.58e-04 | Train Loss: 4.2054 | Acc: 0.1836 | Top5: 0.5273 | ScoreErr: 0.0373 || Val Loss: 4.1574 | Acc: 0.2109 | Top5: 0.5156 | ScoreErr: 0.0356
Step   9200 | lr 2.57e-04 | Train Loss: 4.0830 | Acc: 0.1875 | Top5: 0.5234 | ScoreErr: 0.0374 || Val Loss: 4.3426 | Acc: 0.1875 | Top5: 0.4609 | ScoreErr: 0.0489
Step   9300 | lr 2.56e-04 | Train Loss: 3.8963 | Acc: 0.1797 | Top5: 0.5508 | ScoreErr: 0.0385 || Val Loss: 4.3350 | Acc: 0.2031 | Top5: 0.4648 | ScoreErr: 0.0425
Step   9400 | lr 2.55e-04 | Train Loss: 3.9966 | Acc: 0.2188 | Top5: 0.5234 | ScoreErr: 0.0352 || Val Loss: 4.2696 | Acc: 0.1914 | Top5: 0.4844 | ScoreErr: 0.0565
Step   9500 | lr 2.54e-04 | Train Loss: 4.0908 | Acc: 0.1719 | Top5: 0.4883 | ScoreErr: 0.0411 || Val Loss: 4.2249 | Acc: 0.2070 | Top5: 0.5273 | ScoreErr: 0.0476
Step   9600 | lr 2.53e-04 | Train Loss: 3.6513 | Acc: 0.2578 | Top5: 0.6133 | ScoreErr: 0.0381 || Val Loss: 4.1866 | Acc: 0.1914 | Top5: 0.4883 | ScoreErr: 0.0403
Step   9700 | lr 2.52e-04 | Train Loss: 4.0583 | Acc: 0.2383 | Top5: 0.5078 | ScoreErr: 0.0390 || Val Loss: 4.1341 | Acc: 0.2188 | Top5: 0.4727 | ScoreErr: 0.0502
Step   9800 | lr 2.51e-04 | Train Loss: 3.8474 | Acc: 0.2148 | Top5: 0.5508 | ScoreErr: 0.0487 || Val Loss: 4.0968 | Acc: 0.1953 | Top5: 0.5195 | ScoreErr: 0.0407
Step   9900 | lr 2.50e-04 | Train Loss: 3.8494 | Acc: 0.2461 | Top5: 0.5391 | ScoreErr: 0.0332 || Val Loss: 4.2169 | Acc: 0.1758 | Top5: 0.4688 | ScoreErr: 0.0426
Step  10000 | lr 2.49e-04 | Train Loss: 4.0056 | Acc: 0.2031 | Top5: 0.5156 | ScoreErr: 0.0442 || Val Loss: 3.9798 | Acc: 0.2383 | Top5: 0.5273 | ScoreErr: 0.0419
Step  10100 | lr 2.48e-04 | Train Loss: 4.0075 | Acc: 0.1719 | Top5: 0.4961 | ScoreErr: 0.0455 || Val Loss: 4.2995 | Acc: 0.2031 | Top5: 0.4688 | ScoreErr: 0.0435
Step  10200 | lr 2.47e-04 | Train Loss: 3.9613 | Acc: 0.2109 | Top5: 0.5156 | ScoreErr: 0.0342 || Val Loss: 4.1395 | Acc: 0.1641 | Top5: 0.4414 | ScoreErr: 0.0478
Step  10300 | lr 2.46e-04 | Train Loss: 3.7441 | Acc: 0.2422 | Top5: 0.5430 | ScoreErr: 0.0395 || Val Loss: 4.2212 | Acc: 0.1484 | Top5: 0.4492 | ScoreErr: 0.0333
Step  10400 | lr 2.45e-04 | Train Loss: 3.7332 | Acc: 0.1992 | Top5: 0.5430 | ScoreErr: 0.0421 || Val Loss: 4.1601 | Acc: 0.1523 | Top5: 0.4492 | ScoreErr: 0.0313
Step  10500 | lr 2.44e-04 | Train Loss: 3.8306 | Acc: 0.1992 | Top5: 0.5156 | ScoreErr: 0.0367 || Val Loss: 4.0998 | Acc: 0.1875 | Top5: 0.5078 | ScoreErr: 0.0532
Step  10600 | lr 2.43e-04 | Train Loss: 3.7173 | Acc: 0.2227 | Top5: 0.5820 | ScoreErr: 0.0484 || Val Loss: 4.0240 | Acc: 0.1719 | Top5: 0.5195 | ScoreErr: 0.0432
Step  10700 | lr 2.42e-04 | Train Loss: 3.8470 | Acc: 0.1875 | Top5: 0.5117 | ScoreErr: 0.0345 || Val Loss: 4.0239 | Acc: 0.2344 | Top5: 0.5117 | ScoreErr: 0.0432
Step  10800 | lr 2.41e-04 | Train Loss: 3.7830 | Acc: 0.2383 | Top5: 0.5703 | ScoreErr: 0.0346 || Val Loss: 3.9149 | Acc: 0.1914 | Top5: 0.5469 | ScoreErr: 0.0445
Step  10900 | lr 2.40e-04 | Train Loss: 3.7627 | Acc: 0.2305 | Top5: 0.5898 | ScoreErr: 0.0286 || Val Loss: 4.0237 | Acc: 0.1719 | Top5: 0.5000 | ScoreErr: 0.0347
Step  11000 | lr 2.39e-04 | Train Loss: 3.7291 | Acc: 0.2188 | Top5: 0.5312 | ScoreErr: 0.0408 || Val Loss: 4.2338 | Acc: 0.1445 | Top5: 0.4609 | ScoreErr: 0.0474
Step  11100 | lr 2.38e-04 | Train Loss: 3.7385 | Acc: 0.2188 | Top5: 0.5469 | ScoreErr: 0.0407 || Val Loss: 4.1951 | Acc: 0.1406 | Top5: 0.4062 | ScoreErr: 0.0469
Step  11200 | lr 2.36e-04 | Train Loss: 3.4582 | Acc: 0.2461 | Top5: 0.6250 | ScoreErr: 0.0415 || Val Loss: 3.9091 | Acc: 0.2031 | Top5: 0.5000 | ScoreErr: 0.0400
Step  11300 | lr 2.35e-04 | Train Loss: 3.8268 | Acc: 0.2148 | Top5: 0.4961 | ScoreErr: 0.0458 || Val Loss: 3.9728 | Acc: 0.1914 | Top5: 0.4844 | ScoreErr: 0.0637
Step  11400 | lr 2.34e-04 | Train Loss: 3.3984 | Acc: 0.2734 | Top5: 0.6172 | ScoreErr: 0.0390 || Val Loss: 4.0063 | Acc: 0.1914 | Top5: 0.5000 | ScoreErr: 0.0470
Step  11500 | lr 2.33e-04 | Train Loss: 3.6561 | Acc: 0.2539 | Top5: 0.5898 | ScoreErr: 0.0280 || Val Loss: 3.9169 | Acc: 0.2266 | Top5: 0.4609 | ScoreErr: 0.0463
Step  11600 | lr 2.32e-04 | Train Loss: 3.5655 | Acc: 0.2383 | Top5: 0.5938 | ScoreErr: 0.0347 || Val Loss: 4.0545 | Acc: 0.1641 | Top5: 0.4688 | ScoreErr: 0.0396
Step  11700 | lr 2.31e-04 | Train Loss: 3.5490 | Acc: 0.2656 | Top5: 0.5977 | ScoreErr: 0.0459 || Val Loss: 3.9304 | Acc: 0.1875 | Top5: 0.5234 | ScoreErr: 0.0481
Step  11800 | lr 2.30e-04 | Train Loss: 3.6959 | Acc: 0.2344 | Top5: 0.5547 | ScoreErr: 0.0369 || Val Loss: 4.1439 | Acc: 0.1875 | Top5: 0.4492 | ScoreErr: 0.0574
Step  11900 | lr 2.29e-04 | Train Loss: 3.5709 | Acc: 0.2461 | Top5: 0.5938 | ScoreErr: 0.0345 || Val Loss: 4.1497 | Acc: 0.1758 | Top5: 0.5000 | ScoreErr: 0.0479
Step  12000 | lr 2.27e-04 | Train Loss: 3.4883 | Acc: 0.2383 | Top5: 0.5586 | ScoreErr: 0.0393 || Val Loss: 3.9396 | Acc: 0.1641 | Top5: 0.4531 | ScoreErr: 0.0370
Step  12100 | lr 2.26e-04 | Train Loss: 3.6529 | Acc: 0.2344 | Top5: 0.5547 | ScoreErr: 0.0420 || Val Loss: 3.9344 | Acc: 0.1953 | Top5: 0.5078 | ScoreErr: 0.0564
Step  12200 | lr 2.25e-04 | Train Loss: 3.5646 | Acc: 0.2773 | Top5: 0.5664 | ScoreErr: 0.0308 || Val Loss: 3.9782 | Acc: 0.1914 | Top5: 0.4961 | ScoreErr: 0.0580
Step  12300 | lr 2.24e-04 | Train Loss: 3.5041 | Acc: 0.2656 | Top5: 0.6172 | ScoreErr: 0.0507 || Val Loss: 4.0451 | Acc: 0.1523 | Top5: 0.4688 | ScoreErr: 0.0552
Step  12400 | lr 2.23e-04 | Train Loss: 3.5787 | Acc: 0.2227 | Top5: 0.5898 | ScoreErr: 0.0489 || Val Loss: 3.9974 | Acc: 0.1875 | Top5: 0.5000 | ScoreErr: 0.0409
Step  12500 | lr 2.22e-04 | Train Loss: 3.2946 | Acc: 0.2617 | Top5: 0.6055 | ScoreErr: 0.0386 || Val Loss: 3.8620 | Acc: 0.2188 | Top5: 0.5000 | ScoreErr: 0.0402
Step  12600 | lr 2.21e-04 | Train Loss: 3.5625 | Acc: 0.2188 | Top5: 0.5703 | ScoreErr: 0.0447 || Val Loss: 3.7344 | Acc: 0.1836 | Top5: 0.5039 | ScoreErr: 0.0345
Step  12700 | lr 2.19e-04 | Train Loss: 3.2656 | Acc: 0.2930 | Top5: 0.6406 | ScoreErr: 0.0357 || Val Loss: 3.9880 | Acc: 0.1719 | Top5: 0.5234 | ScoreErr: 0.0414
Step  12800 | lr 2.18e-04 | Train Loss: 3.3518 | Acc: 0.2422 | Top5: 0.6133 | ScoreErr: 0.0337 || Val Loss: 3.8662 | Acc: 0.2148 | Top5: 0.4961 | ScoreErr: 0.0577
Step  12900 | lr 2.17e-04 | Train Loss: 3.4081 | Acc: 0.3047 | Top5: 0.5938 | ScoreErr: 0.0456 || Val Loss: 3.6570 | Acc: 0.2344 | Top5: 0.5664 | ScoreErr: 0.0295
Step  13000 | lr 2.16e-04 | Train Loss: 3.3001 | Acc: 0.3047 | Top5: 0.6250 | ScoreErr: 0.0428 || Val Loss: 3.8741 | Acc: 0.2031 | Top5: 0.4766 | ScoreErr: 0.0540
Step  13100 | lr 2.15e-04 | Train Loss: 3.3454 | Acc: 0.2812 | Top5: 0.6016 | ScoreErr: 0.0328 || Val Loss: 3.5650 | Acc: 0.2188 | Top5: 0.5586 | ScoreErr: 0.0471
Step  13200 | lr 2.13e-04 | Train Loss: 3.3783 | Acc: 0.2930 | Top5: 0.6016 | ScoreErr: 0.0478 || Val Loss: 4.0179 | Acc: 0.1641 | Top5: 0.4844 | ScoreErr: 0.0330
Step  13300 | lr 2.12e-04 | Train Loss: 3.4077 | Acc: 0.2227 | Top5: 0.5508 | ScoreErr: 0.0379 || Val Loss: 4.0652 | Acc: 0.1875 | Top5: 0.4688 | ScoreErr: 0.0351
Step  13400 | lr 2.11e-04 | Train Loss: 3.2468 | Acc: 0.2578 | Top5: 0.6055 | ScoreErr: 0.0475 || Val Loss: 4.0355 | Acc: 0.1797 | Top5: 0.4766 | ScoreErr: 0.0418
Step  13500 | lr 2.10e-04 | Train Loss: 3.4078 | Acc: 0.2500 | Top5: 0.6016 | ScoreErr: 0.0377 || Val Loss: 3.9755 | Acc: 0.1875 | Top5: 0.4766 | ScoreErr: 0.0578
Step  13600 | lr 2.09e-04 | Train Loss: 3.0902 | Acc: 0.2891 | Top5: 0.6758 | ScoreErr: 0.0446 || Val Loss: 3.6058 | Acc: 0.2266 | Top5: 0.5195 | ScoreErr: 0.0421
Step  13700 | lr 2.07e-04 | Train Loss: 3.4832 | Acc: 0.2383 | Top5: 0.5273 | ScoreErr: 0.0472 || Val Loss: 3.5330 | Acc: 0.2188 | Top5: 0.5703 | ScoreErr: 0.0435
Step  13800 | lr 2.06e-04 | Train Loss: 3.2866 | Acc: 0.2578 | Top5: 0.6016 | ScoreErr: 0.0474 || Val Loss: 3.8373 | Acc: 0.2031 | Top5: 0.4766 | ScoreErr: 0.0343
Step  13900 | lr 2.05e-04 | Train Loss: 3.3671 | Acc: 0.2344 | Top5: 0.6250 | ScoreErr: 0.0391 || Val Loss: 3.6113 | Acc: 0.2070 | Top5: 0.5664 | ScoreErr: 0.0406
Step  14000 | lr 2.04e-04 | Train Loss: 3.1133 | Acc: 0.3164 | Top5: 0.6406 | ScoreErr: 0.0377 || Val Loss: 3.3944 | Acc: 0.2422 | Top5: 0.6289 | ScoreErr: 0.0351
Step  14100 | lr 2.02e-04 | Train Loss: 3.2737 | Acc: 0.3086 | Top5: 0.6055 | ScoreErr: 0.0265 || Val Loss: 3.8427 | Acc: 0.1836 | Top5: 0.5078 | ScoreErr: 0.0420
Step  14200 | lr 2.01e-04 | Train Loss: 3.2528 | Acc: 0.2617 | Top5: 0.5859 | ScoreErr: 0.0362 || Val Loss: 3.8221 | Acc: 0.2031 | Top5: 0.5117 | ScoreErr: 0.0449
Step  14300 | lr 2.00e-04 | Train Loss: 3.2105 | Acc: 0.2812 | Top5: 0.6016 | ScoreErr: 0.0401 || Val Loss: 3.7875 | Acc: 0.2148 | Top5: 0.5273 | ScoreErr: 0.0581
Step  14400 | lr 1.99e-04 | Train Loss: 3.0931 | Acc: 0.2773 | Top5: 0.6719 | ScoreErr: 0.0283 || Val Loss: 3.7016 | Acc: 0.2422 | Top5: 0.5156 | ScoreErr: 0.0477
Step  14500 | lr 1.97e-04 | Train Loss: 3.2328 | Acc: 0.2773 | Top5: 0.5977 | ScoreErr: 0.0319 || Val Loss: 3.5417 | Acc: 0.2656 | Top5: 0.5469 | ScoreErr: 0.0436
Step  14600 | lr 1.96e-04 | Train Loss: 3.1152 | Acc: 0.2695 | Top5: 0.6445 | ScoreErr: 0.0389 || Val Loss: 3.5566 | Acc: 0.2461 | Top5: 0.5430 | ScoreErr: 0.0568
Step  14700 | lr 1.95e-04 | Train Loss: 3.0715 | Acc: 0.2422 | Top5: 0.6289 | ScoreErr: 0.0400 || Val Loss: 3.7016 | Acc: 0.1914 | Top5: 0.4844 | ScoreErr: 0.0315
Step  14800 | lr 1.93e-04 | Train Loss: 3.3682 | Acc: 0.2539 | Top5: 0.6406 | ScoreErr: 0.0389 || Val Loss: 3.6026 | Acc: 0.2305 | Top5: 0.5391 | ScoreErr: 0.0404
Step  14900 | lr 1.92e-04 | Train Loss: 3.2103 | Acc: 0.2461 | Top5: 0.6055 | ScoreErr: 0.0338 || Val Loss: 3.6845 | Acc: 0.1719 | Top5: 0.5156 | ScoreErr: 0.0456
Step  15000 | lr 1.91e-04 | Train Loss: 2.9776 | Acc: 0.3008 | Top5: 0.6562 | ScoreErr: 0.0296 || Val Loss: 3.6226 | Acc: 0.2148 | Top5: 0.5391 | ScoreErr: 0.0425
Step  15100 | lr 1.90e-04 | Train Loss: 3.0716 | Acc: 0.2695 | Top5: 0.6289 | ScoreErr: 0.0330 || Val Loss: 3.4539 | Acc: 0.2383 | Top5: 0.5781 | ScoreErr: 0.0507
Step  15200 | lr 1.88e-04 | Train Loss: 3.0657 | Acc: 0.3281 | Top5: 0.6680 | ScoreErr: 0.0397 || Val Loss: 3.4951 | Acc: 0.2227 | Top5: 0.5508 | ScoreErr: 0.0331
Step  15300 | lr 1.87e-04 | Train Loss: 3.1412 | Acc: 0.2969 | Top5: 0.6719 | ScoreErr: 0.0317 || Val Loss: 3.7425 | Acc: 0.2227 | Top5: 0.5195 | ScoreErr: 0.0499
Step  15400 | lr 1.86e-04 | Train Loss: 3.0657 | Acc: 0.2852 | Top5: 0.6680 | ScoreErr: 0.0336 || Val Loss: 3.5812 | Acc: 0.2148 | Top5: 0.5586 | ScoreErr: 0.0468
Step  15500 | lr 1.85e-04 | Train Loss: 2.9112 | Acc: 0.3203 | Top5: 0.6914 | ScoreErr: 0.0402 || Val Loss: 3.8058 | Acc: 0.1953 | Top5: 0.4883 | ScoreErr: 0.0380
Step  15600 | lr 1.83e-04 | Train Loss: 3.0566 | Acc: 0.3125 | Top5: 0.6406 | ScoreErr: 0.0489 || Val Loss: 3.7410 | Acc: 0.2227 | Top5: 0.5273 | ScoreErr: 0.0470
Step  15700 | lr 1.82e-04 | Train Loss: 2.8345 | Acc: 0.3359 | Top5: 0.6992 | ScoreErr: 0.0449 || Val Loss: 3.7940 | Acc: 0.2070 | Top5: 0.5234 | ScoreErr: 0.0502
Step  15800 | lr 1.81e-04 | Train Loss: 3.0322 | Acc: 0.3086 | Top5: 0.6641 | ScoreErr: 0.0292 || Val Loss: 3.7054 | Acc: 0.1719 | Top5: 0.4922 | ScoreErr: 0.0361
Step  15900 | lr 1.79e-04 | Train Loss: 2.8547 | Acc: 0.3398 | Top5: 0.6914 | ScoreErr: 0.0405 || Val Loss: 3.6999 | Acc: 0.1914 | Top5: 0.5117 | ScoreErr: 0.0393
Step  16000 | lr 1.78e-04 | Train Loss: 2.7033 | Acc: 0.3633 | Top5: 0.7383 | ScoreErr: 0.0304 || Val Loss: 3.7161 | Acc: 0.1836 | Top5: 0.4922 | ScoreErr: 0.0334
"""

# ── Parse ─────────────────────────────────────────────────────────────────────
steps, train_losses, val_losses = [], [], []
pattern = re.compile(r"Step\s+(\d+).*?Train Loss:\s*([\d.]+).*?Val Loss:\s*([\d.]+)")

for match in pattern.finditer(TRAINING_LOG):
    steps.append(int(match.group(1)))
    train_losses.append(float(match.group(2)))
    val_losses.append(float(match.group(3)))

print(f"Parsed {len(steps)} entries.")

# ── Smooth ────────────────────────────────────────────────────────────────────
def smooth(values, weight=0.7):
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.plot(steps, smooth(train_losses), label="train", color="#4C9BE8", linewidth=2)
plt.plot(steps, smooth(val_losses),   label="test",  color="#F5A623", linewidth=2)
plt.legend(fontsize=12)
plt.title("Performance curve", fontsize=14)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.tight_layout()
plt.savefig("performance_curve.png", dpi=150)
plt.show()
print("✅ Saved → performance_curve.png")