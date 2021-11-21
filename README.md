# Branch-and-Cut

Implementation of Branch-n-Cut algorithm for Max Clique Problem via cplex-solver

For tests execution run command:
```
python run_test.py
```

|    | benchmark          | bnb_exec_time           |   bnb_exec_time_seconds |   bnb_clique_size |   true_clique_size |
|---:|:-------------------|:------------------------|------------------------:|------------------:|-------------------:|
|  0 | johnson8-2-4.clq   | 0min 0.0sec             |                    0.02 |                 4 |                  4 |
|  1 | johnson8-4-4.clq   | 0min 0.0sec             |                    0.02 |                14 |                 14 |
|  2 | johnson16-2-4.clq  | 0min 0.0sec             |                    0.02 |                 8 |                  8 |
|  3 | MANN_a9.clq        | 0min 0.8sec             |                    0.82 |                16 |                 16 |
|  4 | c-fat200-1.clq     | 0min 0.1sec             |                    0.1  |                12 |                 12 |
|  5 | c-fat200-2.clq     | 0min 0.0sec             |                    0.04 |                24 |                 24 |
|  6 | c-fat200-5.clq     | 0min 7.4sec             |                    7.38 |                58 |                 58 |
|  7 | c-fat500-1.clq     | 0min 0.4sec             |                    0.43 |                14 |                 14 |
|  8 | c-fat500-2.clq     | 0min 1.9sec             |                    1.88 |                26 |                 26 |
|  9 | c-fat500-5.clq     | 0min 1.0sec             |                    0.96 |                64 |                 64 |
| 10 | c-fat500-10.clq    | 0min 0.3sec             |                    0.26 |               126 |                126 |
| 11 | hamming6-2.clq     | 0min 0.0sec             |                    0.03 |                32 |                 32 |
| 12 | hamming6-4.clq     | 0min 1.6sec             |                    1.59 |                 4 |                  4 |
| 13 | hamming8-2.clq     | 0min 0.2sec             |                    0.24 |               128 |                128 |
| 14 | hamming8-4.clq     | 0min 0.2sec             |                    0.15 |                16 |                 16 |
| 15 | san200_0.7_1.clq   | 0min 0.1sec             |                    0.1  |                30 |                 30 |
| 16 | san200_0.7_2.clq   | 4min 1.2sec             |                  241.15 |                18 |                 18 |
| 17 | san200_0.9_1.clq   | 0min 1.0sec             |                    1.02 |                70 |                 70 |
| 18 | san200_0.9_2.clq   | 0min 4.1sec             |                    4.07 |                60 |                 60 |
| 19 | keller4.clq        | 3min 53.6sec            |                  233.59 |                11 |                 11 |
| 20 | MANN_a27.clq       | 5min 0.9sec             |                  300.9  |               126 |                126 |
| 21 | gen200_p0.9_55.clq | 0min 3.1sec             |                    3.11 |                55 |                 55 |
| 22 | gen200_p0.9_44.clq | TIMEOUT: >3600s elapsed |                  nan    |                39 |                 44 |
| 23 | C125.9.clq         | 32min 20.9sec           |                 1940.85 |                34 |                 34 |
| 24 | brock200_2.clq     | 10min 50.7sec           |                  650.74 |                12 |                 12 |
| 25 | brock200_3.clq     | 31min 54.1sec           |                 1914.09 |                15 |                 15 |
| 26 | brock200_4.clq     | TIMEOUT: >3601s elapsed |                  nan    |                17 |                 17 |
