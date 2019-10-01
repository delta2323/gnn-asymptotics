Code for experiments in Section 6.1

Usage
```
bash -x run.sh
```

This command creates the following files:

```
4/
├── 0.5
│   ├── log.txt
│   └── streamplot.pdf
├── 1.0
│   ├── log.txt
│   └── streamplot.pdf
├── 1.2
│   ├── log.txt
│   └── streamplot.pdf
├── 1.5
│   ├── log.txt
│   └── streamplot.pdf
├── 2.0
│   ├── log.txt
│   └── streamplot.pdf
└── 4.0
    ├── log.txt
    └── streamplot.pdf
15/
├── 0.5
│   ├── log.txt
│   └── streamplot.pdf
├── 1.0
│   ├── log.txt
│   └── streamplot.pdf
├── 1.2
│   ├── log.txt
│   └── streamplot.pdf
├── 1.5
│   ├── log.txt
│   └── streamplot.pdf
├── 2.0
│   ├── log.txt
│   └── streamplot.pdf
└── 4.0
    ├── log.txt
    └── streamplot.pdf
```

The naming convention of the directory is `<seed>/<weight_value>/`. This directory contains the result when the seed is `<seed>` and the weight value `W` is `<weight_value>`. Seed=4 corresponds to Case 1 and Seed=15 corresponds to Case 2 in the paper, respectively.
