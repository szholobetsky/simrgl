## dev (200 tasks of exp7, the rule never saw them): 199 tasks

| | P@1 | dir@1 | MRR@5 | hit@5 |
|---|---|---|---|---|
| **compass** | 0.276 | 0.322 | 0.344 | 0.467 |
| ontology alone (gamayun search) | 0.302 | 0.337 | 0.357 | 0.467 |

**By source** (what the compass used)

| source | tasks | share | P@1 compass | P@1 ontology alone on the same tasks | hit@5 compass |
|---|---|---|---|---|---|
| anchor | 39 | 20% | 0.231 | 0.359 | 0.513 |
| ontology | 160 | 80% | 0.287 | 0.287 | 0.456 |

**Calibration** (is "high" really right more often?)

| confidence | tasks | share | P@1 | dir@1 | hit@5 |
|---|---|---|---|---|---|
| high | 21 | 11% | 0.381 | 0.381 | 0.524 |
| medium | 157 | 79% | 0.299 | 0.357 | 0.503 |
| low | 21 | 11% | 0.000 | 0.000 | 0.143 |

mean time per task: 0.6 s (git grep on a revision + ontology)

## test of exp7 without T01..T10: 190 tasks

| | P@1 | dir@1 | MRR@5 | hit@5 |
|---|---|---|---|---|
| **compass** | 0.237 | 0.316 | 0.310 | 0.421 |
| ontology alone (gamayun search) | 0.200 | 0.258 | 0.251 | 0.337 |

**By source** (what the compass used)

| source | tasks | share | P@1 compass | P@1 ontology alone on the same tasks | hit@5 compass |
|---|---|---|---|---|---|
| anchor | 68 | 36% | 0.265 | 0.162 | 0.485 |
| ontology | 122 | 64% | 0.221 | 0.221 | 0.385 |

**Calibration** (is "high" really right more often?)

| confidence | tasks | share | P@1 | dir@1 | hit@5 |
|---|---|---|---|---|---|
| high | 35 | 18% | 0.257 | 0.429 | 0.400 |
| medium | 138 | 73% | 0.261 | 0.326 | 0.471 |
| low | 17 | 9% | 0.000 | 0.000 | 0.059 |

mean time per task: 1.2 s (git grep on a revision + ontology)

## T01..T10 (the rule was set on these): 10 tasks

| | P@1 | dir@1 | MRR@5 | hit@5 |
|---|---|---|---|---|
| **compass** | 0.700 | 0.700 | 0.750 | 0.800 |
| ontology alone (gamayun search) | 0.500 | 0.500 | 0.550 | 0.600 |

**By source** (what the compass used)

| source | tasks | share | P@1 compass | P@1 ontology alone on the same tasks | hit@5 compass |
|---|---|---|---|---|---|
| anchor | 5 | 50% | 0.800 | 0.400 | 1.000 |
| ontology | 5 | 50% | 0.600 | 0.600 | 0.600 |

**Calibration** (is "high" really right more often?)

| confidence | tasks | share | P@1 | dir@1 | hit@5 |
|---|---|---|---|---|---|
| high | 4 | 40% | 1.000 | 1.000 | 1.000 |
| medium | 6 | 60% | 0.500 | 0.500 | 0.667 |

mean time per task: 1.5 s (git grep on a revision + ontology)

