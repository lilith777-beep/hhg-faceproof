# Latency report

- queries: **24**  ·  budget: **200 ms** (retrieval→output, STT excluded)
- **P50 2.49 ms · P70 4.45 ms · P100 18.61 ms**
- decisions: {'answer': 20, 'abstain_ood': 4}

| stage | P50 | P70 | P100 |
|---|---|---|---|
| safety | 0.33 | 0.35 | 0.57 |
| normalize | 0.44 | 0.49 | 0.82 |
| route | 0.33 | 0.34 | 0.46 |
| embed | 0.44 | 0.46 | 0.92 |
| cache | 0.57 | 0.62 | 0.98 |
| retrieve | 6.07 | 6.84 | 8.25 |
| rerank | 4.36 | 4.81 | 6.44 |
| generate | 0.97 | 1.26 | 1.69 |
| ground_check | 0.5 | 0.54 | 0.61 |
| **TOTAL** | **2.49** | **4.45** | **18.61** |
