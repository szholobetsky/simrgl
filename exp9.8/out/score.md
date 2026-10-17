| | P@1 | MRR | MAP | recall@10 | recall of files that existed | hit@10 | needles found | tool calls | output chars |
|---|---|---|---|---|---|---|---|---|---|
| gamayun alone (no model) | 0.60 | 0.60 | 0.106 | 0.13 | 0.16 | 0.60 | 0/5 | - | - |
| G: grep + read | 0.80 | 0.80 | 0.217 | 0.25 | 0.30 | 0.80 | 1/5 | 37 | 146992 |
| P: indexes + read | 0.80 | 0.80 | 0.255 | 0.30 | 0.37 | 0.80 | 3/5 | 32 | 84577 |

right files in the top 10, and the needle (n = found)

| | D1 SONAR-8777 (9) | D2 SONAR-11124 (10) | D3 SONAR-9253 (19) | D4 SONAR-17735 (9) | D5 SONAR-1915 (12) |
|---|---|---|---|---|---|
| gamayun alone (no model) | 0 | 0 | 1 | 4 | 2 |
| G: grep + read | 0 | 3 n | 2 | 6 | 2 |
| P: indexes + read | 0 | 5 n | 3 | 6 n | 2 n |

commands used

- G: grep 30, read 7
- P: read 8, simargl 5, find 5, companions 5, defines 4, compass 3, gamayun 2
