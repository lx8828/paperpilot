# 查询改写 A/B（新默认 nol3j+gate，100 题）

> 题 100 | 裁判 glm-4-flash ≥4=pass

| 分组 | off pass | on pass | Δ | off calls | on calls | off prompt | on prompt |
|---|---|---|---|---|---|---|---|
| all | 71/100 = 71% | 68/100 = 68% | -3 | 3.0 | 3.9 | 9791 | 9987 |
| hard | 27/50 = 54% | 27/50 = 54% | +0 | 3.1 | 4.0 | 9935 | 10186 |
| normal | 44/50 = 88% | 41/50 = 82% | -3 | 2.9 | 3.8 | 9646 | 9789 |

> 历史 oldV（nol3j 73% 那批）同题 pass：64/100 = 64%

> 改写：救回 3 题 / 弄坏 6 题

### ON 救回
  05671d0686 [hard] off=3 on=4
  9555aa8de3 [hard] off=3 on=4
  828ce5faed [normal] off=3 on=4

### ON 弄坏
  3241f90a03 [hard] off=4 on=3
  fb381a5973 [hard] off=5 on=3
  345f65eaff [normal] off=4 on=3
  66f0dee89f [normal] off=5 on=3
  55fb92afa1 [normal] off=4 on=3
  9213159f87 [normal] off=4 on=3
