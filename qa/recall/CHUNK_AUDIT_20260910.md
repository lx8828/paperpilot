# 切块质量审计（base 检索，官方口径，250 题）

> chunk 总数 3247 | 超大块(>2500字符) 712 (22%)

### A. chunk 尺寸分布（全部 recall_set 论文）
- 字符数：中位 1292 | p75 2300 | p90 3449 | max 15981
- 段落数：中位 2 | p75 4 | p90 7 | max 125

### B. 按 gold 所在 chunk 的字符数分组
| 分组 | n | MRR | NDCG@12 | hit@12 | gold@1 |
|---|---|---|---|---|---|
| gold块 ≤1000字符 | 41 | 0.349 | 0.485 | 0.927 | 0.122 |
| gold块 1001-2500 | 111 | 0.489 | 0.597 | 0.955 | 0.315 |
| gold块 2501-4000 | 98 | 0.470 | 0.589 | 0.980 | 0.265 |

### C. 按 gold 所在 chunk 的段落数分组
| 分组 | n | MRR | NDCG@12 | hit@12 | gold@1 |
|---|---|---|---|---|---|
| gold块=1段 | 37 | 0.375 | 0.508 | 0.946 | 0.162 |
| gold块=2段 | 42 | 0.482 | 0.601 | 1.000 | 0.333 |
| gold块=3-4段 | 83 | 0.462 | 0.575 | 0.940 | 0.265 |
| gold块≥5段 | 88 | 0.479 | 0.593 | 0.966 | 0.273 |

### D. gold 落在超大块的样例（rank 靠后）
- [hard][rank#13] Which loss metrics do they try in their new training procedure evaluat
      sec=Continuous Approximation to top-k-argmax len=3246 blocks=6 part=None
      [t] Continuous relaxation to beam search [1] INLINEFORM0 , INLINEFORM1 , INLINEFORM2 , INLINEFORM3 , INLINEFORM4 t = 0 to T INLINEFORM5 i=1 to k INLINEFORM6 INLINEFORM7 is a local output scoring function INLINEFORM8 INLI
- [hard][rank#4] How does this compare to contextual embedding methods?
      sec=Introduction len=3346 blocks=5 part=None
      To model language, we must represent words. We can imagine representing every word with a binary one-hot vector corresponding to a dictionary position. But such a representation contains no valuable semantic information:
- [hard][rank#4] How are discourse features incorporated into the model?
      sec=Models len=3021 blocks=5 part=None
      Building on shrestha2017's work, we employ their character-bigram CNN (CNN2), and propose two extensions which utilize discourse information: (i) CNN2 enhanced with relation probability vectors (CNN2-PV), and (ii) CNN2 e
- [hard][rank#11] What are the state of the art measures?
      sec=Results len=3164 blocks=4 part=1/2
      Training a Fasttext model is not a deterministic process, as different runs could yield different results even using the same training set in each one. To analyze if these differences are significant, we decide to comput
- [hard][rank#6] Why is this work different from text-only UNMT?
      sec=Introduction len=3744 blocks=5 part=1/2
      Our long-term goal is to build intelligent systems that can perceive their visual environment and understand the linguistic information, and further make an accurate translation inference to another language. Since image

> gold块字符数 × gold rank 相关系数：-0.097 （正=块越大排越靠后）
