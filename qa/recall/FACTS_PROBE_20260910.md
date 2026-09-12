# 两步法第一步失效规模（_extract_facts）

> 样本 40 题（ab_rewrite off 臂前 N 题）｜每题 1 次 LLM

| qid | 组 | ctx字符 | parsed facts | raw字符 | 判定 | 备注 |

## 汇总

- 样本 40 题
- **EMPTY（模型返回空清单）**：6 = 15%
- **PARSE（提取到了但 JSON 解析失败 → 被静默丢弃）**：1 = 2%
- OK：33 = 82%
- ERR（调用异常）：0

> 判读：`PARSE` = **信息明明提取到了却被丢掉**（真 bug，需宽松解析兜底）；
> `EMPTY` = 模型判定该上下文无相关事实（可能是检索没送到 / 题目本身无答案）。
|---|---|---|---|---|---|---|
| dd9883f4 | hard | 12304 | 18 | 3467 | OK | ok facts=20 |
| a3f108f6 | hard | 14223 | 7 | 1571 | OK | ok facts=10 |
| 8df89988 | hard | 17144 | 0 | 13 | EMPTY | {"facts": []} |
| d28260b5 | hard | 12578 | 0 | 13 | EMPTY | {"facts": []} |
| be595b20 | hard | 24599 | 17 | 18757 | OK | ok facts=99 |
| de12e059 | hard | 22489 | 0 | 13 | EMPTY | {"facts": []} |
| c30c3e0f | hard | 17958 | 22 | 4047 | OK | ok facts=21 |
| 8051927f | hard | 20294 | 16 | 1410 | OK | ok facts=12 |
| cd179292 | hard | 15630 | 0 | 13 | EMPTY | {"facts": []} |
| 05671d06 | hard | 21771 | 21 | 3690 | OK | ok facts=21 |
| 13b36644 | hard | 20346 | 5 | 1024 | OK | ok facts=5 |
| 9555aa8d | hard | 21871 | 10 | 1180 | OK | ok facts=8 |
| 1d74fd1d | hard | 14075 | 8 | 714 | OK | ok facts=6 |
| af75ad21 | hard | 22515 | 0 | 13 | EMPTY | {"facts": []} |
| 544b68f6 | hard | 11900 | 12 | 2550 | OK | ok facts=19 |
| 86cd1228 | hard | 17802 | 17 | 3771 | OK | ok facts=22 |
| 45893f31 | hard | 19056 | 14 | 3056 | OK | ok facts=15 |
| 0b9021ce | hard | 33981 | 24 | 2839 | OK | ok facts=24 |
| 5a23f436 | hard | 13716 | 5 | 1005 | OK | ok facts=5 |
| 81064bbd | hard | 18367 | 41 | 1894 | OK | ok facts=13 |
| 6e962f1f | hard | 18014 | 13 | 2306 | OK | ok facts=11 |
| bf25a202 | hard | 28630 | 20 | 4114 | OK | ok facts=24 |
| d6e2b276 | hard | 25094 | 11 | 2411 | OK | ok facts=12 |
| aa800b42 | hard | 27580 | 5 | 888 | OK | ok facts=5 |
| 79bb1a1b | hard | 22009 | 21 | 3850 | OK | ok facts=26 |
| 3241f90a | hard | 24326 | 6 | 898 | OK | ok facts=5 |
| f319f2c3 | hard | 15459 | 10 | 1378 | OK | ok facts=10 |
| 3de27c81 | hard | 14487 | 0 | 2569 | PARSE | logies) and 13 to the “syntactic" one (10876 analogies)."}]} |
| 4b0ba460 | hard | 15966 | 18 | 4584 | OK | ok facts=27 |
| 7955dbd7 | hard | 30456 | 30 | 1779 | OK | ok facts=13 |
| b42323d6 | hard | 16263 | 8 | 768 | OK | ok facts=8 |
| fb381a59 | hard | 21021 | 20 | 3353 | OK | ok facts=19 |
| e414d819 | hard | 7851 | 1 | 89 | OK | ok facts=1 |
| 83f14af3 | hard | 14329 | 8 | 1234 | OK | ok facts=8 |
| d427e3d4 | hard | 20121 | 18 | 2793 | OK | ok facts=18 |
| 5e9732ff | hard | 15815 | 11 | 1694 | OK | ok facts=11 |
| 5a02a3dd | hard | 13340 | 49 | 10764 | OK | ok facts=53 |
| 3d34a02c | hard | 24887 | 13 | 2292 | OK | ok facts=17 |
| b13d0e46 | hard | 306 | 0 | 13 | EMPTY | {"facts": []} |
| c33d0bc5 | hard | 23592 | 20 | 3186 | OK | ok facts=22 |
