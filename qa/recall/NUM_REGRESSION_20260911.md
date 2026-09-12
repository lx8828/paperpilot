# 数字归一化回归集结果（零 LLM）

> 真实误杀案例（QASPER 233 闸门诊断）+ 边界写法；改 `numbers.py`/`validator.py` 后必跑

## A. 数值归一化（validator 实际用的 `_canonical_nums`）
- [PASS] 右词界·most → `[15000.0]`
- [PASS] 右词界·kernels → `[2.7]`
- [PASS] 右词界·between → `[0.81]`
- [PASS] 右词界·words → `[500.0]`
- [PASS] 左词界·F1百分点 → `[]`
- [PASS] 左词界·百分整词 → `[]`
- [PASS] 左词界·v前缀仍提取 → `[2.3]`
- [PASS] LaTeX·星号幂 → `[7.3e+115]`
- [PASS] LaTeX·times幂 → `[850000000000.0]`
- [PASS] LaTeX·转义百分号 → `[29.0, 54.0]`
- [PASS] 年份·数据集名保留 → `[2000.0]`
- [PASS] 年份·中文日期剥离 → `[]`
- [PASS] 年份·英文语境剥离 → `[]`
- [PASS] 千分位 → `[1000000.0]`
- [PASS] 欧式小数 → `[7.5]`
- [PASS] 英文单位 → `[36000000.0, 1500000000.0]`
- [PASS] 中文网络单位 → `[36000.0]`
- [PASS] 中文数词 → `[36000]`
- [PASS] 日常百分号 → `[12.5]`

## B. 豁免② 派生数（四则运算 + 四舍五入）
- [PASS] 差量 2000−300=1700 → got=True want=True
- [PASS] 倍数 2000÷300≈6.67 → got=True want=True
- [PASS] 除法+四舍五入 293÷4262≈0.069 → got=True want=True
- [PASS] 舍入 0.0687→0.07 → got=True want=True
- [PASS] 无关数字不该豁免 → got=False want=False

## C. 年份剥离（上下文感知）
- [PASS] 数据集名保留: `Switchboard-2000 hours` → `Switchboard-2000 hours`
- [PASS] 数值保留: `2000 more hours` → `2000 more hours`
- [PASS] 中文日期剥离: `2019年发表` → `    年发表`
- [PASS] 英文语境剥离: `in 2019 the model` → `in      the model`

---

**结果：全部通过 ✅**
