# `evals/` · 评测与回归护栏

本目录**不是"从零新建一套评测"** —— 评测能力早已存在，但它散落在三处
（`cli/eval/`、`retrieval/scripts/`、`retrieval/tmp/`），且**真值 / 判据 / 报告各归谁**
不清楚。`evals/` 的职责是：

1. **给每层评测一个明确的家**（入口 + 真值 + 报告，三者归属写清楚）；
2. **一条命令一条层**（而不是"47 个脚本自己猜该跑哪个"）；
3. **把"干净 clone 能不能评测"变成常态检查**（L0）。

> 历史背景见 `DEVELOPMENT_LOG.md` 与 `retrieval/data/r2dev/README.md`。
> 2026-10-05 的一次普查发现：**真值、报告、迁移、导出**都出现过"只在工作区、没进版本库"
> 的情况 —— 那类失效**不报错**，只是让别人 clone 后跑不起来或拿到错的数。L0 就是为此。

---

## 分层（每层一个触发条件，不混）

| 层 | 判什么 | 触发 | 成本 | 现状 |
|---|---|---|---|---|
| **L0 · 不变量** | 仓库状态与契约（**不跑模型**）：干净 clone 能不能评测、路径引用是否悬空、gold 与导出是否同步 | **每次改动**（CI 默认） | 秒级 / 免费 | ✅ **已有** |
| **L1 · 篇级检索** | LitSearch（64,183 篇）：Recall@k / nDCG | 发版前 / 手动 | 分钟级 | 🟡 能力已有，入口在 `cli/eval/run_retrieval_eval.py` |
| **L2 · 判官（论文线）** | r2dev 28 个 facet：证据召回 / reader 判定 | 发版前 / 手动 | 分钟~小时 | 🟡 能力已有，脚本在 `retrieval/tmp/_r2_*.py` |
| **L3 · 端到端** | 生产全链路（`graph.ask`）+ 5 组 164 题 | 发版前 / 手动 | 小时级 | 🟡 能力已有：`cli/eval/run_group_qa.py` |
| **L4 · 对外可比** | QAMPARI / LoFT 官方口径 | **按需** | 小时级（要重跑） | 🟡 数据在 `retrieval/data/{qampari,loft}`（未入库，可重下） |

**为什么这么分层**：区分「**免费且必须每次跑**」（L0）与「**贵且只需发版前跑**」（L1–L3），
否则一套评测要么贵到没人跑、要么便宜到抓不住问题。L4 单独一档是因为它的价值在
**对外可比**（跟公开榜同口径），与内部回归的触发节奏不同。

## 与三个目标的对齐

| 目标 | 主要落在 |
|---|---|
| **回归防护**（改动别把已有能力改坏） | **L0**（免费、每次）+ L1/L3 作为发版闸 |
| **对外可比**（数字能跟公开工作比） | **L4**（QAMPARI / LoFT 官方口径） |
| **论文线指标**（这套系统的真本事） | **L1 + L2 + L3** |

---

## 目录约定

```
evals/
  README.md                  本文件：分层、触发、现状、归属
  baselines/                 ① 基线（棘轮）：只进不退
    path_liveness.json         已知悬空引用（66 条，历史债）
    tmp_scripts.json           retrieval/tmp 脚本分类（库/入口/归档判定）
    metrics.json               **指标基线**（值 + 容差 + 方向）；空则填 `{}`
  checks/                    ② L0 不变量（只读工具）
    path_liveness.py           路径存活棘轮（--update 收缩基线）
    tmp_scripts_audit.py       retrieval/tmp 脚本分类器（analyze() 纯分析）
    metrics_ratchet.py         **指标棘轮**：掉出容差就报错（--update 建/更新基线）
  report.py                  ③ **统一记录格式** —— 让各层数字能放一张表里比
  reports/                     落盘（**gitignore**：可再生成；入库的是 .md 摘要）
  (待建)
  datasets/                  题集与真值（**只放指针**，实体仍在原处）
  runners/                   一层一个入口（**L1/L2/L3 均已接 `report.emit`**）
```

### 三条硬约定

1. **真值只放一份。** `evals/datasets/` 只写**指针**（指向 `retrieval/tmp/group*/`
   与 `retrieval/data/r2dev/`），**不复制** —— 复制必然漂移。真值的权威说明见
   `retrieval/data/r2dev/README.md`（哪份 gold 是现役）。
2. **判据只放一处。** 例如"gold ↔ 导出同步"只由 `retrieval/scripts/_check_export_sync.py`
   定义，`evals/` 与 `tests/` 都**调用**它，不重写。（重写 = 漂移 = 假测试。）
3. **报告入库，产物不入库。** `retrieval/results/*.md` 入库（现 85 份，是不可再生的记录）；
   跑批产出（`results/*/`、`qa/multi/_runs/`）不入库。

---

## L0 现状（已可跑）

| 检查 | 在哪 | 断言什么 |
|---|---|---|
| gold 是否全部入库 | `tests/test_eval_assets.py` | 磁盘上**每个** gold 都已在 git（曾 15 个只入库 7 个） |
| r2dev 真值是否入库 | 同上 | 现役真值 + 题集 + 子查询在库；大件/可再生件**不**在库 |
| gold ↔ 导出是否同步 | 同上 + `_check_export_sync.py` | 不同步 = 跑批**静默测另一份题** |
| 悬空路径引用 | `evals/checks/path_liveness.py` + `tests/test_path_liveness.py` | 无**新增**悬空引用（棘轮，66 条历史债不阻塞） |
| `retrieval/tmp` 脚本归属 | `evals/checks/tmp_scripts_audit.py` + `tests/test_tmp_scripts_audit.py` | 保留脚本**已入库**；归档**不造断 import** |
| 统一记录格式 | `evals/report.py` + `tests/test_evals_report.py` | 缺 `n`/`note` 的记录**写不进去**；NaN/inf 被拒；表格列宽自适应且**对齐** |
| **指标棘轮** | `evals/checks/metrics_ratchet.py` + `tests/test_metrics_ratchet.py` | 跑批后指标**掉出容差**就报错；★ 带容差（M1 ±16pt）与**方向**（健康指标越低越好） |
| 草稿/备份是否混在 gold 旁 | `tests/test_eval_assets.py` | 无（同前缀会让人改错文件） |
| CLI 入口能否跑 | 同上（`@pytest.mark.local`） | `cli/eval/*.py --help` 退出码 0（验证 `parents[2]`） |

跑法：`uv run pytest -q`（L0 全在默认套件里，无需 key / GPU / 网络）。

---

## 下一步（按价值排序）

1. **L2/L3 脚本的归属 —— 已定，不搬**（2026-10-05 实测结论）。
   `retrieval/tmp/_*.py`（135 个：库 14 + 有出处的入口 ~120）**原地入库**，
   归属说明见 `retrieval/tmp/README.md`；一次性实验 161 个已归档
   （判定在 `evals/baselines/tmp_scripts.json`）。

   **为什么不能"搬去 `evals/`"**：实测 **133/135 个脚本用 `parents[N]` 定位**、
   88 个改过 `sys.path`、50 个用 `spec_from_file_location` 动态加载兄弟脚本，
   而它们**按「相对 `retrieval/`」算路径**（`HERE / "scripts"` → `retrieval/scripts`）。
   换目录会让这些**语义变化**，不是换名能修的 → 得逐处判断。
   **要拆 `lib/` + `runners/`，前提是先改成包导入**（`import evals.lib.x`），
   那是独立的一次重构，不该混在"归档清理"里做。

2. **L1 / L2 / L3 已全部接上 `report.emit`**（2026-10-05 完成）。

   | 层 | 文件 | 落的指标 |
   |---|---|---|
   | L1 | `cli/eval/run_retrieval_eval.py` | `retrieval.recall@k` / `mrr@k` + 健康度 `gold_map_failed` |
   | L2 | `retrieval/tmp/_r2_retr_eval.py` | `r2.mrecall@10` / `strecall@10` / `setf1@10` / `andcg@10` / `ev_recall_c12k` |
   | L2 | `retrieval/tmp/_r2_reader.py` | `r2.reader_p@10` / `reader_r@10` / `reader_f1@10` / `ret_f1@10`（基线对照） |
   | L3 | `cli/eval/run_group_qa.py` | `qa.ok*` / `qa.kind_*` + 三条健康度 |

   ⚠️ 接的时候**没动各自的判分口径**（那是另一件事）。

3. **指标棘轮已就位**：`evals/baselines/metrics.json` 初始 `{}`，
   **跑过一次批后执行一次**即可建立基线：

       python evals/checks/metrics_ratchet.py --update   # 建/更新基线
       python evals/checks/metrics_ratchet.py            # 比对（掉出容差 → 退出码 1）

   ★ 两条必须记住的：**容差**（M1 ±16pt，硬比天天误报）与**方向**
   （`reader_offlabel` / `gold_map_failed` 这些**越低越好**，一律"降了就红"
   会把**修好了**判成回退）。

4. **L4 重跑能力**：QAMPARI / LoFT 目前只有历史记录；要做成"一条命令重跑并出官方口径数"。

5. **`evals/reports/` 是 gitignore 的** → 干净 clone 里指标棘轮**无数据可比**（会跳过并说明）。
   若要让"指标历史"进版本库，应入库 `--md` 摘要（`python evals/report.py --md`），
   而不是每次跑的原始 JSONL。
