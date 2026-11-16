---
name: agx-racing-local-optimisation
description: Calibrate your AGX racing approximation from your own match logs, optimise locally and pack a submission.
---

# 操作步骤 / Steps

1. 先读 README.md 和公开规则，确认 `agent/agent.py` 的 init/act 接口。Read README.md and public rules; preserve the init/act interface.
2. 请学生从本人已结束对局下载本人席位 training.jsonl；不索取账号凭据。Ask the student to download their own seat log from a completed match; never request account credentials.
3. 运行 `calibrate/fit.py training.jsonl --game racing --output fitted.json`。检查样本数与 null，分组赛道/天气/轮胎，保留验证集；补齐自己的参数估计并记录依据。Run calibration, inspect counts/nulls, stratify conditions, retain held-out logs and document student estimates.
4. 按 README 在 sim 里优化路线/站位或行车线/速度。所有动力学和判定参数都用本人日志标定；不将示例系数描述成平台真实值。Run the local model; fit dynamics/adjudication parameters from the student's logs. Example coefficients are not platform constants.
5. 将选中行为接入 agent/agent.py；只依赖提交的 Python、NPZ、标准库和可用 NumPy。赛车保留合法 design.json。Transfer selected behaviour into agent/agent.py; preserve a legal racing design.json.
6. 运行 `python tools/pack.py --output submission.zip`；核对只含 agent.py、辅助 .py、赛车 design.json 和 weights/*.mapdata.npz，且 allow_pickle=False 可读。不包含 JSON 地图、日志、本文或 sim/calibrate。Pack and inspect names, limits and numeric NPZ loading.
7. 由学生在 Agent 提交页上传、发布并更新报名版本。之后用新日志复查偏差。The student uploads, publishes and updates enrollment; validate assumptions with later match logs.

模型指标只用于本地候选比较，不冒充正式分数。Do not report local model objectives as platform scores.
