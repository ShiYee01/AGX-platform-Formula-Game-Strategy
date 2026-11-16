# AGX Racing local optimisation kit / 本地优化工具包

赛车策略保留 PPT 原版。
The racing policy is the original PPT example.

这是地图、最小策略和自建近似环境的示例。模型结果与正式对局有偏差。所有动力学/判定参数都要用本人的日志标定。
These are map parts, a minimal policy and examples for YOUR approximate environment. Model results differ from matches. Fit all dynamics/adjudication parameters using your own logs.

1. 下载 `agx-racing-kit.zip`，解压到新目录并进入该目录。Install Python 3.10+ and unzip into a new folder, then enter it.
2. 创建 Python 环境并安装 NumPy / create an environment and install NumPy:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy
```

可选：`python tools/read_map.py data/racing-parkland.mapdata.npz` 查看字段，`--output restored.json` 还原完整 JSON。Optional: inspect all fields or restore JSON using the reader.

3. 在 AGX 的本人已结束对局页下载本人席位的 `training.jsonl`，放到此目录；不把日志、账号信息或下载凭据装进提交 ZIP。In your completed match, download YOUR seat's training data to this directory. Keep logs and credentials out of the submission.
4. 标定 / calibrate:

```sh
python calibrate/fit.py training.jsonl --game racing --output fitted.json
```

输出是有效观测包络，不是确定的物理规律。按赛道、天气与轮胎条件分别采样；留出后续对局验证。空值表示缺对应动作/运动样本，需补采样或填入自己估计的正数。手动在 `fitted.json` 的 `parameters` 里补入 `safety_margin_m`（你选择的几何边距，米 / chosen geometric margin in metres），并记录你的依据。FPS 的反应时间和赛车的安全边距不由这个简单拟合器识别。
Fits are effective observed envelopes, not identified physics. Separate tracks, weather and tyres and keep held-out matches. Null means missing samples: collect suitable actions or add your own positive estimate. Add `safety_margin_m`（你选择的几何边距，米 / chosen geometric margin in metres） to `parameters` in `fitted.json`, documenting your choice. This fitter does not identify reaction time or safety margins.

5. 运行本地近似环境 / run your local approximation:

```sh
python sim/optimize.py data/racing-parkland.mapdata.npz --params fitted.json --output profile.npz
```

FPS：搜索双方路线/站位，可改成简化自博弈。赛车：搜索行车线/速度表，输出近似时间和能耗。优化完把选中的行为写回 `agent/agent.py`；赛车可将 `profile.npz` 放到 `data/`，从 `weights/profile.npz` 加载速度表。`sim/` 输出不会自动改变策略或平台分数。
FPS searches both sides' routes/positions and can be extended for simplified self-play. Racing searches lines/speeds and estimates time/energy. Transfer chosen behaviour into `agent/agent.py`; racing can copy `profile.npz` to `data/` and load the table from `weights/profile.npz`. Simulator output does not automatically modify the policy or a match score.

6. 打包最小策略和所有地图 / pack the agent and maps:

```sh
python tools/pack.py --output submission.zip
```

`tools/pack.py` 只收集 `agent/*.py`、根设计书 `agent/design.json` 和 `data/*.npz`；写入 ZIP 根目录 `agent.py`、赛车 `design.json`、`weights/*.mapdata.npz`。不会放入 JSON 地图、日志、文档、环境代码或目录条目。ZIP ≤64 MiB、≤64 条目，每份源文件 ≤256 KiB、源文件合计 ≤1 MiB、NPZ 和展开张量各合计 ≤64 MiB。打包器检查常用约束，最终仍以平台准入为准。
The packer collects only `agent/*.py`, `agent/design.json` and `data/*.npz`. It writes root `agent.py`, racing `design.json`, and `weights/*.mapdata.npz`, with no explicit directories. ZIP ≤64 MiB/64 entries; each source ≤256 KiB, combined source ≤1 MiB; NPZ and tensor bytes each ≤64 MiB. It checks common limits; platform admission remains authoritative.

7. 在对应比赛的 Agent 提交页上传 `submission.zip`、发布版本并更新报名版本。看后续本人对局日志，重新标定/优化。
Upload `submission.zip` on the contest Agent submission page, publish the version and update the enrolled version. Repeat calibration using your later match logs.

`init(context)` 从 `context['weights_dir']` 给出的提交根目录定位 `weights/`，用 `np.load(..., allow_pickle=False)`；`act(observation)` 使用已加载数据。所附策略保持上述母版；地图加载与优化结果由学生自行接入。不要在策略里导入 `sim` 或 `calibrate`。
`init(context)` locates `weights/` under the submitted root given by `context['weights_dir']` and uses `allow_pickle=False`; `act(observation)` uses the loaded data. The supplied policy preserves the template above; connect map loading and optimisation results yourself. Do not import `sim` or `calibrate` from the submitted policy.

## 地图数据能做什么：赛车

用 `centerline` 的中心线、左右半宽和 `curvature_1pm` 离线优化行车线、速度曲线、弯道入弯限速和刹车点。质点近似下 `v_corner = sqrt(a_lat/abs(curvature))`，制动前向约束用 `v² <= v_corner² + 2*a_brake*distance`；这里的 `a_lat`、制动、加速、速度上限和能量消耗率都需要由你用本人 `training.jsonl` 标定，不是平台模型参数。

`sim/optimize.py` 在赛道宽度内搜索九条平滑横向偏移线，重新计算曲率，再循环传播加速/制动约束，给出近似单圈时间和能耗。安全边距由你选择并用日志检查，不能当成正式越界阈值。最优结果写到 `profile.npz`；读懂后将速度表或横向目标接入自己的 `agent.py`。示例 a 已演示在 `init(context)` 加载地图，在 `act()` 使用按曲率估计的入弯速度。

`pit.entry_s/exit_s`、车道、维修格和限速用于估计到站距离、进站窗口与路线；结合本人圈速、轮胎变化、油量和公开赛制要求比较早停/晚停。`design_limits` 提供每条赛道的轮胎配置、胎压和外倾约束、回收限制；检查 `effective_pressure_floor_psi`，并同时遵守设计书的其他公开规则。扇区/路肩的 authored station 标记与 `length_m` 可能不同；观察 `s` 使用 `length_m` 环绕，别把官方长度直接当作该坐标的周期。

NPZ 与 JSON 一一对应，另有 `centerline` 矩阵、UTF-8 字节数组 `centerline_columns_utf8` 和 `track_length_m`。文字和完整结构存于 `structure_utf8`，数值存于 `i/f/b` 池。只用 `allow_pickle=False` 加载。提交包用 `weights/*.mapdata.npz`，普通 JSON 留在本地。

这是你自己的近似质点环境，忽略真实车辆的许多作用，局部最优不等于比赛最优。用本人日志按赛道、天气和轮胎条件分别校准并留出验证对局，在本地优化完再提交；平台不会提供比赛引擎供本地运行。

## What map data can do: racing

Use `centerline`, left/right half-widths and `curvature_1pm` to optimise a racing line, speed profile, corner entry speeds and braking points offline. A particle approximation uses `v_corner = sqrt(a_lat/abs(curvature))` and the braking approach constraint `v² <= v_corner² + 2*a_brake*distance`. Fit lateral acceleration, braking, acceleration, speed cap and energy use with YOUR `training.jsonl`; they are your model parameters.

`sim/optimize.py` searches nine smooth lateral offset lines within the road width, recomputes curvature and propagates circular acceleration/braking constraints. It reports estimated lap time and energy use. Choose and validate your own safety margin; it is not a match penalty threshold. Results are written to `profile.npz`; adapt the speed table or lateral targets into your own `agent.py`. The supplied policy is the original PPT example; connect your map and speed profile in `init(context)`/`act()` yourself.

Use `pit.entry_s/exit_s`, lane, boxes and speed limit to estimate approach distance, pit windows and routes. Compare early/late stops using your own lap times, tyre/fuel changes and the published sporting rules. `design_limits` gives track tyre allocation, pressure/camber constraints and recharge limits. Check `effective_pressure_floor_psi` together with the other public design rules. Authored sector/kerb station labels can differ from `length_m`; observation `s` wraps at `length_m`, so do not substitute the official circuit length as its coordinate period.

NPZ represents the complete JSON, plus a `centerline` matrix, UTF-8 `centerline_columns_utf8` bytes and `track_length_m`. Names/structure use `structure_utf8`; values use `i/f/b` pools. Load with `allow_pickle=False`. Submit `weights/*.mapdata.npz`; keep ordinary JSON locally.

Your particle environment is approximate and omits many vehicle effects. A local optimum is not a match optimum. Calibrate separately by track, weather and tyres, retain held-out matches, optimise locally and submit. The platform does not provide its match engine for local execution.

