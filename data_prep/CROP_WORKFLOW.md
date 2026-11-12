# 裁切、重放与背景方差

人工裁切可以在 Windows 的 VS Code Notebook 或 HPC 的 Jupyter 中完成。裁切阶段无需 GalSim。拟合阶段在 Linux/HPC 使用 GalSim；`scripts/main_fitting.py` 会从保存的 Astropy WCS 重建其 WCS。

## 首次确定裁切方案

1. 在仓库根目录打开 `data_prep/manual_crop.ipynb`。选择能读取五份光谱 FITS 的 Python 环境。安装 `numpy scipy astropy matplotlib ipympl ipywidgets`；Windows 环境无需安装 GalSim。
2. 填入 A1/A2/A3/B/C 的 FITS 路径、发射线名称、静止波长和红移。在第一个单元使用 `%matplotlib widget`。鼠标点选各组，或在 x/y 输入框中用键盘修改。单击按钮保存自动拟合结果或根据选点重新拟合后保存。
3. 每条线都保存到同一个 JSON 文件。Windows 与 HPC 若不共享文件系统，将该 JSON 文件复制到 HPC，并在运行时提供其绝对路径。输入 FITS 必须与首次确定方案时完全一致；程序会检查源光谱内容、形状和粗切波长列。

方案坐标使用 A 已叠加、A/C/B 已翻转但尚未粗切的完整光谱数组，以 NumPy 零基索引表示。方案还记录粗切的原始列、最终拟合线心、人工点击、整数边界和更新后的红移。O2 的 1D 拟合用实际波长步长（Å/pixel）计算双线间距。不同来源数据不能混用方案。

## HPC 重放并包装

从 `data_prep/` 目录运行现有脚本，指定方案文件：

```bash
python main.py --slitID 7 --crop_plan /absolute/path/slit_007_crops.json
```

Windows 本地也可以重放。在 `data_prep/` 目录使用当前目录中的方案，例如：

```powershell
python main.py --slitID 112 --crop_plan ./slit_112_crops.json --spec_width 40 --spec_height 40
```

上述宽高对应此方案的 40×40 切片；脚本默认高度为 20，若要保留 Notebook 中的尺寸，须显式指定。方案放在当前目录时必须传入 `--crop_plan`，否则脚本只查找输出目录。

如果不传 `--crop_plan`，默认查找输出目录的 `slit_007_crops.json`。包装 PKL 前，每条发射线都必须已在 Notebook 中审核并保存到方案文件；缺少文件或缺少任何一条线时直接报错。包装时只重放方案，不再拟合或交互。需要修改裁切时，重新运行 `manual_crop.ipynb` 并保存方案。若命令行指定新的裁切宽高，会围绕保存的最终线心重新计算固定尺寸边界。红移或波长与方案不符时会报错。

越界像素的 flux 和方差填零，`coverage_mask` 与 `mask_base` 标记无效位置；波长网格在边缘按局部步长外推供绘图使用。连续谱、背景方差和线宽仅在真实覆盖的矩形区域估计，然后映射到固定尺寸数组。最终 pkl 在原有字段之外保存这两个 mask。

## 只重新估计背景方差

从仓库根目录运行：

```bash
python -m data_prep.reestimate_variance input.pkl output.pkl
```

输入须为新流程产生、含有 `mask_base` 和 `coverage_mask` 的 pkl。命令只更新成像和光谱方差，以及依赖新方差的最终光谱 mask；不会重新裁切、重新拟合线心或重新计算连续谱。输出路径必须与输入不同。


## 光谱总方差（管线 err²）

`main.py` 同时读取各观测目录的 `obj_counts_slits_lin.fits` 和
`obj_counts_err_slits_lin.fits`。按 `SLITTARG` 定位目标，核对 split 光谱
（允许导出时的 −200 截断）与 counts 像素，以及 counts/err 坐标头。
缺少文件或对应不一致时直接报错，不退回经验背景估计。HPC 需保留这两个文件。

总方差为 `err**2`，与光谱一起翻转、按相同方案裁切。Set A 为三份观测的
算术平均，总方差为 `(err1**2 + err2**2 + err3**2) / 9`。未覆盖全部输入的
行、无效误差和管线误差下限 `err <= 1` 的像素不参与拟合。

新 pkl 保存 `var_total`。为兼容下游的 `data + var`，保存
`var = var_total - data`（data 已扣连续谱），使实际总方差恰为 `var_total`。
此时 `var` 是有效加性项，不能解释为独立测量的高斯背景方差，也不要求其
自身为正。补齐区域的 var 和 var_total 为零。

重新运行原有裁切重放命令即可生成新 pkl；旧文件不会自动改变。
`reestimate_variance` 对含 `var_total` 的新 pkl 保留管线总方差；对不含该字段
的旧 pkl 仍使用经验估计，不能单靠该命令将旧文件升级为管线方差。
