# PowerShell 批处理与日志

## 任务与当前进展
- 在 PowerShell 7 中批量运行 `data_prep/main.py`，slitID 原计划 1..142；Python 返回非零退出码时继续下一项，最终保存失败号码。
- 本会话未修改项目代码。用户最后分享下列 6..10 批处理片段，仅要求阅读并备份，没有要求继续排错。

## 用户最后分享的代码
```powershell
foreach ($i in 6..10) {
    $cropPlan = "slit_{0:D3}_crops.json" -f $i
    python main.py --slitID $i --crop_plan $cropPlan

    if ($LASTEXITCODE -ne 0) {
        $failedSlits += $i
        Write-Warning "slitID=$i 执行失败，跳过"
        continue
    }
}

# 每行一个失败号码；没有失败时写入空文件
Set-Content -Path "failed_slits.txt" -Value $failedSlits -Encoding utf8

Write-Host "运行结束，共失败 $($failedSlits.Count) 个，已保存到 failed_slits.txt"
```
- 此段原样保留。此前提供的完整版在循环前有 `$failedSlits = @()`；用户最后片段省略了初始化。独立重跑时应初始化，以免沿用旧值或受变量类型影响。
- `failed_slits.txt` 只保存失败号码，每行一个，覆盖旧文件；不是完整 stdout/stderr 日志。运行目录应为 `data_prep`。

## 已确认的排错信息
- `data_prep/main.py:13` 导入 `plot`，`data_prep/plot.py:4` 导入 `core.find_snr`。
- 已核对 `core/find_snr.py` 在项目根目录下，与 `data_prep` 同级。曾出现 `ModuleNotFoundError: No module named 'core'`。
- 建议在当前 PowerShell 设置 `$env:PYTHONPATH = 'D:\_RschArchives\RSCH3\kl_binospec'`，并进入 `data_prep`。设置仅对当前终端及子进程生效。
- 用户直接通过编辑器运行的解释器为 `D:\Miniconda3\envs\Spyder-env\python.exe`，终端曾仅显示失败标记，无 traceback。
- 用 `-u -X faulthandler` 运行并通过 `2>&1 | Tee-Object -FilePath run_debug.log` 捕获后，用户提供了 Windows fatal exception `0xc06d007f`，退出码 `-1066598273`。
- 堆栈显示在导入 Astropy 坐标模块时崩溃：`icrs_fk5_transforms.py` 的 `_icrs_to_fk5_matrix`；经 `astropy.wcs`、`meta_config.py:3` 到 `main.py:14`。发生在处理 slit 数据之前。

## 关键判断与未解决问题
- 不能将上述导入/底层崩溃视为各 slit 数据本身的问题。
- 优先怀疑 Conda 未完整激活造成 DLL 加载问题，但尚未证实，不能断言依赖损坏。
- 已建议用 Conda PowerShell hook 激活 `Spyder-env`，再单独执行 Astropy 导入测试；尚无测试结果，不记录为已修复。
- 参考：Conda 官方环境激活说明 https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#activating-an-environment

## 如用户要求继续排错
在 PowerShell 中执行：
```powershell
(& 'D:\Miniconda3\Scripts\conda.exe' 'shell.powershell' 'hook') |
    Out-String | Invoke-Expression
conda activate Spyder-env
python -X faulthandler -c "import sys; print(sys.executable, flush=True); import astropy.coordinates; print('Astropy 导入成功', flush=True)"
```
若成功，再在已设置 PYTHONPATH 的 `data_prep` 目录执行 `python -u -X faulthandler main.py`；若仍崩溃，再核对依赖版本及 DLL。当前不主动继续排错或重装包。

## 相关文件与验证边界
- `data_prep/main.py`、`data_prep/plot.py`、`data_prep/meta_config.py`、`core/find_snr.py`。
- `data_prep/failed_slits.txt`；诊断输出目标 `data_prep/run_debug.log`。
- 已只读核对源码导入及目录布局；运行崩溃证据来自用户粘贴，未在本会话自行复现，也未验证完整批处理结果。
