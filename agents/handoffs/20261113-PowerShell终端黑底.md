# PowerShell 终端黑底

- 任务：解决 VS Code 浅色主题下，本地 PowerShell 终端按上下键浏览历史命令时，输入行出现突兀的黑底白字。
- 现状：用户截图显示黑底从当前命令延伸至右侧空白区域，并非历史建议列表或选中文字。先前建议修改 PSReadLine 的 `Selection` 颜色，用户试过后无效；已向用户纠正判断。
- 可能原因：旧版 PSReadLine 重画命令行时可能错误设置背景色。微软终端故障说明称 PSReadLine 2.0.3 修复了黑色横条问题；尚未确认用户安装的版本。
- 已建议用户运行 `(Get-Module PSReadLine).Version`。如果版本为 2.0.0–2.0.2，运行 `Install-Module PSReadLine -Scope CurrentUser -Force`，然后关闭并重新打开 PowerShell 终端验证；若版本已是 2.0.3 或更高，再检查 VS Code 终端渲染设置。
- 未解决：用户尚未反馈 PSReadLine 版本和更新后的效果。此前临时执行的 `Selection` 设置只对当前终端有效；若用户曾写入 `$PROFILE`，应删去该行。
- 相关文件：本问题尚未修改项目代码或 PowerShell 配置文件。
