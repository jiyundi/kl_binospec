# 成像邻近源手动遮罩

1. 在 `data_prep/` 目录打开并运行 `manual_image_mask.ipynb`。设置 `slit_num` 和预览尺寸；需要 `numpy astropy matplotlib ipympl ipywidgets`。
2. 点击“添加椭圆”，再点击左图设置中心；可在输入框精调相对目标中心的 x/y 偏移、椭圆半轴和旋转角度。重复添加可遮罩多个邻近源。右图实时显示结果；确认后点击“保存方案”。不需要遮罩时也可保存空方案。
3. 默认写入 `data_prep/image_mask_plans/slit_NNN_image_masks.json`。`main.py` 默认从该目录读取，也可使用 `--image_mask_plan /path/to/plan.json`。从未有过旧遮罩、且未保存方案的 slit 不额外遮罩；原先有硬编码遮罩的 slit 若缺少方案则报错，避免旧遮罩静默消失。跨电脑运行时复制该 JSON 文件。

方案中的椭圆中心以相对目标星系的像素偏移保存。运行时依据当前裁切图 WCS 定位目标，再加上偏移，因此改变 `--img_width` 或 `--img_height` 不会平移遮罩。半轴单位是成像像素；角度单位为度，从图像 x 轴向 y 轴逆时针旋转。若更换图像或像素尺度，需重新检查方案。

原有硬编码遮罩不会自动迁移；请在受影响的 slit 上逐个审核并保存新方案。
