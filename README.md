1. core/ 放被其他文件 import 的东西。
   * `doublet_utils.py`, `fitting_result_utils.py`, `post_fitting.py` 是纯库代码，没有 __main__，是依赖的起点而不是终点。
   * `find_pkl_structures.py`, `survey_chi2.py`, `survey_progress.py` 虽然有 __main__，但它们的核心是函数，其他分析脚本可能也会 import 它们，所以也归这里。

3. scripts/ 放有 __main__ 且直接驱动 pipeline 的文件
   * 你每次跑拟合会 `python run_nautilus.py` 这种。
   * 三个文件都建议顺带改名：去掉 binospec_ 前缀（repo 名字已经是 context 了），main_fitting 改成 run_nautilus 更清楚说明"做什么"而不是"是什么"。

5. analysis/ 放产出可见结果（图、表）的脚本。
   * 和 scripts/ 的区别是：scripts/ 负责跑拟合，analysis/ 负责读结果出图。
   * `survey_progress.py` → `survey_sky_plot.py`
   * `make_a_simple_RC.py` → `make_rc_diagram.py`
   * 这两个改名可以让陌生人一眼看出文件产出什么。

7. archive/ 比直接删更 github 友好。
   * `binospec_main_fitting.py`（ultranest 旧版）、`binospec_plot_best_fit_only.py`（已被 post_fitting 替代）、以及那些"不用管"的调试草稿放这里，README 里一行说明即可，clone 的人不会困惑。
