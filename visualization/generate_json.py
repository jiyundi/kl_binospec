from pathlib import Path
import json

# 修改成你的仓库本地路径
ROOT = Path(r"../../../../_Archived/Websites/jiyundi.github.io/assets/runs_to_view")
json_path =  "../../../../_Archived/Websites/jiyundi.github.io/assets/data/spectra.json"

result = {}

for run_dir in sorted(ROOT.glob("runs_*")):

    run = run_dir.name
    result[run] = {}

    for slit_dir in sorted(run_dir.glob("Slit_*")):

        slit = slit_dir.name

        spec = slit_dir / "best_fit_spec.png"
        corner = slit_dir / "corner_all.png"

        if spec.exists() and corner.exists():

            result[run][slit] = {
                "spec": f"/assets/runs_to_view/{run}/{slit}/best_fit_spec.png",
                "corner": f"/assets/runs_to_view/{run}/{slit}/corner_all.png"
            }

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"Generated spectra.json with {len(result)} runs.")