from pathlib import Path
import shutil

all_runs_dir     = Path("../../../RSCH3/kl_github/.")
all_runs_dir_des = Path('../../../RSCH3/kl_github/the_converted/.')

# 如果你想先看看会发生什么，把 DRY_RUN=True
DRY_RUN = False # True 

for runs_dir in all_runs_dir.glob("runs_20260414"):
    if not runs_dir.is_dir():
        continue

    runs_name = runs_dir.name  # e.g. runs_20260103/

    for slit_dir in runs_dir.glob("Slit_*"):
        if not slit_dir.is_dir():
            continue

        target_slit = all_runs_dir / slit_dir.name
        target_runs = all_runs_dir_des / slit_dir.name / runs_name
        
        print(f"Note: {slit_dir}\n"+
              f"----> {target_runs}\n")

        if not DRY_RUN:
            target_runs.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(slit_dir, target_runs)

    # 如果 runs_2026xxxx 变空，可以选择删
    if not DRY_RUN and not any(runs_dir.iterdir()):
        runs_dir.rmdir()

# Step 2. Delete empty corner plot JPGs
# SIZE_LIMIT = 10 * 1024   # 10 KB
# for img in all_runs_dir.rglob("corner.jpg"):
#     size = img.stat().st_size

#     if size < SIZE_LIMIT:
#         print(f"Delete: {img}  ({size/1024:.1f} KB)")

#         if not DRY_RUN:
#             img.unlink()

print('Done.')