import argparse
import os
from PIL import Image

def main():
    parser = argparse.ArgumentParser(
        description="将当前目录下所有 Slit_* 文件夹内的图表合并为一个 PDF 文件。"
    )
    parser.add_argument(
        "-c", "--corner-only",
        action="store_true",
        help="只提取每个文件夹内的 corner_all.png"
    )
    parser.add_argument(
        "-b", "--best-fit-only",
        action="store_true",
        help="只提取每个文件夹内的 best_fit_spec.png"
    )

    args = parser.parse_args()

    # 根据参数决定包含哪些图片
    if args.corner_only and not args.best_fit_only:
        target_files = ["corner_all.png"]
        output_pdf   = "merged_corner.pdf"
    elif args.best_fit_only and not args.corner_only:
        target_files = ["best_fit_spec.png"]
        output_pdf   = "merged_best_fit.pdf"
    else:
        # 默认或同时指定 -c -b 时，两张图都包含
        target_files = ["best_fit_spec.png", "corner_all.png"] # corner_compare, corner_all
        output_pdf   = "combined_best_fit_corner.pdf"

    current_dir = os.getcwd()
    print(f"正在扫描目录: {current_dir}")

    # 筛选并自然排序 Slit_* 文件夹
    slit_dirs = [
        d for d in os.listdir(current_dir)
        if os.path.isdir(os.path.join(current_dir, d)) and d.startswith("Slit_")
    ]
    slit_dirs.sort()

    if not slit_dirs:
        print("未找到任何以 'Slit_' 开头的文件夹！")
        return

    images_to_convert = []

    # 遍历文件夹收集指定图片
    for s_dir in slit_dirs:
        dir_path = os.path.join(current_dir, s_dir)

        for fname in target_files:
            img_path = os.path.join(dir_path, fname)
            if not os.path.exists(img_path):
                print(f"警告: {s_dir} 中缺失 {fname}，跳过该图。")
                continue

            img = Image.open(img_path).convert("RGB")
            images_to_convert.append(img)

    if not images_to_convert:
        print("没有找到可用于合并的图片！")
        return

    # 输出为 merged_slits.pdf
    first_image = images_to_convert[0]
    rest_images = images_to_convert[1:]

    first_image.save(
        output_pdf,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=rest_images
    )

    print(f"成功导出 {len(images_to_convert)} 页图表到: {os.path.abspath(output_pdf)}")

if __name__ == "__main__":
    main()
