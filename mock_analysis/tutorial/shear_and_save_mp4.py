import numpy as np
import cv2


def build_A(g1, g2, k=0):
    return np.array([
        [1 - k - g1 + k*g1,  k*g2 - g2],
        [k*g2 - g2,         1 + g1 - k - k*g1]
    ])


def warp_frame_cv2(frame, A):
    coords_intr = A @ coords + (I - A) @ center

    map_x = coords_intr[0].reshape(h, w).astype(np.float32)
    map_y = coords_intr[1].reshape(h, w).astype(np.float32)
    
    frame_out = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return frame_out

frame_start = 19 * 60
npoints_g1 , npoints_g2 = 180, 180
g1_min, g1_max = 0, 0.3
g2_min, g2_max = 0, 0.3

ks  = np.zeros(npoints_g1)
g1s = np.linspace(g1_min, g1_max, npoints_g1)
g2s = np.zeros(npoints_g1)

ks  = np.append(ks,  ks[ -1] * np.ones(60))
g1s = np.append(g1s, g1s[-1] * np.ones(60))
g2s = np.append(g2s, g2s[-1] * np.ones(60))

ks  = np.append(ks,  ks[ -1] * np.ones(npoints_g2))
g1s = np.append(g1s, g1s[-1] * np.ones(npoints_g2))
g2s = np.append(g2s, np.linspace(g2_min, g2_max, npoints_g2))

ks  = np.append(np.zeros(frame_start),  ks)
g1s = np.append(np.zeros(frame_start), g1s)
g2s = np.append(np.zeros(frame_start), g2s)

folder = '../../../../../../Pr_Project_Materials/PR55科研KL演示/'
input_video_path  = f"{folder}弱剪切几何（本征形状）.mp4"
output_video_path = f"{folder}弱剪切几何（本征形状）sheared.mp4"

# ====== 打开视频 ======
cap = cv2.VideoCapture(input_video_path)

# 获取视频参数
fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
xs = np.indices((h, w))[1]
ys = np.indices((h, w))[0]
x_center = w/2
y_center = h/2
coords0 = np.stack([xs, ys], axis=0).reshape(2, -1)
center  = np.array([[x_center], [y_center]], dtype=np.float32)
I = np.eye(2, dtype=np.float32)

# Stop shear after npoints
k_vals  = np.append(ks,  ks[ -1] * np.ones(nframes - npoints_g1 - npoints_g2))
g1_vals = np.append(g1s, g1s[-1] * np.ones(nframes - npoints_g1 - npoints_g2))
g2_vals = np.append(g2s, g2s[-1] * np.ones(nframes - npoints_g1 - npoints_g2))

# ====== 输出视频 ======
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
# ====== 主循环 ======
i = 0
while True:
    coords = coords0.copy()
    ret, frame_in = cap.read()
    if not ret:
        break

    A = build_A(g1_vals[i], g2_vals[i], k_vals[i])

    frame_out = warp_frame_cv2(frame_in, A)

    out.write(frame_out)

    i += 1
    if i % 60 == 0:
        print(f"Processed {i}/{nframes}")

cap.release()
out.release()
print("Done!")