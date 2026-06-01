import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


kwds_list = [
    {'slit_num':8, 'mask_x0':11, 'mask_y0':18, 'dx':2, 'dy':2.5},
    {'slit_num':13, 'mask_x0':[3,23], 'mask_y0':[0,23], 'dx':[8,9], 'dy':[5,9]},
    {'slit_num':14, 'mask_x0':0, 'mask_y0':18, 'dx':3, 'dy':3},
    {'slit_num':27, 'mask_x0':22, 'mask_y0':4, 'dx':3, 'dy':3},
    {'slit_num':28, 'mask_x0':0, 'mask_y0':0, 'dx':4, 'dy':4},
    {'slit_num':35, 'mask_x0':8, 'mask_y0':20, 'dx':3, 'dy':1}, 
    {'slit_num':42, 'mask_x0':3, 'mask_y0':16, 'dx':5, 'dy':4},
    {'slit_num':55, 'mask_x0':20, 'mask_y0':23, 'dx':9, 'dy':7},
    {'slit_num':58, 'mask_x0':13, 'mask_y0':4, 'dx':3, 'dy':3},
    {'slit_num':64, 'mask_x0':3, 'mask_y0':3, 'dx':3, 'dy':3}, 
    {'slit_num':67, 'mask_x0':0, 'mask_y0':8, 'dx':3, 'dy':4},
    {'slit_num':69, 'mask_x0':0, 'mask_y0':19, 'dx':3, 'dy':3},
    {'slit_num':70, 'mask_x0':21, 'mask_y0':5, 'dx':3, 'dy':6},
    {'slit_num':75, 'mask_x0':25, 'mask_y0':12, 'dx':8, 'dy':9},
    {'slit_num':91, 'mask_x0':4.5, 'mask_y0':5, 'dx':1.5, 'dy':1.5},
    {'slit_num':99, 'mask_x0':15, 'mask_y0':2, 'dx':3, 'dy':3},
    {'slit_num':128, 'mask_x0':[10,22], 'mask_y0':[0,13], 'dx':[5,3], 'dy':[2,4]},
    {'slit_num':132, 'mask_x0':[4,12,21], 'mask_y0':[0,18,16], 'dx':[3,2,2], 'dy':[3,2,2]},
    {'slit_num':133, 'mask_x0':5, 'mask_y0':20, 'dx':5, 'dy':4},
    {'slit_num':139, 'mask_x0':12, 'mask_y0':23, 'dx':6, 'dy':4},
    ]


def find_mask_pars(slit_num):
    for dic in kwds_list:
        if dic['slit_num'] == slit_num:
            return dic['mask_x0'], dic['mask_y0'], dic['dx'], dic['dy']


def mask_neighbor_star(image_data, image_mask, 
                       mask_x0=0, mask_y0=0, dx=4, dy=3,
                       theta=0/180*np.pi):
    ny, nx = image_data.shape

    if not isinstance(mask_x0, list):
        mask_x0 = [mask_x0]
        mask_y0 = [mask_y0]
        dx = [dx]
        dy = [dy]

    masks = np.ones(image_mask.shape, dtype=bool)
    for i in range(len(mask_x0)):
        xs = np.arange(np.min([0,  int(mask_x0[i])]),
                       np.max([nx, int(mask_x0[i]) + nx + 2]))
        ys = np.arange(np.min([0,  int(mask_y0[i])]),
                       np.max([ny, int(mask_y0[i]) + ny + 2]))
        xxs, yys = np.meshgrid(xs, ys)
        mask_pad = np.ones(xxs.shape, dtype=bool)

        r_tranf = _Gauss_2d(
            xxs, yys,
            mask_x0[i], mask_y0[i], dx[i], dy[i],
            theta) # θ parallel dx

        x_origin, y_origin = list(xs).index(0), list(ys).index(0)

        mask_pad[r_tranf > np.exp(-1)] = False

        mask = mask_pad[y_origin: y_origin + ny,
                        x_origin: x_origin + nx]
        masks = masks & mask
        
    mask = masks
    final_mask = (mask & image_mask)

    fig = plt.figure(figsize=(6, 3), dpi=100) # length, height
    gs = fig.add_gridspec(nrows=1, ncols=2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1.imshow(np.where(image_mask, image_data, np.nan))
    ax2.imshow(np.where(final_mask, image_data, np.nan))
    ax1.set_title('Before masking')
    ax2.set_title('After masking')
    plt.show()

    return final_mask


def _Gauss_2d(x, y, x0, y0, dx, dy, theta):
    # x′ =  (x − x0) cosθ + (y − y0) sinθ
    # y′ = −(x − x0) sinθ + (y − y0) cosθ
    # G  = A exp[−​ x′^2 / (2 dx^2) ​− ​y′^2 / (2 dy^2)​]

    a = (np.cos(theta) ** 2) / (2 * dx ** 2) + \
        (np.sin(theta) ** 2) / (2 * dy ** 2)
    b = -(np.sin(2 * theta)) / (4 * dx ** 2) + \
        (np.sin(2 * theta)) / (4 * dy ** 2)
    c = (np.sin(theta) ** 2) / (2 * dx ** 2) + \
        (np.cos(theta) ** 2) / (2 * dy ** 2)
    
    return np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))

        