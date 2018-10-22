import numpy as np
import scipy.misc as misc
from PIL import Image
import cv2
import time

def normalize(F_L):
    return F_L/np.sqrt(np.sum(np.square(F_L)))

def cal_distance(A, B, A_prime, B_prime, a_x, a_y, b_x, b_y, patch_size):
    A_H = A.shape[0]
    A_W = A.shape[1]
    B_H = B.shape[0]
    B_W = B.shape[1]
    dx0 = dy0 = patch_size // 2
    dx1 = dy1 = patch_size // 2 + 1
    dx0 = min(a_x, b_x, dx0)
    dx1 = min(A_H - a_x, B_H - b_x, dx1)
    dy0 = min(a_y, b_y, dy0)
    dy1 = min(A_W - a_y, B_W - b_y, dy1)
    patch_A = A[a_x - dx0:a_x + dx1, a_y - dy0:a_y + dy1]
    patch_A_prime = A_prime[a_x - dx0:a_x + dx1, a_y - dy0:a_y + dy1]
    patch_B = B[b_x - dx0:b_x + dx1, b_y - dy0:b_y + dy1]
    patch_B_prime = B_prime[b_x - dx0:b_x + dx1, b_y - dy0:b_y + dy1]
    dist = (np.sum((patch_A - patch_B) ** 2 + (patch_A_prime - patch_B_prime) ** 2)) / ((dx0 + dx1) * (dy0 + dy1))
    return dist

def init_nnf(A, B):
    A_H = A.shape[0]
    A_W = A.shape[1]
    nnf = np.zeros([A_H, A_W, 2], dtype=np.int32)
    nnf[:, :, 0] = np.random.randint(0, B.shape[0], size=[A_H, A_W])
    nnf[:, :, 1] = np.random.randint(0, B.shape[1], size=[A_H, A_W])
    return nnf

def init_nnd(A, B, A_prime, B_prime, SHAPE, nnf, patch_size):
    A_H = SHAPE[0]
    A_W = SHAPE[1]
    dist = np.zeros([A_H, A_W])
    for i in range(A_H):
        for j in range(A_W):
            dist[i, j] = cal_distance(A, B, A_prime, B_prime, i, j, nnf[i, j, 0], nnf[i, j, 1], patch_size)
    return dist

def propagation(A, B, A_prime, B_prime, SHAPE, a_x, a_y, nnf, nnd, patch_size, is_odd):
    A_H = SHAPE[0]
    A_W = SHAPE[1]
    B_H = SHAPE[2]
    B_W = SHAPE[3]
    if is_odd:
        d_best = nnd[a_x, a_y]
        best_b_x = nnf[a_x, a_y, 0]
        best_b_y = nnf[a_x, a_y, 1]
        if a_y - 1 >= 0:
            b_x = nnf[a_x, a_y - 1, 0]
            b_y = nnf[a_x, a_y - 1, 1] + 1
            if b_y < B_W:
                dist = cal_distance(A, B, A_prime, B_prime, a_x, a_y, b_x, b_y, patch_size)
                if dist < d_best:
                    best_b_x, best_b_y, d_best = b_x, b_y, dist
        if a_x - 1 >= 0:
            b_x = nnf[a_x - 1, a_y, 0] + 1
            b_y = nnf[a_x - 1, a_y, 1]
            if b_x < B_H:
                dist = cal_distance(A, B, A_prime, B_prime, a_x, a_y, b_x, b_y, patch_size)
                if dist < d_best:
                    best_b_x, best_b_y, d_best = b_x, b_y, dist
        nnf[a_x, a_y] = [best_b_x, best_b_y]
        nnd[a_x, a_y] = d_best
    else:
        d_best = nnd[a_x, a_y]
        best_b_x = nnf[a_x, a_y, 0]
        best_b_y = nnf[a_x, a_y, 1]
        if a_y + 1 < A_W:
            b_x = nnf[a_x, a_y + 1, 0]
            b_y = nnf[a_x, a_y + 1, 1] - 1
            if b_y >= 0:
                dist = cal_distance(A, B, A_prime, B_prime, a_x, a_y, b_x, b_y, patch_size)
                if dist < d_best:
                    best_b_x, best_b_y, d_best = b_x, b_y, dist
        if a_x + 1 < A_H:
            b_x = nnf[a_x + 1, a_y, 0] - 1
            b_y = nnf[a_x + 1, a_y, 1]
            if b_x >= 0:
                dist = cal_distance(A, B, A_prime, B_prime, a_x, a_y, b_x, b_y, patch_size)
                if dist < d_best:
                    best_b_x, best_b_y, d_best = b_x, b_y, dist
        nnf[a_x, a_y] = [best_b_x, best_b_y]
        nnd[a_x, a_y] = d_best

    return nnf, nnd

def random_search(A, B, A_prime, B_prime, SHAPE, a_x, a_y, nnf, nnd, search_radius, patch_size):
    B_H = SHAPE[2]
    B_W = SHAPE[3]
    best_b_x = nnf[a_x, a_y, 0]
    best_b_y = nnf[a_x, a_y, 1]
    best_dist = nnd[a_x, a_y]
    while search_radius >= 1:
        start_x = max(best_b_x - search_radius, 0)
        end_x = min(best_b_x + search_radius + 1, B_H)
        start_y = max(best_b_y - search_radius, 0)
        end_y = min(best_b_y + search_radius + 1, B_W)
        b_x = np.random.randint(start_x, end_x)
        b_y = np.random.randint(start_y, end_y)
        dist = cal_distance(A, B, A_prime, B_prime, a_x, a_y, b_x, b_y, patch_size)
        if dist < best_dist:
            best_dist = dist
            best_b_x = b_x
            best_b_y = b_y
        search_radius /= 2
    nnf[a_x, a_y, 0] = best_b_x
    nnf[a_x, a_y, 1] = best_b_y
    nnd[a_x, a_y] = best_dist
    return nnf, nnd

def NNF_Search(A, B, A_prime, B_prime, nnf, patch_size, itrs, search_radius):
    A = normalize(A)
    B = normalize(B)
    A_prime = normalize(A_prime)
    B_prime = normalize(B_prime)
    A_H = A.shape[0]
    A_W = A.shape[1]
    B_H = B.shape[0]
    B_W = B.shape[1]
    SHAPE = [A_H, A_W, B_H, B_W]
    nnd = init_nnd(A, B, A_prime, B_prime, SHAPE, nnf, patch_size)
    for itr in range(1, itrs + 1):
        if itr % 2 == 0:
            for i in range(A_H - 1, -1, -1):
                for j in range(A_W - 1, -1, -1):
                    nnf, nnd = propagation(A, B, A_prime, B_prime, SHAPE, i, j, nnf, nnd, patch_size, False)
                    nnf, nnd = random_search(A, B, A_prime, B_prime, SHAPE, i, j, nnf, nnd, search_radius, patch_size)
        else:
            for i in range(A_H):
                for j in range(A_W):
                    nnf, nnd = propagation(A, B, A_prime, B_prime, SHAPE, i, j, nnf, nnd, patch_size, True)
                    nnf, nnd = random_search(A, B, A_prime, B_prime, SHAPE, i, j, nnf, nnd, search_radius, patch_size)
    return nnf

def warp(f, B):
    A_h = np.size(f, 0)
    A_w = np.size(f, 1)
    A_c = np.size(B, 2)
    temp = np.zeros([A_h, A_w, A_c])
    for i in range(A_h):
        for j in range(A_w):
            temp[i, j, :] = B[f[i, j][0], f[i, j][1], :]
    return temp

def reconstruction_avg(nnf, B, patch_size):
    A_h = nnf.shape[0]
    A_w = nnf.shape[1]
    B_h = B.shape[0]
    B_w = B.shape[1]
    A_c = B.shape[2]
    rec = np.zeros([A_h, A_w, A_c])
    x0 = y0 = patch_size // 2
    x1 = y1 = patch_size // 2 + 1
    for i in range(A_h):
        for j in range(A_w):
            b_x = nnf[i, j, 0]
            b_y = nnf[i, j, 1]
            start_x = max(b_x - x0, 0)
            end_x = min(b_x + x1, B_h)
            start_y = max(b_y - y0, 0)
            end_y = min(b_y + y1, B_w)
            rec[i, j, :] = np.mean(B[start_x:end_x, start_y:end_y, :], axis=(0, 1))
    return rec

def upsample_nnf(nnf):
    """
    Upsample NNF based on size. It uses nearest neighbour interpolation
    :param size: INT size to upsample to.

    :return: upsampled NNF
    """

    temp = np.zeros((nnf.shape[0], nnf.shape[1], 3))

    for y in range(nnf.shape[0]):
        for x in range(nnf.shape[1]):
            temp[y][x] = [nnf[y][x][0], nnf[y][x][1], 0]

    # img = np.zeros(shape=(size, size, 2), dtype=np.int)
    # small_size = nnf.shape[0]
    aw_ratio = 2#((size) // small_size)
    ah_ratio = 2#((size) // small_size)

    temp = cv2.resize(temp, None, fx=aw_ratio, fy=aw_ratio, interpolation=cv2.INTER_NEAREST)
    img = np.zeros(shape=(temp.shape[0], temp.shape[1], 2), dtype=np.int)
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            pos = temp[i, j]
            img[i, j] = pos[0] * aw_ratio, pos[1] * ah_ratio

    return img

if __name__ == "__main__":
    A = np.array(Image.open("img.jpg"))
    A_prime = np.array(Image.open("img_.jpg"))
    B = np.array(Image.open("img.jpg"))
    B_prime = np.array(Image.open("img_.jpg"))
    patch_size = 5
    search_radius = 6
    itrs = 2
    nnf = init_nnf(A, B)
    start = time.time()
    nnf = NNF_Search(A, B, A_prime, B_prime, nnf, patch_size, itrs, search_radius)
    rec = warp(nnf, B)
    end = time.time()
    print(end - start)
    Image.fromarray(np.uint8(rec)).show()
    Image.fromarray(np.uint8(rec)).save("C://Users//gmt//Desktop//gmt.jpg")
    aaa = 0