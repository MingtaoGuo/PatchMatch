import numpy as np
from PIL import Image
import time


def to_valid(x, min, max):
    if x < min:
        x = np.random.randint(min, max)
    if x > max:
        x = np.random.randint(min, max)
    return x

def cal_distance(A, B, a, b, w_size):
    a_x, a_y, b_x, b_y = a[0], a[1], b[0], b[1]
    temp = w_size // 2
    A_h, A_w, B_h, B_w = A.shape[0], A.shape[1], B.shape[0], B.shape[1]
    a_x_start, a_x_end, b_x_start, b_x_end= max(a_x - temp, 0), min(a_x + temp, A_h-1), max(b_x - temp, 0), min(b_x + temp, B_h - 1)
    a_y_start, a_y_end, b_y_start, b_y_end = max(a_y - temp, 0), min(a_y + temp, A_w - 1), max(b_y - temp, 0), min(b_y + temp, B_w - 1)
    d_x = min(a_x_end - a_x_start + 1, b_x_end - b_x_start + 1)
    d_y = min(a_y_end - a_y_start + 1, b_y_end - b_y_start + 1)
    patch_a = A[a_x_start:a_x_start + d_x - 1, a_y_start:a_y_start + d_y - 1]
    patch_b = B[b_x_start:b_x_start + d_x - 1, b_y_start:b_y_start + d_y - 1]
    dist = np.mean(np.abs(patch_b - patch_a))
    return dist


def initialization(A, B):
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    b_x = np.random.randint(0, B_h-1, [A_h])
    b_y = np.random.randint(0, B_w-1, [A_w])
    b = np.zeros([B_h, B_w], dtype=np.object)
    a = np.zeros([B_h, B_w], dtype=np.object)
    for r in range(A_h):
        for c in range(A_w):
            b[r, c] = np.array([b_x[r], b_y[c]])
            a[r, c] = np.array([r, c])
    f = b - a
    return f

def Propatation(f, a, A, B, w_size=7, is_even=True):
    #f: a->offset, a is the current coordinate of A, b = a + offset
    minimum = np.inf
    x = a[0]
    y = a[1]
    if not is_even:
        #f(x, y)
        b = a + f[x, y]#mapping the coordinate a to b
        D_f_x_y = cal_distance(A, B, a, b, w_size)
        if D_f_x_y < minimum:
            f[x, y] = f[x, y]
            minimum = D_f_x_y
        #f(x-1, y)
        a = np.array([x-1, y])
        b = a + f[x-1, y]
        D_f_x_1_y = cal_distance(A, B, a, b, w_size)
        if D_f_x_1_y < minimum:
            f[x, y] = f[x-1, y] + a - np.array([x, y])
            minimum = D_f_x_1_y
        #f(x, y-1)
        a = np.array([x, y - 1])
        b = a + f[x, y - 1]
        D_f_x_y_1 = cal_distance(A, B, a, b, w_size)
        if D_f_x_y_1 < minimum:
            f[x, y] = f[x, y-1] + a - np.array([x, y])
    else:
        # f(x, y)
        b = a + f[x, y]  # mapping the coordinate a to b
        D_f_x_y = cal_distance(A, B, a, b, w_size)
        if D_f_x_y < minimum:
            f[x, y] = f[x, y]
            minimum = D_f_x_y
        # f(x+1, y)
        a = np.array([x + 1, y])
        b = a + f[x + 1, y]
        D_f_x_1_y = cal_distance(A, B, a, b, w_size)
        if D_f_x_1_y < minimum:
            f[x, y] = f[x + 1, y] + a - np.array([x, y])
            minimum = D_f_x_1_y
        # f(x, y+1)
        a = np.array([x, y + 1])
        b = a + f[x, y + 1]
        D_f_x_y_1 = cal_distance(A, B, a, b, w_size)
        if D_f_x_y_1 < minimum:
            f[x, y] = f[x, y + 1] + a - np.array([x, y])

def random_search(f, a, A, B, w_size=7, alpha=0.5):
    h = np.size(B, 0)
    w = np.size(B, 1)
    W = np.array([h, w])
    x = a[0]
    y = a[1]
    v_0 = f[x, y]
    i = 0
    Dist = np.inf
    while h * alpha**i > 1 and w * alpha**i > 1:
        R_i = np.random.uniform(-1, 1, [2])
        u_i = np.int32(v_0 + W * alpha**i * R_i)
        b = a + u_i
        b[0] = to_valid(b[0], 0, h-1)
        b[1] = to_valid(b[1], 0, w-1)
        d = cal_distance(A, B, a, b, w_size)
        if d < Dist:
            f[x, y] = b - a
            Dist = d
        i += 1

def NNF(f, img, img_ref, w_size=7, total_itr=5):
    img_h = img.shape[0]
    img_w = img.shape[1]
    for iteration in range(total_itr):
        if iteration % 2 == 0:
            for i in range(0, img_h-2):
                for j in range(0, img_w-2):
                    Propatation(f, np.array([i, j]), img, img_ref, w_size, is_even=True)
                    random_search(f, np.array([i, j]), img, img_ref, w_size)
            print("Iteration: %d" % (iteration))
        else:
            for i in range(img_h-2, 0, -1):
                for j in range(img_w-2, 0, -1):
                    Propatation(f, np.array([i, j]), img, img_ref, w_size, is_even=False)
                    random_search(f, np.array([i, j]), img, img_ref, w_size)
            print("Iteration: %d"%(iteration))
        reconstruct(str(iteration), f, img, img_ref)

def reconstruct(name, f, A, B):
    h = np.size(A, 0)
    w = np.size(A, 1)
    c = np.size(A, 2)
    temp = np.zeros([h, w, c])
    for i in range(h):
        for j in range(w):
            temp[i, j, :] = B[f[i, j][0] + i, f[i, j][1] + j, :]
    Image.fromarray(np.uint8(temp)).save(name+".jpg")


if __name__ == "__main__":
    img = np.array(Image.open("cup_b.jpg"))
    img_ref = np.array(Image.open("cup_a.jpg"))
    h = img_ref.shape[0]
    w = img_ref.shape[1]
    start = time.time()
    f = initialization(img, img_ref)
    end = time.time()
    print(end-start)
    NNF(f, img, img_ref, w_size=7, total_itr=5)
    print(time.time() - start)
