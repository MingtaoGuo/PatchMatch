import numpy as np
from PIL import Image
import scipy.misc as misc
import time

def distance(patch1, pathc2):
    return np.mean(np.abs(patch1 - pathc2))

def to_valid(x, min, max):
    if x < min:
        x = np.int32(min)
    if x > max:
        x = np.int32(max)
    return np.int32(x)

def pick_patch(center, w_size, img):
    #center: [r, c], the center location of the patch
    h = np.size(img, 0)
    w = np.size(img, 1)
    c = np.size(img, 2)
    pad_size = (w_size//2)*2
    temp = np.zeros([h+pad_size, w+pad_size, c])
    temp[w_size//2:-(w_size//2), w_size//2:-(w_size//2), :] = img
    # temp = np.pad(img, ((1, 1), ()))
    p = w_size//2
    r = center[0] + p
    c = center[1] + p
    return temp[r-p:r+p+1, c-p:c+p+1, :]

def initialization(A, B):
    h = np.size(B, 0)
    w = np.size(B, 1)
    b = np.zeros([h, w], dtype=np.object)
    for r in range(h):
        for c in range(w):
            b[r, c] = np.array([r, c])
    b = np.reshape(b, [-1, 1]) * 1
    np.random.shuffle(b)
    b = np.reshape(b, [h, w])
    h = np.size(A, 0)
    w = np.size(A, 1)
    a = np.zeros([h, w], dtype=np.object)
    for r in range(h):
        for c in range(w):
            a[r, c] = np.array([r, c])
    f = b[:h, :w] - a
    return f

def Propatation(f, a, A, B, w_size=7, is_even=True):
    #f: a->offset, a is coordinates of A, b = a + offset
    #pick a patch which the center is a, the size is w_size from A
    minimum = np.inf
    x = a[0]
    y = a[1]
    if is_even:
        #f(x, y)
        b = a + f[x, y]#mapping the coordinate a to b
        patch_a = pick_patch(a, w_size, A)
        patch_b = pick_patch(b, w_size, B)
        D_f_x_y =distance(patch_a, patch_b)
        if D_f_x_y < minimum:
            f[x, y] = f[x, y]
            minimum = D_f_x_y
        #f(x-1, y)
        a = np.array([x-1, y])
        b = a + f[x-1, y]
        patch_a = pick_patch(a, w_size, A)
        patch_b = pick_patch(b, w_size, B)
        D_f_x_1_y = distance(patch_a, patch_b)
        if D_f_x_1_y < minimum:
            f[x, y] = f[x-1, y] + a - np.array([x, y])
            minimum = D_f_x_1_y
        #f(x, y-1)
        a = np.array([x, y - 1])
        b = a + f[x, y - 1]
        patch_a = pick_patch(a, w_size, A)
        patch_b = pick_patch(b, w_size, B)
        D_f_x_y_1 = distance(patch_a, patch_b)
        if D_f_x_y_1 < minimum:
            f[x, y] = f[x, y-1] + a - np.array([x, y])
    else:
        # f(x, y)
        b = a + f[x, y]  # mapping the coordinate a to b
        patch_a = pick_patch(a, w_size, A)
        patch_b = pick_patch(b, w_size, B)
        D_f_x_y = distance(patch_a, patch_b)
        if D_f_x_y < minimum:
            f[x, y] = f[x, y]
            minimum = D_f_x_y
        # f(x+1, y)
        a = np.array([x + 1, y])
        b = a + f[x + 1, y]
        patch_a = pick_patch(a, w_size, A)
        patch_b = pick_patch(b, w_size, B)
        D_f_x_1_y = distance(patch_a, patch_b)
        if D_f_x_1_y < minimum:
            f[x, y] = f[x + 1, y] + a - np.array([x, y])
            minimum = D_f_x_1_y
        # f(x, y+1)
        a = np.array([x, y + 1])
        b = a + f[x, y + 1]
        patch_a = pick_patch(a, w_size, A)
        patch_b = pick_patch(b, w_size, B)
        D_f_x_y_1 = distance(patch_a, patch_b)
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
        patch_b = pick_patch(b, w_size, B)
        patch_a = pick_patch(a, w_size, A)
        d = distance(patch_a, patch_b)
        if d < Dist:
            f[x, y] = b - a
            Dist = d
        i += 1

def map_show(name, f, A, B):
    h = np.size(A, 0)
    w = np.size(A, 1)
    c = np.size(A, 2)
    temp = np.zeros([h, w, c])
    for i in range(h):
        for j in range(w):
            temp[i, j, :] = B[f[i, j][0] + i, f[i, j][1] + j, :]
    Image.fromarray(np.uint8(temp)).save(name+".jpg")


if __name__ == "__main__":
    img_ref = np.array(Image.open("bike_b.png"))
    h = img_ref.shape[0]
    w = img_ref.shape[1]
    img = np.array(Image.open("bike_a.png"))
    img_h = img.shape[0]
    img_w = img.shape[1]
    start = time.time()
    f = initialization(img, img_ref)
    end = time.time()
    print(end-start)
    for iteration in range(10):
        if iteration % 2 !=0:
            for i in range(img_h-1, 0, -1):
                for j in range(img_w-1, 0, -1):
                    Propatation(f, np.array([i, j]), img, img_ref, w_size=3, is_even=True)
                    random_search(f, np.array([i, j]), img, img_ref, w_size=3)
                print(i)
        else:
            for i in range(0, img_h-1):
                for j in range(0, img_w-1):
                    Propatation(f, np.array([i, j]), img, img_ref, w_size=3, is_even=False)
                    random_search(f, np.array([i, j]), img, img_ref, w_size=3)
                print(i)
        print("Iteration: %d"%(iteration))
        map_show(str(iteration), f, img, img_ref)
    a = 0
