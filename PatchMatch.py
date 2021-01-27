import numpy as np
from PIL import Image
import time
from numba import jit
import matplotlib.pyplot as plt

#cette fonction est tres souvent appelée
@jit(nopython=True)
def cal_distance(a, b, A_padding, B, p_size):
    p = p_size // 2
    patch_a = A_padding[a[0]:a[0]+p_size, a[1]:a[1]+p_size, :]
    patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, :]
    temp = patch_b - patch_a
    num = 0
    dist = 0
    for i in range(p_size):
        for j in range(p_size):
            for k in range(3):
                if not np.isnan(temp[i,j,k]):
                    num +=1 
                    dist += np.square(temp[i,j,k])
    dist /= num
    return dist

def reconstruction(f, A, B):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    temp = np.zeros_like(A)
    for i in range(A_h):
        for j in range(A_w):
            temp[i, j, :] = B[f[i, j][0], f[i, j][1], :]
    Image.fromarray(temp).show()

def initialization(A, B, p_size):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    random_B_r = np.random.randint(p, B_h-p, [A_h, A_w])
    random_B_c = np.random.randint(p, B_w-p, [A_h, A_w])
    A_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
    A_padding[p:A_h+p, p:A_w+p, :] = A
    f = np.zeros([A_h, A_w], dtype=object)
    dist = np.zeros([A_h, A_w])
    for i in range(A_h):
        for j in range(A_w):
            a = np.array([i, j])
            b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)
            f[i, j] = b
            dist[i, j] = cal_distance(a, b, A_padding, B, p_size)
    return f, dist, A_padding

# je fais cette fonction car numba arrive pas 
# a trouver le type de f dans random_search
@jit(nopython=True)
def inter(b_x, search_h,p,B_h,b_y, search_w,B_w,alpha,i): 
    search_min_r = max(b_x - search_h, p)
    search_max_r = min(b_x + search_h, B_h-p)
    random_b_x = np.random.randint(search_min_r, search_max_r)
    search_min_c = max(b_y - search_w, p)
    search_max_c = min(b_y + search_w, B_w - p)
    random_b_y = np.random.randint(search_min_c, search_max_c)
    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i
    b = np.array([random_b_x, random_b_y])
    return search_h, search_w, b
    

def random_search(f, a, dist, A_padding, B, p_size, alpha=0.5):
    x = a[0]
    y = a[1]
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    i = 4
    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y][0]
    b_y = f[x, y][1]
    alpha=alpha
    while search_h > 1 and search_w > 1:
        search_h, search_w, b = inter(b_x, search_h,p,B_h,b_y, search_w,B_w,alpha,i)
        d = cal_distance(a, b, A_padding, B, p_size)
        if d < dist[x, y]:
            dist[x, y] = d
            f[x, y] = b
        i += 1
            
def propagation(f, a, dist, A_padding, B, p_size, is_odd):
    p = p_size // 2
    A_h = np.size(A_padding, 0) - p_size + 1
    A_w = np.size(A_padding, 1) - p_size + 1
    x = a[0]
    y = a[1]
    if is_odd:
        i_left,j_left = f[max(x - 1, 0), y]
        i_left = min(i_left+1,A_h-1-p)
        b_left = np.array([i_left,j_left],dtype="int64")
        d_left = cal_distance(a,b_left, A_padding, B, p_size)
        
        i_up,j_up = f[x, max(y - 1, 0)]
        j_up = min(j_up+1,A_w-1-p)
        b_up = np.array([i_up,j_up],dtype="int64")
        d_up = cal_distance(a, b_up, A_padding, B, p_size)
        
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_left, d_up]))
        if idx == 1:
            f[x, y] = b_left
            dist[x, y] = d_left
        if idx == 2:
            f[x, y] = b_up
            dist[x, y] = d_up
    else:
        i_right,j_right = f[min(x + 1, A_h-1), y]
        i_right = max(i_right-1,p)
        b_right = np.array([i_right,j_right],dtype="int64")
        d_right = cal_distance(a, b_right, A_padding, B, p_size)
        
        i_down,j_down = f[x, min(y + 1, A_w-1)]
        j_down = max(j_down-1,p)
        b_down = np.array([i_down,j_down],dtype="int64")
        d_down = cal_distance(a, b_down, A_padding, B, p_size)
        
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y] = b_right
            dist[x, y] = d_right
        if idx == 2:
            f[x, y] = b_down
            dist[x, y] = d_down

def NNS(img, ref, p_size, itr):
    A_h = np.size(img, 0)
    A_w = np.size(img, 1)
    print("initialization")
    f, dist, img_padding = initialization(img, ref, p_size)
    print(f"score : {dist.mean()}")
    score_list = []
    for itr in range(1, itr+1):
        print("iteration: %d"%(itr))
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_w - 1, -1, -1):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, p_size, False)
                    random_search(f, a, dist, img_padding, ref, p_size)
        else:
            for i in range(A_h):
                for j in range(A_w):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, p_size, True)
                    random_search(f, a, dist, img_padding, ref, p_size)
        print(f"score : {dist.mean()}")
        score_list.append(dist.mean())
    return f, dist, score_list


if __name__ == "__main__":
    img = np.array(Image.open("./cup_a.jpg"))
    ref = np.array(Image.open("./cup_b.jpg"))
    p_size = 3
    itr = 5
    start = time.time()
    f, dist, score_list = NNS(img, ref, p_size, itr)
    end = time.time()
    print(f"Time :  {end - start}")
    reconstruction(f, img, ref)
    plt.plot(score_list)
