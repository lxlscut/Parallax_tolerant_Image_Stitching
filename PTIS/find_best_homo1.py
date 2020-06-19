# todo 获取最佳的单应性矩阵
import numpy as np
import random
import copy
import cv2
from numba import jit
from util.get_border import get_border
from util.glabal_homography import GloablHomography
from util.image_info import Image_info
import NUMPY
import MSE
import time
import math
import maxflow


# todo 计算单应性矩阵的误差
def cauculate_h_error(h, src_point, dst_point):
    padding = np.ones((src_point.shape[0], 1), dtype=np.float)
    cal_src = np.concatenate([src_point, padding], axis=-1)
    cal_dst = h.dot(cal_src.T)
    cal_dst = cal_dst.T
    for i in range(cal_dst.shape[0]):
        cal_dst[i, :] = cal_dst[i, :] / cal_dst[i, 2]
    error = np.sqrt(((cal_dst[:, 0] - dst_point[:, 0]) ** 2) + ((cal_dst[:, 1] - dst_point[:, 1]) ** 2))
    return error


# todo 计算相似矩阵的误差
def calculate_s_error(origin, transformed, similar_matrix):
    origin = np.pad(origin, ((0, 0), (0, 1)), 'constant', constant_values=1)
    dst = similar_matrix.dot(origin.T)
    dst = dst.T
    error = np.linalg.norm((dst - transformed), axis=1, keepdims=False)
    error = np.sum(error)
    return error


# todo 填充空洞
def fill_hole(img):
    temp = np.zeros([img.shape[0] + 2, img.shape[1] + 2], dtype=np.uint8)
    temp[1:img.shape[0] + 1, 1:img.shape[1] + 1] = img
    cv2.floodFill(temp, None, (0, 0), 255)
    cut_img = temp[1:img.shape[0] + 1, 1:img.shape[1] + 1]
    result = img | (~cut_img)
    return result


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


# todo 考虑到速度，这里使用numba加速
@jit(nopython=True)
def get_fc(kps, ps, width, height):
    distance = np.zeros((ps.shape[0], ps.shape[1]), dtype=np.float64)
    gama = 0.01
    sigma = 0.05 * min(width, height)
    # sigma = 1.5
    for row in range(ps.shape[0]):
        for col in range(ps.shape[1]):
            for kp in kps:
                dis = np.sqrt((kp[0] - ps[row, col, 0]) ** 2 + (kp[1] - ps[row, col, 1]) ** 2)
                dis = math.exp(-(dis) ** 2 / (2 * sigma ** 2))
                distance[row, col] += dis
            distance[row, col] = 1. / (distance[row, col] + gama)
    return distance


def graph_cut(I, src_pts):
    g = maxflow.Graph[float]()
    i_inf = np.inf
    nodeids = g.add_grid_nodes(I.shape)
    h, w = np.shape(I)
    mu, sigma = 0, 1.5  # mean and standard deviation
    weights = np.zeros((h, w))
    l = 0
    for k in src_pts:
        weights_x = np.zeros(h)
        weights_y = np.zeros(w)
        weights_tmp = np.zeros((h, w))
        for i in range(h):
            weights_x[i] += (k[0][0] - i) ** 2
        weights_x = np.tile(weights_x, (w, 1))
        weights_x = np.transpose(weights_x)
        for i in range(w):
            weights_y[i] += (k[0][1] - i) ** 2
        weights_y = np.tile(weights_y, (h, 1))
        weights_tmp = weights_x + weights_y
        weights_tmp = gaussian(np.sqrt(weights_tmp), np.mean(weights_tmp), np.std(weights_tmp))
        weights += weights_tmp
    weights = 1.0 / (weights + 0.01)
    for i in range(h):
        for j in range(w):
            weights[i][j] = weights[i][j] * I[i][j]

    structure = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    g.add_grid_edges(nodeids, weights, structure=structure, symmetric=True)
    left_most = np.concatenate((np.arange(I.shape[0]).reshape(1, I.shape[0]), np.zeros((1, I.shape[0])))).astype(
        np.uint64)
    left_most = np.ravel_multi_index(left_most, I.shape)
    g.add_grid_tedges(left_most, i_inf, 0)

    right_most = np.concatenate(
        (np.arange(I.shape[0]).reshape(1, I.shape[0]), np.ones((1, I.shape[0])) * (np.size(I, 1) - 1))).astype(
        np.uint64)
    right_most = np.ravel_multi_index(right_most, I.shape)
    g.add_grid_tedges(right_most, 0, i_inf)
    x = g.maxflow()
    print(x)
    return x


def get_best_homography1(src_point, dst_point, src_img, dst_img, Hg_fitness_threshold=3, sum_threshold=0.01):
    # todo 论文中对应的惩罚项
    penalty = np.zeros((src_point.shape[0],), dtype=np.int32)
    # todo 看该点是否已经被选作种子点
    flag = np.zeros((src_point.shape[0],), dtype=np.int32)
    # todo 计算所有特征点之间的距离
    x_dim = src_point[:, 0]
    y_dim = src_point[:, 1]
    xx = np.zeros((src_point.shape[0], src_point.shape[0]), dtype=np.float32)
    yy = np.zeros((src_point.shape[0], src_point.shape[0]), dtype=np.float32)
    for i in range(src_point.shape[0]):
        xx[i, :] = x_dim
        yy[i, :] = y_dim
    distance = np.abs(xx - xx.T) ** 2 + np.abs(yy - yy.T) ** 2

    GET_MATRIX = False

    while not GET_MATRIX:
        # todo 表明所有的点都已经被抽样过
        if np.mean(flag) == 1:
            break
        # todo 随机抽取一点，并按距离来group点，拟合单应性矩阵
        a = random.randint(0, src_point.shape[0] - 1)
        if flag[a] == 1 or penalty[a] > np.average(penalty):
            continue
        else:
            flag[a] = 1
        current_distance = distance[a, :]
        index = np.argsort(current_distance)
        # todo 为什么从10开始，因为如果是4的话条件太容易满足，达到的条件的单应性矩阵会很多，达不到筛选的目的，这就会导致后面的运算量大大增加。
        base_point = index[:10]
        gh = GloablHomography()
        for i in range(10, src_point.shape[0], 1):
            current_src = src_point[index[: i], :]
            current_dst = dst_point[index[: i], :]
            cal_current_src = copy.deepcopy(current_src)
            cal_current_dst = copy.deepcopy(current_dst)
            h = gh.get_global_homo(src_point=cal_current_src, dst_point=cal_current_dst)
            err = cauculate_h_error(h=h, src_point=current_src, dst_point=current_dst)
            # todo 如果已经不能很好的拟合，
            if np.max(err) > Hg_fitness_threshold:
                # todo 如果拟合的点数太少，直接淘汰
                if i == 10:
                    GET_MATRIX = False
                    feature_group = index[: i]
                    break
                else:
                    GET_MATRIX = True
                    feature_group = index[: i - 1]
                    one = src_point[feature_group, :]
                    tow = dst_point[feature_group, :]
                    cal_current_src1 = copy.deepcopy(one)
                    cal_current_dst1 = copy.deepcopy(tow)
                    h = gh.get_global_homo(src_point=cal_current_src1, dst_point=cal_current_dst1)
                    err = cauculate_h_error(h=h, src_point=one, dst_point=tow)
                    # todo 所有的点惩罚值都加一
                    break
            elif i == src_point.shape[0]:
                feature_group = index[: i]
                break
        penalty[feature_group] = penalty[feature_group] + 1
        # 如果这一步没有获取到正确的单应性矩阵，那么直接下一步
        if not GET_MATRIX:
            continue
        # todo 获取初步拼接的图像
        image_info = Image_info()
        image_info.get_final_size(src_img=src_img, dst_img=dst_img, H=h)
        src_warp = NUMPY.warp_global(src_img, np.linalg.pinv(h), image_info.height, image_info.width,
                                     image_info.offset_x,
                                     image_info.offset_y)
        dst_warp = np.zeros([image_info.height, image_info.width, 3], dtype=np.uint8)
        dst_warp[image_info.offset_y:image_info.offset_y + dst_img.shape[0],
        image_info.offset_x:image_info.offset_x + dst_img.shape[1], :] = dst_img
        # temp1 = warp_global(img=src_img, image_info=image_info, H=h)

        src_warp = np.array(src_warp, dtype=np.uint8)
        cv2.imshow("dst_warp", dst_warp)
        cv2.imshow("src_warp", src_warp)
        # cv2.imshow("temp1", temp1)
        # todo 如果成功获取了单应性矩阵，那么接下来衡量其对其标准
        # todo 1.通过相似性矩阵衡量，也即homography screening
        # todo ||Hs.dot(Ci) - C^i ||**2,求最小的Hs
        transformed = np.array(
            [image_info.left_top, image_info.left_button, image_info.right_top, image_info.right_button],
            dtype=np.float)
        origin = np.array([[0, 0], [0, src_img.shape[0]], [src_img.shape[1], 0], [src_img.shape[1], src_img.shape[0]]],
                          dtype=np.float)
        a = MSE.get_similar_matrix(origin, transformed)
        similar_matrix = np.array([[a[0], -a[1], a[2]], [a[1], a[0], a[3]]], dtype=np.float)
        error_sum = calculate_s_error(origin, transformed, similar_matrix)
        error_sum = error_sum / (src_img.shape[0] * src_img.shape[1])
        if error_sum > sum_threshold:
            GET_MATRIX = False
        # todo 通过最佳缝合线来对对齐效果进行衡量

        # todo 源图转为2值图像
        # todo 获取掩膜
        ret, src_warp_b = cv2.threshold(cv2.cvtColor(src_warp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
        ret, dst_warp_b = cv2.threshold(cv2.cvtColor(dst_warp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
        src_warp_b = fill_hole(src_warp_b)
        dst_warp_b = fill_hole(dst_warp_b)
        mask = src_warp_b & dst_warp_b
        cv2.imshow("mask", mask)

        # todo 获取mask的范围
        (a, b) = np.where(mask > 0)
        minx = min(b)
        maxx = max(b)
        miny = min(a)
        maxy = max(a)

        kp_img = np.zeros((image_info.height, image_info.width))
        for i in feature_group:
            ssr = src_point[i, :]
            dds = dst_point[i, :]
            ssr = h.dot(np.array([ssr[0], ssr[1], 1]))
            ssr = ssr / ssr[2]
            # print(type(round(ssr[1] + image_info.offset_y)))
            kp_img[
                round(ssr[1] + image_info.offset_y).astype(int), round(ssr[0] + image_info.offset_x).astype(int)] = 255.
            kp_img[
                round(dds[1] + image_info.offset_y).astype(int), round(dds[0] + image_info.offset_x).astype(int)] = 255.
        kp_img = kp_img.astype(np.bool) & mask.astype(np.bool)
        kp_img = kp_img[miny:maxy, minx:maxx]
        (kpy, kpx) = np.where(kp_img > 0)
        kp = np.stack([kpx, kpy], axis=0).astype(np.int64)
        kp = kp.T
        print(kp.shape)

        mask = mask[miny:maxy + 1, minx:maxx + 1]
        wd = maxx - minx
        wh = maxy - miny

        x_dim = np.linspace(0, wd, wd + 1)
        y_dim = np.linspace(0, wh, wh + 1)
        (x_, y_) = np.meshgrid(x_dim, y_dim)
        print(x_.shape)
        pixel = np.stack([x_, y_], axis=-1).astype(np.int64)
        print(pixel.shape)
        # todo 计算fc，也即pixel与kp之间的距离等参数
        # todo 获取重叠区域的fc参数
        fc = get_fc(kps=kp, ps=pixel, height=src_img.shape[0], width=src_img.shape[1])
        mask = mask / 255
        fc_img = fc * mask

        # todo 准备进行边缘检测
        gray_warp_src = cv2.cvtColor(src_warp, cv2.COLOR_BGR2GRAY)
        gray_warp_dst = cv2.cvtColor(dst_warp, cv2.COLOR_BGR2GRAY)
        warp_src_edge = cv2.Canny(gray_warp_src, 200, 300)
        warp_dst_edge = cv2.Canny(gray_warp_dst, 200, 300)

        edge_map = np.abs(warp_src_edge - warp_dst_edge)

        x = graph_cut(edge_map, pixel)


        # different_map = np.bitwise_xor(warp_dst_edge.astype(np.bool), warp_src_edge.astype(np.bool))
        # different_map = different_map[miny:maxy + 1, minx:maxx + 1]
        # infinity = np.ones((maxy - miny + 1, maxx - minx + 1)) * 1e10
        # fc_img = fc_img * different_map
        #
        # # todo x与y方向相邻节点的能量值，在mask之外也即不相交的部分设置为无穷大
        # mask = mask / 255
        # fc_img_x = fc_img + np.roll(fc_img, shift=-1, axis=1)
        # fc_img_x = fc_img_x * mask + (~mask.astype(np.bool)) * np.roll(infinity, shift=-1, axis=1)
        # fc_img_x[:, -1] = 0
        #
        # fc_img_y = fc_img + np.roll(fc_img, shift=-1, axis=0)
        # fc_img_y = fc_img_y * mask + (~mask.astype(np.bool)) * np.roll(infinity, shift=-1, axis=0)
        # fc_img_y[-1, :] = 0
        #
        # # todo 寻找最佳路径
        # g = maxflow.GraphFloat()
        # nodeids = g.add_grid_nodes(fc_img.shape)
        # structure = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        # g.add_grid_edges(nodeids, fc_img, structure=structure, symmetric=True)
        # left_most = np.concatenate(
        #     (np.arange(fc_img.shape[0]).reshape(1, fc_img.shape[0]), np.zeros((1, fc_img.shape[0])))).astype(np.uint64)
        # left_most = np.ravel_multi_index(left_most, fc_img.shape)
        # g.add_grid_tedges(left_most, np.inf, 0)
        #
        # right_most = np.concatenate(
        #     (np.arange(fc_img.shape[0]).reshape(1, fc_img.shape[0]),
        #      np.ones((1, fc_img.shape[0])) * (np.size(fc_img, 1) - 1))).astype(
        #     np.uint64)
        # right_most = np.ravel_multi_index(right_most, fc_img.shape)
        # g.add_grid_tedges(right_most, 0, np.inf)
        #
        # x = g.maxflow()
        # print("edgecost is" + str(x))
        #
        # cv2.imshow("warp_src_edge", warp_src_edge)
        # cv2.imshow("warp_dst_edge", warp_dst_edge)
        # cv2.imshow("different_map", different_map.astype(np.uint8) * 255)
        # cv2.imshow("fc_img", fc_img.astype(np.uint8))
        # cv2.imshow("kp_img", cv2.UMat(kp_img.astype(np.uint8) * 255))
