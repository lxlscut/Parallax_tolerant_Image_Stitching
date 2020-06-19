import cv2
import numpy as np
import time
from PTIS.find_best_homo import get_best_homography
from PTIS.find_best_homo1 import get_best_homography1
from feature.vlfeat import Match
from util.draw import Draw
from util.ransac import RANSAC

if __name__ == '__main__':
    src = cv2.imread("image/lib1.jpg")
    dst = cv2.imread("image/lib2.jpg")

    # cv2.imshow("src", src)
    # cv2.imshow("dst", dst)
    """获取匹配点"""
    match = Match(src, dst)
    match.getInitialFeaturePairs()
    src_point = match.src_match
    dst_point = match.dst_match
    # print(src_point.shape)
    """RANSAC 对特征点对进行筛选"""
    ransac = RANSAC(src_point=src_point, dst_point=dst_point)
    final_num, final_src_point, final_dst_point = ransac.ransac()
    draw = Draw()
    src = draw.draw(src=src, src_point=final_src_point)
    dst = draw.draw(src=dst, src_point=final_dst_point)
    # cv2.imshow("src_p", src)
    # cv2.imshow("dst_p", dst)
    start = time.time()
    homography, similar_matrix = get_best_homography(src_point=src_point, dst_point=dst_point, src_img=src, dst_img=dst)
    end = time.time()
    print(end - start)
    cv2.waitKey(0)
