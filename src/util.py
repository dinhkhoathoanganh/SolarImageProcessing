import re

from UserInput import *

import os as os
from os import path
import cv2
import numpy as np
import statistics as stats


def create_output_folder(image_file_src):
    # Create output folder
    if not path.exists(image_folder + 'output/'):
        os.mkdir(image_folder + 'output/')
        print("Directory output created")

    folder_path_img = image_folder + 'output/' + image_file_src + '/'

    if not path.exists(folder_path_img):
        os.mkdir(folder_path_img)
        print("Directory created " + folder_path_img)


def read_img_list():
    if not path.exists(image_folder + 'input/'):
        print("Directory not exist!")
        return None

    img_list = [f.replace(file_extension, '') for f in os.listdir(image_folder + 'input/') if f.endswith(file_extension)]

    return img_list


def pre_process_img(img):
    # Rotate and resize
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))  # blur the image
    ret, thresh = cv2.threshold(blur, 50, 250, cv2.THRESH_BINARY)

    # Finding contours for the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    # drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    longestarray = -1
    position = -1

    for i in range(len(hull)):
        if len(hull[i]) > longestarray:
            longestarray = len(hull[i])
            position = i

    points = hull[position]

    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    res = cv2.bitwise_and(img, img, mask=mask)

    # create the white background of the same size of original image
    wbg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(wbg, wbg, mask=mask)

    dst = wbg + res

    cnt = points
    canvas = dst
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())

    if len(approx_corners) != 4:
        approx_corners = [(0, 0), (WIDTH, 0), (0, HEIGHT), (WIDTH, HEIGHT)]

    return [img, approx_corners]


def transform_img(src_img, approx_corners):
    new = {sum(a): i for i, a in enumerate(approx_corners)}

    indexes = [0, 1, 2, 3]
    lt = new[min(new.keys())]
    rb = new[max(new.keys())]
    indexes.remove(lt)
    indexes.remove(rb)

    if approx_corners[indexes[0]][0] > approx_corners[indexes[0]][1]:
        lb = approx_corners[indexes[0]]
        rt = approx_corners[indexes[1]]
    else:
        lb = approx_corners[indexes[1]]
        rt = approx_corners[indexes[0]]

    lt = approx_corners[lt]
    rb = approx_corners[rb]

    dst_corners = np.float32([(0, 0), (WIDTH, 0), (0, HEIGHT), (WIDTH, HEIGHT)])
    src_corners = np.float32([lt, lb, rt, rb])

    H = cv2.getPerspectiveTransform(src_corners, dst_corners)

    un_warped = cv2.warpPerspective(src_img, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)

    # plt.subplot(121), plt.imshow(img), plt.title('Input')
    # plt.subplot(122), plt.imshow(un_warped), plt.title('Output')
    # plt.show()

    return un_warped


def process_contour(src_img):
    # Get contours
    thresh = cv2.adaptiveThreshold(src_img, 245, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,
                                   2)  # cv2.threshold(src_img,50,500,cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts


def count_contour(src_img, cnts, min_area, max_area, exp_splices, mode):
    out_ROI, xx, yy, ww, hh = [], [], [], [], []
    w_ave = HEIGHT / exp_splices[0]
    h_ave = WIDTH / exp_splices[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            # print(area)
            x, y, w, h = cv2.boundingRect(c)

            if mode == 'test': # or mode == 'train': #FIXX
                # interpolation of missing splices
                xx.append(x)
                yy.append(y)
                ww.append(w)
                hh.append(h)

            out_ROI.append(src_img[y:y + h, x:x + h])
            cv2.rectangle(src_img, (x, y), (x + w, y + h), (30, 255, 10), 2)

    return [out_ROI, src_img, xx, yy, ww, hh]


def cell_splice(src_un_warped, min_area, max_area, exp_splices, blur_type='gaussian', mode='train', blur_threshold=5,
                delta=10000):

    print("RUNNING CELL SPICING")
    # # cell_splice(un_warped, 3000, 10000, SPLICES, blur_type = 'gaussian', mode = 'test', blur_threshold = 5)
    no_splices = exp_splices[0] * exp_splices[1]

    while True:

        # Adjust blur_threshold
        if blur_threshold > 21 or blur_threshold < 1 or delta == 0:
            return [ROI, dst_un_warped, delta, blur_threshold]

        print("blur_threshold = " + str(blur_threshold))
        dst_un_warped = src_un_warped.copy()

        # Process image
        if blur_type == 'median':
            blur = cv2.medianBlur(dst_un_warped, blur_threshold)
        elif blur_type == 'box':
            blur = cv2.medianBlur(dst_un_warped, blur_threshold)
        else:
            blur = cv2.GaussianBlur(dst_un_warped, (5, blur_threshold), 0)

        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        gray = cv2.cvtColor(sharpen, cv2.COLOR_BGR2GRAY)

        # Get contours
        process_cnts = process_contour(gray)

        # Count splices
        [ROI, dst_un_warped, xx, yy, ww, hh] = count_contour(dst_un_warped, process_cnts, min_area, max_area,
                                                             exp_splices, mode)


        # For test mode
        if mode == 'test':
            xx.reverse()
            yy.reverse()
            ww.reverse()
            hh.reverse()
            return [ROI, dst_un_warped, delta, blur_threshold, xx, yy, ww, hh]

        # Adjust blur_threshold
        prev_delta = delta
        delta = len(ROI) - no_splices

        print("delta: " + str(delta) + " " + str(prev_delta))

        if abs(delta) >= abs(prev_delta):  # and prev_delta*delta < 0:
            print("Optimized = " + str(prev_blur_threshold))
            print("Done")
            return [prev_ROI, prev_dst_un_warped, prev_delta, prev_blur_threshold]

        prev_blur_threshold = blur_threshold

        if delta > 0:
            print("Decrease")
            blur_threshold -= 2
        elif delta < 0:
            print("Increase")
            blur_threshold += 2

        prev_ROI = ROI.copy()
        prev_dst_un_warped = dst_un_warped.copy()


def train_cell_splicing(src_un_warped):
    blur_tp = 'gaussian'
    [_, _, dlt, blur_thres] = cell_splice(src_un_warped, MIN_CELL_AREA, MAX_CELL_AREA, SPLICES, blur_type=blur_tp,
                                          mode='train', blur_threshold=5)

    # delta not 0 yet, try the other type of blur
    if dlt != 0:
        [_, _, dlt2, blur_thres2] = cell_splice(src_un_warped, MIN_CELL_AREA, MAX_CELL_AREA, SPLICES, 'median',
                                                mode='train', blur_threshold=5)
        if abs(dlt2) < abs(dlt):
            blur_tp = 'median'
            blur_thres = blur_thres2

    [ROI_list, new_un_warped, dlt, blur_thres, xx, yy, ww, hh] = cell_splice(src_un_warped, MIN_CELL_AREA, MAX_CELL_AREA, SPLICES,
                                                                             blur_type=blur_tp, mode='test',
                                                                             blur_threshold=blur_thres)

    return [ROI_list, new_un_warped, dlt, blur_thres, xx, yy, ww, hh]


def sort_corners(xx, yy, h_err):
    xx_sorted, yy_sorted = [], []
    last_i = 0
    for i in range(1, len(yy)):
        delta = yy[i] - yy[i - 1]
        if delta > h_err or i == len(yy) - 1:  # end of row
            list1, list2 = zip(*sorted(zip(xx[last_i:i], yy[last_i:i])))
            xx_sorted.extend(list(list1))
            yy_sorted.extend(list(list2))
            last_i = i
    return [xx_sorted, yy_sorted]


def get_ave_and_error(h, w):
    h_ave = stats.median(h)  # y
    w_ave = stats.median(w)  # x

    if round(round(WIDTH / SPLICES[1]) / w_ave) != 1:
        print("w_ave median too large = " + str(w_ave))
        w_ave = round(WIDTH / SPLICES[1])

    if round(round(HEIGHT / SPLICES[0]) / h_ave) != 1:
        print("h_ave median too large = " + str(h_ave))
        h_ave = round(HEIGHT / SPLICES[0])

    h_err = ERROR_MARGIN_SPLICING * h_ave
    w_err = ERROR_MARGIN_SPLICING * w_ave

    return [h_ave, w_ave, h_err, w_err]


def splice_missing_cells(xx, yy, ww, hh):
    print("CLEANING MISSING CELL SPICING")

    [h_ave, w_ave, h_err, w_err] = get_ave_and_error(hh, ww)
    [xx, yy] = sort_corners(xx, yy, h_err)

    # Go horizontally to find missing splices
    idx_current = 0
    w_delta = 0.17 * w_ave
    h_delta = 0.17 * h_ave

    for y_idx in range(0, SPLICES[0]):
        prev_x = -w_ave
        for x_idx in np.arange(0, SPLICES[1]):
            # print("# " + str(idx_current))
            if idx_current < len(xx):
                if x_idx != 0 and ((x_idx % (SPLICES[1] - 1) == 0) or (
                        yy[idx_current] - yy[idx_current - 1] > h_err)):  # last cell of row
                    error = (WIDTH - w_ave) - xx[idx_current]
                else:
                    error = xx[idx_current] - (prev_x + w_ave)
            else:
                if x_idx != 0:
                    error = (WIDTH - w_ave) - xx[idx_current - 1]
                else:
                    error = WIDTH

            if (abs(error) < w_err) and idx_current < len(xx):  # horizontal split correctly
                if x_idx != 0 and x_idx % (SPLICES[1] - 1) == 0:  # last cell of row
                    prev_x = -w_ave
                else:
                    prev_x = xx[idx_current]

                # Check the w and h
                if (idx_current > len(xx) and abs(ww[idx_current - 1] - w_ave) > w_delta) or (
                        x_idx != 0 and abs(ww[idx_current] - w_ave) > w_delta):
                    ww[idx_current] = ww[idx_current - 1]

                if (idx_current > len(xx) and abs(hh[idx_current - 1] - h_ave) > h_delta) or (
                        x_idx != 0 and abs(hh[idx_current] - h_ave) > h_delta):
                    hh[idx_current] = hh[idx_current - 1]



            else:
                print("missing x = " + str(x_idx) + "; y = " + str(y_idx) + "; idx = " + str(
                    idx_current) + "; err = " + str(
                    error))

                # Interpolate the w and h to insert for new splice
                if x_idx != 0 and abs(ww[idx_current - 1] - w_ave) > w_delta:
                    w_incr = ww[idx_current - 1]
                else:
                    w_incr = w_ave

                if x_idx != 0 and abs(hh[idx_current - 1] - h_ave) > h_delta:
                    h_incr = hh[idx_current - 1]
                else:
                    h_incr = h_ave

                # Interpolate the starting point of new splice
                if x_idx == 0:  # first cell of row
                    prev_x = 0
                    y_pos = yy[idx_current - 1] + h_incr
                else:
                    prev_x = xx[idx_current - 1] + w_incr
                    y_pos = 0 if y_idx == 0 else yy[idx_current - 1]

                if error > 0:  # idx_current-1 split too long
                    xx.insert(idx_current, prev_x)
                    yy.insert(idx_current, y_pos)
                    ww.insert(idx_current, w_incr)
                    hh.insert(idx_current, h_incr)
                else:
                    xx[idx_current] = prev_x  # idx_current-1 split too short
                    yy[idx_current] = y_pos
                    ww[idx_current] = w_incr
                    hh[idx_current] = h_incr

            idx_current += 1

    xx = [int(i) for i in xx]
    yy = [int(i) for i in yy]
    ww = [int(i) for i in ww]
    hh = [int(i) for i in hh]
    print("Done splicing to " + str(len(xx)))

    return [xx, yy, ww, hh]


def draw_splices(src_img, xx, yy, ww, hh):
    out_ROI = []
    dst_img = src_img.copy()

    for c in range(0, len(xx)):
        out_ROI.append(src_img[yy[c]:yy[c] + hh[c], xx[c]:xx[c] + ww[c]])
        cv2.rectangle(dst_img, (xx[c], yy[c]), (xx[c] + ww[c], yy[c] + hh[c]), (30, 255, 10), 2)

    return [out_ROI, dst_img]


def save_splices(image_file_src, ROI_list, new_un_warped, proc_un_warped):
    folder_path = image_folder + 'output/' + image_file_src + '/'

    cv2.imwrite(folder_path + '{}_AFT_TRAIN.png'.format(image_file_src), new_un_warped)
    cv2.imwrite(folder_path + '{}_AFT_CLEAN.png'.format(image_file_src), proc_un_warped)

    image_number = 0
    for r in ROI_list:
        file_name = folder_path + '{}_ROI_{}.png'.format(image_file_src, image_number)
        cv2.imwrite(file_name, r)
        image_number += 1


def run_splicing_image(image_file_src):

    ################################################################################
    # Process Image
    ################################################################################

    print("**** PROCESSING " + image_file_src)

    # Import image
    img = cv2.imread(image_folder + 'input/' + image_file_src + file_extension)

    # Remove background
    [img, approx_corners] = pre_process_img(img)

    # Transform and flatten
    un_warped = transform_img(img, approx_corners)

    # Train and splice each cell
    [ROI_list, new_un_warped, _, _, xx, yy, ww, hh] = train_cell_splicing(un_warped)

    # Interpolate missing cells
    if len(ROI_list) != SPLICES[0] * SPLICES[1]:
        [proc_xx, proc_yy, proc_ww, proc_hh] = splice_missing_cells(xx, yy, ww, hh)
    else:
        [proc_xx, proc_yy, proc_ww, proc_hh] = [xx, yy, ww, hh]

    ################################################################################

    # Results
    proc_img = un_warped.copy()
    [proc_ROI_list, proc_un_warped] = draw_splices(proc_img, proc_xx, proc_yy, proc_ww, proc_hh)

    print("AFTER SPLICE TRAINING")
    print("Splices = " + str(len(ROI_list)))
    print("AFTER CLEANING AND PROCESSING")
    print("Splices = " + str(len(proc_ROI_list)))

    if verbose:
        save_splices(image_file_src, proc_ROI_list, new_un_warped, proc_un_warped)