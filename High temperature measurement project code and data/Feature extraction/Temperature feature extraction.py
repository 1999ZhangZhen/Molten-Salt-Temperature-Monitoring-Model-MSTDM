import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import os


# cap = cv2.VideoCapture('D:\\temperature\\label_video_1114_two\\1030-1035.avi')

# def foreground_extraction(image_path):
#
#     color_image = cv2.imread(image_path)
#     gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#     ret1, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     contours = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
#     max_contour = max(contours, key=cv2.contourArea)
#     result = np.zeros_like(binary_otsu)
#     one_foreground = cv2.drawContours(result, [max_contour], -1, 255, thickness=cv2.FILLED)
#     result_eroded = cv2.erode(one_foreground, np.ones((25, 25), np.uint8), iterations=1)
#     final_foreground_color = cv2.bitwise_and(color_image, color_image, mask=result_eroded)
#     center_x, center_y = max_contour.mean(axis=0).astype(int).ravel()
#     cv2.circle(final_foreground_color, (center_x, center_y), 50, (0, 0, 0), -1)
#     final_foreground_color_median = final_foreground_color
#     final_foreground_gray_median = cv2.cvtColor(final_foreground_color_median, cv2.COLOR_RGB2GRAY)
#     final_foreground_HSV_median = cv2.cvtColor(final_foreground_color_median, cv2.COLOR_RGB2HSV)

#     return final_foreground_color_median, final_foreground_gray_median, final_foreground_HSV_median

def process_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    # 获得的7个特征，与温度相关
    B_whole = []
    G_whole = []
    R_whole = []
    gray_whole= []

    H_whole = []
    S_whole = []
    V_whole = []
    l_whole = []
    a_whole = []
    b_whole = []

    R_G_whole = []
    R_B_whole = []
    G_B_whole = []

    H_S_whole = []
    H_V_whole = []
    S_V_whole = []

    l_a_whole = []
    l_b_whole = []
    a_b_whole = []

    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        i += 1
        print(i)
        if not ret:
            break

        color_image = frame
        # cv2.waitKey()
        gray_zz = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # print(gray.shape)

        # cv2.imshow("zz", gray)
        # cv2.waitKey()

        ret1, binary_otsu = cv2.threshold(gray_zz, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        max_contour = max(contours, key=cv2.contourArea)
        result = np.zeros_like(binary_otsu)
        one_foreground = cv2.drawContours(result, [max_contour], -1, 255, thickness=cv2.FILLED)

        result_eroded = cv2.erode(one_foreground, np.ones((25, 25), np.uint8), iterations=1)
        final_foreground_color = cv2.bitwise_and(color_image, color_image, mask=result_eroded)

        one_foreground_3D = final_foreground_color.copy()

        center_x, center_y = max_contour.mean(axis=0).astype(int).ravel()
        cv2.circle(final_foreground_color, (center_x, center_y), 60, (0, 0, 0), -1)

        final_foreground_color_median = final_foreground_color

        # final_foreground_color_median = cv2.medianBlur(final_foreground_color_median, 5)     # 中值滤波
        # final_foreground_color_median = cv2.blur(final_foreground_color_median, (5, 5))  # 均值滤波

        final_foreground_gray_median = cv2.cvtColor(final_foreground_color_median, cv2.COLOR_RGB2GRAY)
        final_foreground_HSV_median = cv2.cvtColor(final_foreground_color_median, cv2.COLOR_RGB2HSV)
        final_foreground_lab_median = cv2.cvtColor(final_foreground_color_median, cv2.COLOR_BGR2LAB)
        ret, final_foreground_erzhi_median = cv2.threshold(final_foreground_gray_median, 1, 255, cv2.THRESH_BINARY)


        if not B_whole:
            B_whole.append(0)
            G_whole.append(0)
            R_whole.append(0)
            gray_whole.append(0)
            H_whole.append(0)
            S_whole.append(0)
            V_whole.append(0)
            l_whole.append(0)
            a_whole.append(0)
            b_whole.append(0)

            R_G_whole.append(0)
            R_B_whole.append(0)
            G_B_whole.append(0)
            H_S_whole.append(0)
            H_V_whole.append(0)
            S_V_whole.append(0)
            l_a_whole.append(0)
            l_b_whole.append(0)
            a_b_whole.append(0)

        else:
            feihei_pixels = np.where((np.any(final_foreground_color_median != [0, 0, 0], axis=-1)))
            non_black_pixels_rgb = final_foreground_color_median[feihei_pixels]
            B = non_black_pixels_rgb[:, 0]
            G = non_black_pixels_rgb[:, 1]
            R = non_black_pixels_rgb[:, 2]
            B_whole.append(np.mean(B))
            G_whole.append(np.mean(G))
            R_whole.append(np.mean(R))

            gray_feihei = np.where(final_foreground_gray_median != 0)
            gray = final_foreground_gray_median[gray_feihei]
            gray_whole.append(np.mean(gray))

            feihei_pixels2 = np.where((np.any(final_foreground_HSV_median != [0, 0, 0], axis=-1)))
            non_black_pixels_hsv = final_foreground_HSV_median[feihei_pixels2]
            H = non_black_pixels_hsv[:, 0]
            S = non_black_pixels_hsv[:, 1]
            V = non_black_pixels_hsv[:, 2]
            H_whole.append(np.mean(H))
            S_whole.append(np.mean(S))
            V_whole.append(np.mean(V))

            feihei_pixels3 = np.where((np.any(final_foreground_lab_median != [0, 0, 0], axis=-1)))
            non_black_pixels_lab = final_foreground_lab_median[feihei_pixels3]
            l = non_black_pixels_lab[:, 0]
            a = non_black_pixels_lab[:, 1]
            b = non_black_pixels_lab[:, 2]
            l_whole.append(np.mean(l))
            a_whole.append(np.mean(a))
            b_whole.append(np.mean(b))

            R_G_whole.append(np.mean(R)/np.mean(G))
            R_B_whole.append(np.mean(R)/np.mean(B))
            G_B_whole.append(np.mean(G)/np.mean(B))

            H_S_whole.append(np.mean(H)/np.mean(S))
            H_V_whole.append(np.mean(H)/np.mean(V))
            S_V_whole.append(np.mean(S)/np.mean(V))

            l_a_whole.append(np.mean(l)/np.mean(a))
            l_b_whole.append(np.mean(l)/np.mean(b))
            a_b_whole.append(np.mean(a)/np.mean(b))


            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes[0, 0].imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("color_image")
            axes[0, 0].axis('off')

            gray_image = cv2.cvtColor(gray_zz, cv2.COLOR_GRAY2RGB)
            axes[0, 1].imshow(gray_image)
            axes[0, 1].set_title("gray_image")
            axes[0, 1].axis('off')

            axes[0, 2].imshow(binary_otsu, cmap='gray')
            axes[0, 2].set_title("gray_2D")
            axes[0, 2].axis('off')

            axes[0, 3].imshow(one_foreground, cmap='gray')
            axes[0, 3].set_title("one_foreground")
            axes[0, 3].axis('off')

            axes[0, 4].imshow(result_eroded, cmap='gray')
            axes[0, 4].set_title("one_foreground_eroded")
            axes[0, 4].axis('off')

            axes[1, 0].imshow(cv2.cvtColor(one_foreground_3D, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title("one_foreground_3D")
            axes[1, 0].axis('off')

            axes[1, 1].imshow(cv2.cvtColor(final_foreground_color_median, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title("final_foreground_color_lo")
            axes[1, 1].axis('off')

            axes[1, 2].imshow(final_foreground_HSV_median)
            axes[1, 2].set_title("final_foreground_HSV")
            axes[1, 2].axis('off')

            axes[1, 3].imshow(final_foreground_lab_median)
            axes[1, 3].set_title("final_foreground_lab")
            axes[1, 3].axis('off')

            axes[1, 4].imshow(final_foreground_erzhi_median, cmap='gray')
            axes[1, 4].set_title("final_foreground_erzhi_median")
            axes[1, 4].axis('off')

            plt.tight_layout()
            plt.show()
            plt.close()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 在这里添加保存 .xlsx 文件的代码
    output_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(video_path))[0] + '.xlsx')
    number = range(len(B_whole))
    def pd_toexcel(number, B_whole, G_whole, R_whole, gray_whole, H_whole, S_whole, V_whole, l_whole, a_whole, b_whole,
                   R_G_whole, R_B_whole, G_B_whole, H_S_whole, H_V_whole, S_V_whole, l_a_whole, l_b_whole, a_b_whole, filename):
        dfData = {
            '序号':number,
            'B均值':B_whole,
            'G均值':G_whole,
            'R均值':R_whole,
            '灰度均值':gray_whole,
            'H均值':H_whole,
            'S均值':S_whole,
            'V均值':V_whole,
            'l均值': l_whole,
            'a均值': a_whole,
            'b均值': b_whole,

            'R/G均值': R_G_whole,
            'R/B均值': R_B_whole,
            'G/B均值': G_B_whole,
            'H/S均值': H_S_whole,
            'H/V均值': H_V_whole,
            'S/V均值': S_V_whole,
            'l/a均值': l_a_whole,
            'l/b均值': l_b_whole,
            'a/b均值': a_b_whole,
        }
        df = pd.DataFrame(dfData)
        df.to_excel(filename, index=False)

    pd_toexcel(number, B_whole, G_whole, R_whole, gray_whole, H_whole, S_whole, V_whole, l_whole, a_whole, b_whole,
               R_G_whole, R_B_whole, G_B_whole, H_S_whole, H_V_whole, S_V_whole, l_a_whole, l_b_whole, a_b_whole, output_filename)

    cap.release()
    # cv2.destroyAllWindows()

# 遍历文件夹中的所有.avi文件
input_folder = r'D:\Molten salt temperature prediction\11'
output_folder = r'D:\Molten salt temperature prediction\test'

for filename in os.listdir(input_folder):
    if filename.endswith(".avi"):
        video_path = os.path.join(input_folder, filename)
        process_video(video_path, output_folder)
