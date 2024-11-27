import cv2
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

def plot_gray_distribution_frame_by_frame(**kwargs):
    input_filename = kwargs.get('input_filename', '')
    output_filename = kwargs.get('output_filename', '')

    # 创建一个 PDF 文件
    # with PdfPages("microscope/track_detection/TEST/gray_distribution_all_pages.pdf") as pdf:
    with PdfPages(output_filename) as pdf:
        with tifffile.TiffFile(input_filename) as tif:
            num_pages = len(tif.pages)
            print(f"Number of pages in the TIFF file: {num_pages}")
            print("="*50)
    
            for i, page in enumerate(tif.pages):
                if i < 5 or i >= num_pages - 5:
                    continue
    
                # if i > 10:
                #     break
                
                img = page.asarray()
    
                # 展平图像为一维数组
                pixel_values = img.flatten()
    
                # 绘制灰度值的直方图
                plt.figure(figsize=(8, 6))
                plt.hist(pixel_values, bins=50, color='gray', alpha=0.7)
    
                # 设置 y 轴为对数尺度
                plt.yscale('log')
                
                plt.title("Pixel Intensity Distribution - Frame {}\n{}".format(i+1, experiment_type))
                plt.xlabel("Pixel Intensity (Gray Level)")
                plt.ylabel("Count")
                plt.grid(True)
    
                # 将当前图形添加到 PDF 文件中
                pdf.savefig()  # 保存当前图形到 PDF
                plt.close()    # 关闭当前图形
    
            # print(f"Saved histogram for page {i + 1} to PDF")



def plot_specific_frame_gray_distribution(**kwargs):
    input_filename = kwargs.get('input_filename', '')
    output_filename = kwargs.get('output_filename', '')

    # 创建一个 PDF 文件
    # with PdfPages("microscope/track_detection/TEST/gray_distribution_all_pages.pdf") as pdf:
    with PdfPages(output_filename) as pdf:
        with tifffile.TiffFile(input_filename) as tif:
            num_pages = len(tif.pages)
            print(f"Number of pages in the TIFF file: {num_pages}")
            
            for i, page in enumerate(tif.pages):
                if i < 5 or i >= num_pages - 5:
                    continue

                img = page.asarray()
                
                if np.max(img) > 5000:

                    high_value_positions = np.where(img > 5000)

                    # high_value_positions 是一个包含两个元素的元组，第一个元素是行索引，第二个元素是列索引
                    rows, cols = high_value_positions

                    # 打印这些位置的行列索引
                    print("Positions of pixels greater than 5000:")
                    for row, col in zip(rows, cols):
                        print(f"Row: {row}, Column: {col}")

                    print("="*50)

                    plt.figure(figsize=(8, 6))
                    # plt.imshow(img, cmap='gray', interpolation='nearest')
                    # plt.colorbar()  # 可选，显示色条
                    # plt.imshow(img, cmap='hot', interpolation='nearest')  # 'hot' 是一种常见的伪彩色映射
                    # plt.colorbar()
                    # img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))

                    # 绘制图像
                    plt.imshow(img, cmap='gray')
                    plt.colorbar()
                    
                    plt.title("Image - Frame {}\n{}".format(i+1, experiment_type))
                    plt.xlim(0, 1400)
                    plt.ylim(0, 1400)
                    plt.xlabel("pixel")
                    plt.ylabel("pixel")
                    # plt.axis('off')  # 可选，关闭坐标轴

                    # 将当前图像保存到 PDF 文件
                    pdf.savefig()  # 保存当前图形到 PDF
                    plt.close()    # 关闭当前图形
                
                    # 展平图像为一维数组
                    pixel_values = img.flatten()
                
                    # 绘制灰度值的直方图
                    plt.figure(figsize=(8, 6))
                    plt.hist(pixel_values, bins=50, color='gray', alpha=0.7)
                
                    # 设置 y 轴为对数尺度
                    plt.yscale('log')
                
                    plt.title("Pixel Intensity Distribution - Frame {}\n{}".format(i+1, experiment_type))
                    plt.xlabel("Pixel Intensity (Gray Level)")
                    plt.ylabel("Count")
                    plt.grid(True)
                
                    # 将当前图形添加到 PDF 文件中
                    pdf.savefig()  # 保存当前图形到 PDF
                    plt.close()    # 关闭当前图形
                

                
            
def plot_all_frames_gray_distribution(**kwargs):
    experiment_type = kwargs.get('experiment_type', '')
    input_filename = kwargs.get('input_filename', '')
    output_filename = kwargs.get('output_filename', '')

    # 创建一个 PDF 文件
    with PdfPages(output_filename) as pdf:
        with tifffile.TiffFile(input_filename) as tif:
            num_pages = len(tif.pages)
            print(f"Number of pages in the TIFF file: {num_pages}")

            # 初始化全局最小值和最大值
            global_min_value = float('inf')
            global_max_value = float('-inf')

            # 初始化一个直方图累加器
            total_histogram = None

            # 遍历每一帧，计算直方图并累加，同时更新最小值和最大值
            for i, page in enumerate(tif.pages):
                if i < 5 or i >= num_pages - 5:
                    continue

                img = page.asarray()

                # 获取当前帧的最小值和最大值
                current_min = np.min(img)
                current_max = np.max(img)

                # 更新全局最小值和最大值
                global_min_value = min(global_min_value, current_min)
                global_max_value = max(global_max_value, current_max)

            print(f"Global Min Value: {global_min_value}, Global Max Value: {global_max_value}")
            print("="*50)

            # 再次处理每一帧，计算灰度直方图并累加
            for i, page in enumerate(tif.pages):
                if i < 5 or i >= num_pages - 5:
                    continue

                img = page.asarray()

                # 计算当前帧的灰度直方图
                frame_histogram, bin_edges = np.histogram(img.flatten(), bins=100, range=(global_min_value, global_max_value))

                # 如果是第一帧，初始化总的直方图
                if total_histogram is None:
                    total_histogram = frame_histogram
                else:
                    total_histogram += frame_histogram

            # 绘制所有帧的总灰度分布直方图
            plt.figure(figsize=(8, 6))
            plt.bar(bin_edges[:-1], total_histogram, width=np.diff(bin_edges), color='gray', alpha=0.7)

            # 设置 y 轴为对数尺度
            plt.yscale('log')
            
            plt.title("Pixel Intensity Distribution - All Frames\n{}".format(experiment_type))
            plt.xlabel("Pixel Intensity (Gray Level)")
            plt.ylabel("Count")

            plt.grid(True)

            # 将总的灰度分布直方图添加到 PDF 文件中
            pdf.savefig()  # 保存当前图形到 PDF
            plt.close()    # 关闭当前图形



def plot_diff_experiment_all_frames_gray_distribution():

    plot_experiment_types = ["microscope.2h", "beta_crystal_oil_microscope.2h", "crystal_oil_microscope.2h", "alpha_crystal_oil_microscope.2h", "alpha_crystal_microscope.2h"]

    hist_range = (0, 5000)
    bin_count = 50
    
    # 创建一个 PDF 文件
    with PdfPages("microscope/track_detection/TEST/range_0_5000.hist_and_line.with_alpha_2h.pdf") as pdf:
      
        total_histograms = []
        bin_edges_list = []
        
        for experiment_type in plot_experiment_types:
            with tifffile.TiffFile("microscope/track_detection/TEST/{}.tif".format(experiment_type)) as tif:
                num_pages = len(tif.pages)
                print(f"Processing file: {experiment_type}, Number of pages: {num_pages}")
    
                total_histogram = None
                
                # 计算灰度直方图并累加
                for i, page in enumerate(tif.pages):
                    if i < 5 or i >= num_pages - 5:
                        continue
    
                    img = page.asarray()
                    
                    # 计算当前帧的灰度直方图
                    frame_histogram, bin_edges = np.histogram(img.flatten(), bin_count, hist_range)
    
                    # 如果是第一帧，初始化总的直方图
                    if total_histogram is None:
                        total_histogram = frame_histogram
                    else:
                        total_histogram += frame_histogram
    
                    # 将当前文件的直方图和 bin_edges 保存到列表中
    
                total_histograms.append(total_histogram)
                bin_edges_list.append(bin_edges)

        
        # hist
        plt.figure(figsize=(10, 6))

        for i, total_histogram in enumerate(total_histograms):
            # 归一化
            # total_histogram_normalized = total_histogram / total_histogram.sum()

            plt.hist(bin_edges_list[i][:-1], bins=bin_edges_list[i], weights=total_histogram, label=plot_experiment_types[i], alpha=0.7, histtype='step')

        plt.yscale('log')
        plt.title("Pixel Intensity Distribution Comparison")
        plt.xlabel("Pixel Intensity (Gray Level)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
    
        pdf.savefig() 
        plt.close()   
     

        # line
        plt.figure(figsize=(10, 6))

        for i, total_histogram in enumerate(total_histograms):
            plt.plot(bin_edges_list[i][:-1], total_histogram, label=plot_experiment_types[i], alpha=0.7)

        plt.yscale('log')
        plt.title("Pixel Intensity Distribution Comparison")
        plt.xlabel("Pixel Intensity (Gray Level)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        pdf.savefig() 
        plt.close()  

        


def plot_average_all_frames(**kwargs):
    output_filename = kwargs.get('output_filename', '')

    plot_experiment_types = ["microscope.2h", "beta_crystal_oil_microscope.2h", "crystal_oil_microscope.2h", "alpha_crystal_oil_microscope.2h", "alpha_crystal_microscope.2h"]

    with PdfPages(output_filename) as pdf:
        for experiment_type in plot_experiment_types:
            with tifffile.TiffFile("microscope/track_detection/TEST/{}.tif".format(experiment_type)) as tif:
                num_pages = len(tif.pages)
                print(f"Processing file: {experiment_type}, Number of pages: {num_pages}")
                                
                sum_img = None
                num_images = 0
                
                for i, page in enumerate(tif.pages):
                    if i < 5 or i >= num_pages - 5:
                        continue
    
                    img = page.asarray()
    
                    if sum_img is None:
                        sum_img = np.zeros_like(img, dtype=np.float64)
                    
                    # 累加图像数据
                    sum_img += img
                    num_images += 1
                
                if num_images > 0:
                    # 计算平均图像
                    avg_img = sum_img / num_images
                    
                # 绘制平均图像并保存到PDF
                # plt.imshow(avg_img, cmap='hot', vmin=0, vmax=300)
                plt.imshow(avg_img, cmap='hot')
                plt.xlim(0, 1400)
                plt.ylim(0, 1400)
                plt.xlabel("pixel")
                plt.ylabel("pixel")
              
                cbar = plt.colorbar()
                cbar.set_label("Pixel Intensity (Gray Level)")
                plt.title("Average Image\n{}".format(experiment_type))
                pdf.savefig()  # 保存到PDF
                plt.close()
    


def merge_tiff_files(input_filename1, input_filename2, output_filename):
    # 读取第一个 TIFF 文件，并排除最后 5 帧
    with tifffile.TiffFile(input_filename1) as tif1:
        images1 = [page.asarray() for i, page in enumerate(tif1.pages) if i < len(tif1.pages) - 5]

    # 读取第二个 TIFF 文件，并排除前 5 帧
    with tifffile.TiffFile(input_filename2) as tif2:
        images2 = [page.asarray() for i, page in enumerate(tif2.pages) if i >= 5]

    # 合并两个文件中的所有图像
    all_images = images1 + images2

    # 将合并后的图像保存到一个新的 TIFF 文件
    with tifffile.TiffWriter(output_filename) as tif_writer:
        for img in all_images:
            tif_writer.save(img)

    print(f"合并后的文件保存为: {output_filename}")

    
#last

                    
# experiment_types = ["microscope.2h", "beta_crystal_oil_microscope.2h", "crystal_oil_microscope.2h", "alpha_crystal_oil_microscope.2h", "alpha_crystal_microscope.2h"]

# for experiment_type in experiment_types:
#     print("experiment type = {}".format(experiment_type))
#     plot_gray_distribution_frame_by_frame(
#         input_filename = "microscope/track_detection/TEST/{}.tif".format(experiment_type),
#         output_filename = "microscope/track_detection/TEST/{}.gray_distribution_frame_by_frame.pdf".format(experiment_type)
#     )


# for experiment_type in experiment_types:
#     print("experiment type = {}".format(experiment_type))
#     plot_all_frames_gray_distribution(
#         experiment_type = experiment_type,
#         input_filename = "microscope/track_detection/TEST/{}.tif".format(experiment_type),
#         output_filename = "microscope/track_detection/TEST/{}.gray_distribution_all_frames.pdf".format(experiment_type)
#     )

# experiment_types = ["beta_crystal_oil_microscope.2h", "crystal_oil_microscope.2h", "alpha_crystal_oil_microscope.2h"]

# for experiment_type in experiment_types:
#     print("experiment type = {}".format(experiment_type))
#     plot_specific_frame_gray_distribution(
#         experiment_type = experiment_type,
#         input_filename = "microscope/track_detection/TEST/{}.tif".format(experiment_type),
#         output_filename = "microscope/track_detection/TEST/{}.large_intensity.pdf".format(experiment_type)
#     )


# plot_diff_experiment_all_frames_gray_distribution()

# input_filename1 = "/lustre/neutrino/chenzhangming/microscope/track_detection/TEST/microscope.0.5h.tif"
# input_filename2 = "/lustre/neutrino/chenzhangming/microscope/track_detection/TEST/microscope.1.5h.tif"
# output_filename = "/lustre/neutrino/chenzhangming/microscope/track_detection/TEST/microscope.2h.tif"

# merge_tiff_files(input_filename1, input_filename2, output_filename)


plot_average_all_frames(
        output_filename = "microscope/track_detection/TEST/average_all_frames.pdf"
    )


# plot_diff_experiment_all_frames_gray_distribution()


# ======================================= opencv =========================================

# # 视频文件路径
# video_path = "/lustre/neutrino/chenzhangming/microscope/track_detection/TEST/without_LUTs.mp4"  # 替换为你的实际视频文件路径

# # 打开视频文件
# cap = cv2.VideoCapture(video_path)

# # 获取视频的总帧数
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# print("total frames = {}".format(total_frames))

# # 读取每一帧
# for frame_index in range(100):
#     # 读取帧
#     ret, frame = cap.read()
    
#     if not ret:
#         print(f"无法读取第{frame_index + 1}帧，视频结束或读取出错")
#         break

#     # print("视频帧的形状:", frame.shape)

#     # # 判断视频是否是灰度图像或彩色图像
#     # if len(frame.shape) == 2:
#     #     print("这是灰度视频。")
#     # elif len(frame.shape) == 3 and frame.shape[2] == 3:
#     #     print("这是RGB彩色视频。")


#     # 将帧转换为灰度图
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     print(f"第{frame_index + 1}帧灰度值：")

#     # rows = gray_frame.shape[0]
#     # columns = gray_frame.shape[1]

#     # print("rows = {}, columns = {}".format(rows, columns))
    
#     # 打印每个像素的灰度值
#     for i in range(gray_frame.shape[0]):  # 行数
#         for j in range(gray_frame.shape[1]):  # 列数
#             # 获取灰度值
#             pixel_value = gray_frame[i, j]
#             if pixel_value > 0:
#                 print(f"({i},{j}): {pixel_value}", end="  ")

#     print()
#     print("="*50)

# # # 释放视频资源
# cap.release()
