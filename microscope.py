import cv2
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

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
                # if i < 5 or i >= num_pages - 5:
                #     continue
                
                img = page.asarray()
                if i == 0:
                    print("Image Size:")
                    print(img.shape)
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
                # if i < 5 or i >= num_pages - 5:
                #     continue

                img = page.asarray()

                # 获取当前帧的最小值和最大值
                current_min = np.min(img)
                current_max = np.max(img)

                # 更新全局最小值和最大值
                global_min_value = min(global_min_value, current_min)
                global_max_value = max(global_max_value, current_max)

            print(f"Global Min Value: {global_min_value}, Global Max Value: {global_max_value}")
            print("="*50)

            global_min_value = 0
            global_max_value = 5000
            
            # 再次处理每一帧，计算灰度直方图并累加
            for i, page in enumerate(tif.pages):
                # if i < 5 or i >= num_pages - 5:
                #     continue

                img = page.asarray()

                # 计算当前帧的灰度直方图
                frame_histogram, bin_edges = np.histogram(img.flatten(), bins=100, range=(global_min_value, global_max_value))

                # 如果是第一帧，初始化总的直方图
                if total_histogram is None:
                    total_histogram = frame_histogram
                else:
                    total_histogram += frame_histogram

            # 绘制所有帧的总灰度分布直方图
            plt.figure(figsize=(12, 6))
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

    # plot_experiment_types = ["alpha_exp1min_10min"]

    # plot_experiment_types = ["microscope.2h", "beta_crystal_oil_microscope.2h", "crystal_oil_microscope.2h", "alpha_crystal_oil_microscope.2h", "alpha_crystal_microscope.2h"]
    # plot_experiment_types = ["lightoff_nocooling", "lightoff_watercooling", "lighton_nocooling", "lighton_watercooling"]
    # plot_experiment_types = ["lightoff_nocooling", "lightoff_watercooling", "lightoff_watercooling.test0"]
    # plot_experiment_types = ["nosource_exp1s_2min", "alphasource_exp1s_2min", "betasource_exp1s_2min", "nosource_exp10s_1min", "betasource_exp10s_1min"]
    # plot_experiment_types = ["60x_alphasource_uppersurface", "60x_betasource_uppersurface", "60x_nosource_uppersurface", "60x_film_alphasource_uppersurface", "60x_film_betasource_uppersurface", "60x_film_nosource_uppersurface", "60x_oil_alphasource_uppersurface_lowerbound", "60x_oil_betasource_uppersurface_lowerbound", "60x_oil_nosource_uppersurface_lowerbound", "60x_oil_alphasource_uppersurface_upperbound", "60x_oil_betasource_uppersurface_upperbound", "60x_oil_nosource_uppersurface_upperbound"]
    plot_experiment_types = ["60x.2.51V.1kHz.exp_100ms"]
    
    with PdfPages(output_filename) as pdf:
        for experiment_type in plot_experiment_types:
            with tifffile.TiffFile("microscope/track_detection/LED_Fiber/{}.tif".format(experiment_type)) as tif:
                num_pages = len(tif.pages)
                print(f"Processing file: {experiment_type}, Number of pages: {num_pages}")
                                
                sum_img = None
                num_images = 0
                
                for i, page in enumerate(tif.pages):
                    # if i < 5 or i >= num_pages - 5:
                    #     continue
    
                    img = page.asarray()
    
                    if sum_img is None:
                        sum_img = np.zeros_like(img, dtype=np.float64)
                    
                    # 累加图像数据
                    sum_img += img
                    num_images += 1
                
                if num_images > 0:
                    # 计算平均图像
                    avg_img = sum_img / num_images
                    print("Mean = {}".format(np.mean(avg_img)))
                    
                # 绘制平均图像并保存到PDF
                # plt.imshow(avg_img, cmap='hot', vmin=180, vmax=230)
                plt.imshow(avg_img, cmap='hot')
                # plt.xlim(0, 1400)
                # plt.ylim(0, 1400)
                plt.xlim(0, 2304)
                plt.ylim(0, 4096)
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




def calculate_photon_number(**kwargs):
    output_filename = kwargs.get('output_filename', '')

    voltage_list = ["2.60", "2.70", "2.80", 2.99, 3.02, 3.08, "3.10", 3.12, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19, "3.20", 3.21, 3.22]
    photon_number = {}
    
    with PdfPages(output_filename) as pdf:
        for voltage in voltage_list:
            photon_number[voltage] = []
            with tifffile.TiffFile("/lustre/neutrino/chenzhangming/microscope/track_detection/LED_Fiber/{}V.100kHz.tif".format(voltage)) as tif:
                for i, page in enumerate(tif.pages):
                    img = page.asarray().astype(np.float64)
                    img = 0.11 * (img - 200) / 0.83
                    # pixel_values = img.flatten()
                    # photon_number[voltage].append(0.11*(np.sum(pixel_values) - 200 * len(pixel_values))/0.83)
                    photon_number[voltage].append(np.sum(img))
                    
                plt.figure(figsize=(8, 6))
                plt.hist(photon_number[voltage], bins=50, color='gray', alpha=0.7)
    
                # plt.yscale('log')
                plt.title("Toltal Photon Number Distribution\nLED Voltage = {} V".format(voltage))
                plt.xlabel("Total Photon Number of one Frame")
                plt.ylabel("Frame Count")
                plt.grid(True)
    
                # 将当前图形添加到 PDF 文件中
                pdf.savefig()  # 保存当前图形到 PDF
                plt.close()    # 关闭当前图形
        
    output_1 = "/lustre/neutrino/chenzhangming/microscope/track_detection/LED_Fiber/new.Photon_Number_vs_Voltage.txt"
    with open(output_1, "w") as f:
        for key, value in photon_number.items():
            f.write("{} {}\n".format(key, np.mean(value)))


            
def plot_average_phton_number_all_frames(**kwargs):
    output_filename = kwargs.get('output_filename', '')

    # plot_experiment_types = ["alpha_exp1min_10min"]

    # plot_experiment_types = ["60x.2.51V.1kHz.exp_100ms"]
    plot_experiment_types = ["60x.2.99V.100kHz.pulse_width_30ns.exp_1s"]
    # plot_experiment_types = ["60x.background.exp_1s"]

    with PdfPages(output_filename) as pdf:
        for experiment_type in plot_experiment_types:
            with tifffile.TiffFile("microscope/track_detection/LED_Fiber/{}.tif".format(experiment_type)) as tif:
                num_pages = len(tif.pages)
                print(f"Processing file: {experiment_type}, Number of pages: {num_pages}")

                sum_img = None
                num_images = 0

                for i, page in enumerate(tif.pages):
                    # if i < 5 or i >= num_pages - 5:
                    #     continue

                    img = page.asarray()

                    if sum_img is None:
                        sum_img = np.zeros_like(img, dtype=np.float64)

                    # 累加图像数据
                    sum_img += img
                    num_images += 1

                if num_images > 0:
                    # 计算平均图像
                    avg_img = (0.11*(sum_img / num_images - 200)/0.85)
                    print("Mean = {}".format(np.mean(avg_img)))
                    
                # background = np.mean(avg_img[0:200, :])
                # avg_img = avg_img - background

                # norm = LogNorm(vmin=10, vmax=35)
                # plt.imshow(avg_img, cmap='hot', norm=norm)

                # plt.imshow(avg_img, cmap='hot', vmin=0, vmax=5)
                plt.imshow(avg_img, cmap='hot')
                # plt.xlim(0, 1400)
                # plt.ylim(0, 1400)
                plt.xlim(0, 2304)
                plt.ylim(0, 4096)
                plt.xlabel("pixel")
                plt.ylabel("pixel")

                cbar = plt.colorbar()
                cbar.set_label("Photon Number")
                plt.title("Average Image\n{}".format(experiment_type))
                pdf.savefig()  # 保存到PDF
                plt.close()




def plot_all_frames_photon_number_distribution(**kwargs):
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
                # if i < 5 or i >= num_pages - 5:
                #     continue

                img = page.asarray()
                img = 0.11*(img - 200)/0.85
            
                # 获取当前帧的最小值和最大值
                current_min = np.min(img)
                current_max = np.max(img)

                # 更新全局最小值和最大值
                global_min_value = min(global_min_value, current_min)
                global_max_value = max(global_max_value, current_max)

            print(f"Global Min Value: {global_min_value}, Global Max Value: {global_max_value}")
            print("="*50)

            # global_min_value = 0
            # global_max_value = 5000

            # 再次处理每一帧，计算灰度直方图并累加
            for i, page in enumerate(tif.pages):
                # if i < 5 or i >= num_pages - 5:
                #     continue

                img = page.asarray()
                img = 0.11*(img - 200)/0.85

                # 计算当前帧的灰度直方图
                frame_histogram, bin_edges = np.histogram(img.flatten(), bins=100, range=(global_min_value, global_max_value))

                # 如果是第一帧，初始化总的直方图
                if total_histogram is None:
                    total_histogram = frame_histogram
                else:
                    total_histogram += frame_histogram

            # 绘制所有帧的总灰度分布直方图
            plt.figure(figsize=(12, 6))
            plt.bar(bin_edges[:-1], total_histogram, width=np.diff(bin_edges), color='gray', alpha=0.7)

            # 设置 y 轴为对数尺度
            plt.yscale('log')

            plt.title("Photon Number Distribution - All Frames\n{}".format(experiment_type))
            plt.xlabel("Pixel Intensity (Gray Level)")
            plt.ylabel("Count")

            plt.grid(True)

            # 将总的灰度分布直方图添加到 PDF 文件中
            pdf.savefig()  # 保存当前图形到 PDF
            plt.close()    # 关闭当前图形



def plot_average_phton_number_distribution(**kwargs):
    output_filename = kwargs.get('output_filename', '')

    # plot_experiment_types = ["alpha_exp1min_10min"]

    # plot_experiment_types = ["60x.2.51V.1kHz.exp_100ms"]
    plot_experiment_types = ["60x.2.99V.100kHz.pulse_width_30ns.exp_1s"]
    # plot_experiment_types = ["60x.background.exp_1s"]
    
    with PdfPages(output_filename) as pdf:
        for experiment_type in plot_experiment_types:
            with tifffile.TiffFile("microscope/track_detection/LED_Fiber/{}.tif".format(experiment_type)) as tif:
                num_pages = len(tif.pages)
                print(f"Processing file: {experiment_type}, Number of pages: {num_pages}")

                sum_img = None
                num_images = 0

                for i, page in enumerate(tif.pages):
                    # if i < 5 or i >= num_pages - 5:
                    #     continue

                    img = page.asarray()

                    if sum_img is None:
                        sum_img = np.zeros_like(img, dtype=np.float64)

                    # 累加图像数据
                    sum_img += img
                    num_images += 1

                if num_images > 0:
                    # 计算平均图像
                    avg_img = (0.11*(sum_img / num_images - 200)/0.85)
                    print("Mean = {}".format(np.mean(avg_img)))

                # background = np.mean(avg_img[0:200, :])
                # avg_img = avg_img - background
                # signal = np.mean(avg_img[1800:2300, :])
                # print("Background = {}".format(background))
                # print("Signal = {}".format(signal))

                pixel_values = avg_img.flatten()

                # 绘制灰度值的直方图
                plt.figure(figsize=(8, 6))
                plt.hist(pixel_values, bins=50, color='gray', alpha=0.7)

                # 设置 y 轴为对数尺度
                plt.yscale('log')

                plt.title("Photon Number Distribution\n{}".format(experiment_type))
                plt.xlabel("Photon Number")
                plt.ylabel("Pixel Count")
                plt.grid(True)

                # 将当前图形添加到 PDF 文件中
                pdf.savefig()  # 保存当前图形到 PDF
                plt.close()    # 关闭当前图形



def plot_substract_background_average_phton_number_all_frames(**kwargs):
    output_filename = kwargs.get('output_filename', '')
    plot_range = kwargs.get('plot_range', '')
    # plot_experiment_types = ["alpha_exp1min_10min"]

    # plot_experiment_types = ["60x.2.51V.1kHz.exp_100ms"]
    plot_experiment_types = ["60x.2.99V.100kHz.pulse_width_30ns.exp_1s", "60x.background.exp_1s"]
    # plot_experiment_types = ["60x.background.exp_1s"]

    record_img = {}
    with PdfPages(output_filename) as pdf:
        for img_number, experiment_type in enumerate(plot_experiment_types):
            print("img_number = {}".format(img_number))
            with tifffile.TiffFile("microscope/track_detection/LED_Fiber/{}.tif".format(experiment_type)) as tif:
                num_pages = len(tif.pages)
                print(f"Processing file: {experiment_type}, Number of pages: {num_pages}")

                sum_img = None
                num_images = 0

                for i, page in enumerate(tif.pages):
                    # if i < 5 or i >= num_pages - 5:
                    #     continue

                    img = page.asarray()

                    if sum_img is None:
                        sum_img = np.zeros_like(img, dtype=np.float64)

                    # 累加图像数据
                    sum_img += img
                    num_images += 1

                if num_images > 0:
                    # 计算平均图像
                    avg_img = (0.11*(sum_img / num_images - 200)/0.85)
                    print("Mean = {}".format(np.mean(avg_img)))

                    record_img[img_number] = avg_img
                # background = np.mean(avg_img[0:200, :])
                # avg_img = avg_img - background

                # norm = LogNorm(vmin=10, vmax=35)
                # plt.imshow(avg_img, cmap='hot', norm=norm)

                # plt.imshow(avg_img, cmap='hot', vmin=0, vmax=5)
        substract_background_img = record_img[0] - record_img[1]
        plt.imshow(substract_background_img, cmap='hot', vmin=plot_range[0], vmax=plot_range[1])
        # plt.xlim(0, 1400)
        # plt.ylim(0, 1400)
        plt.xlim(0, 2304)
        plt.ylim(0, 4096)
        plt.xlabel("pixel")
        plt.ylabel("pixel")
        
        cbar = plt.colorbar()
        cbar.set_label("Photon Number")
        # plt.title("Substract Background Image\n{} - {}".format(plot_experiment_types[0], plot_experiment_types[1]))
        plt.title("Substract Background Image")
        pdf.savefig()  # 保存到PDF
        plt.close()




def plot_diff_experiment_average_photon_number_all_frames_distribution():

    # plot_experiment_types = ["microscope.2h", "beta_crystal_oil_microscope.2h", "crystal_oil_microscope.2h", "alpha_crystal_oil_microscope.2h", "alpha_crystal_microscope.2h"]
    plot_experiment_types = ["60x.2.99V.100kHz.pulse_width_30ns.exp_1s", "60x.background.exp_1s"]

    hist_range = (0, 25)
    bin_count = 250

    # 创建一个 PDF 文件
    with PdfPages("microscope/track_detection/LED_Fiber/average_photon_number_distribution.range_0_50.trial2_3.pdf") as pdf:

        total_histograms = []
        bin_edges_list = []

        for experiment_type in plot_experiment_types:
            with tifffile.TiffFile("microscope/track_detection/LED_Fiber/{}.tif".format(experiment_type)) as tif:
                num_pages = len(tif.pages)
                print(f"Processing file: {experiment_type}, Number of pages: {num_pages}")

                sum_img = None
                num_images = 0
                
                # 计算灰度直方图并累加
                for i, page in enumerate(tif.pages):

                    img = page.asarray()

                    if sum_img is None:
                        sum_img = np.zeros_like(img, dtype=np.float64)

                    # 累加图像数据
                    sum_img += img
                    num_images += 1

                if num_images > 0:
                    # 计算平均图像
                    avg_img = (0.11*(sum_img / num_images - 200)/0.85)

                    # 计算当前帧的灰度直方图
                    total_histogram, bin_edges = np.histogram(avg_img.flatten(), bin_count, hist_range)

                    total_histograms.append(total_histogram)
                    bin_edges_list.append(bin_edges)


        if len(total_histograms) >= 2:
            # 获取直方图的差值
            hist_diff = total_histograms[0] - total_histograms[1]
            
            # 在同一张图上绘制两个直方图和它们的差值
            plt.figure(figsize=(10, 6))
    
            # 绘制第一个实验的直方图
            plt.hist(bin_edges_list[0][:-1], bins=bin_edges_list[0], weights=total_histograms[0], label=plot_experiment_types[0], alpha=0.7, histtype='step')
    
            # 绘制第二个实验的直方图
            plt.hist(bin_edges_list[1][:-1], bins=bin_edges_list[1], weights=total_histograms[1], label=plot_experiment_types[1], alpha=0.7, histtype='step')
    
            # 绘制两个直方图的差值
            plt.hist(bin_edges_list[0][:-1], bins=bin_edges_list[0], weights=hist_diff, label="Signal - Background", alpha=0.7, histtype='step')
    
            # plt.yscale('log')
            plt.title("Average Photon Number Distribution (Difference)")
            plt.xlabel("Photon Number")
            plt.ylabel("Pixel Count")
            plt.legend()
            plt.grid(True)
    
            # 将图形保存到 PDF 文件
            pdf.savefig()
            plt.close()
                        # 将当前文件的直方图和 bin_edges 保存到列表中

        #         total_histograms.append(total_histogram)
        #         bin_edges_list.append(bin_edges)


        # total_histograms.append(total_histogram[0] - total_histogram[1])
        # bin_edges_list.append(bin_edges_list[-1])
        # plot_experiment_types.append("Signal - Background")
        # hist
        # plt.figure(figsize=(10, 6))

        # for i, total_histogram in enumerate(total_histograms):
        #     # 归一化
        #     # total_histogram_normalized = total_histogram / total_histogram.sum()

        #     plt.hist(bin_edges_list[i][:-1], bins=bin_edges_list[i], weights=total_histogram, label=plot_experiment_types[i], alpha=0.7, histtype='step')

        # plt.yscale('log')
        # plt.title("Average Photon Number Distibution")
        # plt.xlabel("Photon Number")
        # plt.ylabel("Pixel Count")
        # plt.legend()
        # plt.grid(True)

        # pdf.savefig()
        # plt.close()


        # # line
        # plt.figure(figsize=(10, 6))

        # for i, total_histogram in enumerate(total_histograms):
        #     plt.plot(bin_edges_list[i][:-1], total_histogram, label=plot_experiment_types[i], alpha=0.7)

        # plt.yscale('log')
        # plt.title("Pixel Intensity Distribution Comparison")
        # plt.xlabel("Pixel Intensity (Gray Level)")
        # plt.ylabel("Count")
        # plt.legend()
        # plt.grid(True)

        # pdf.savefig()
        # plt.close()




def plot_fiber_vs_camera():

    x = []
    y1 = []
    y2 = []
    
    with open("/lustre/neutrino/chenzhangming/lab_class/LED_Fiber/Photon_Number_vs_Voltage.txt", 'r') as file:
        for line in file:
            voltage, photon = line.split()
            x.append(float(voltage))
            y1.append(float(photon))

    with open("/lustre/neutrino/chenzhangming/microscope/track_detection/LED_Fiber/Photon_Number_vs_Voltage.txt", 'r') as file:
        for line in file:
            voltage, photon = line.split()
            y2.append(float(photon)/ 1e5 * 1.00607)


    with PdfPages("/lustre/neutrino/chenzhangming/microscope/track_detection/LED_Fiber/Compare_Fiber_Camera.pdf") as pdf:
 
        plt.rcParams.update({'font.size': 13})
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # 创建两个子图

        ax1.plot(x, y1, label="Come out from Fiber", marker='o', linestyle='-', color='b')
        ax1.plot(x, y2, label="Arrive Camera", marker='s', linestyle='-', color='r')

        ax1.set_xlabel('LED Voltage (V)')
        ax1.set_ylabel('Photon Number')
        ax1.grid(True)
        ax1.legend()

        ratio = [y2_i / y1_i if y1_i != 0 else 0 for y2_i, y1_i in zip(y2, y1)]
    
        ax2.plot(x, ratio, label="Ratio", marker='x', linestyle='-', color='g')
        ax2.set_xlabel('LED Voltage (V)')
        ax2.set_ylabel('Ratio')
        ax2.set_ylim(0, 0.4)
        ax2.grid(True)
        ax2.legend()
        
        # 将当前图形保存到PDF文件中
        pdf.savefig(fig)  # 保存当前图形
        plt.close()  # 关闭图形，避免过多内存占用



def plot_substract_background_frame_by_frame(**kwargs):
    output_filename = kwargs.get('output_filename', '')
    background = kwargs.get('background', '')
    signal = kwargs.get('signal', '')
    plot_range = kwargs.get('plot_range', '')

    # plot_experiment_types = ["60x.2.51V.1kHz.exp_100ms"]
    with tifffile.TiffFile("microscope/track_detection/scan_depth/{}.tif".format(background)) as tif:
        sum_img = None
        num_images = 0
        for i, page in enumerate(tif.pages):
            img = page.asarray().astype(np.float64)
            if sum_img is None:
                sum_img = np.zeros_like(img, dtype=np.float64)
            sum_img += img
            num_images += 1
            
        avg_img = (0.11*(sum_img / num_images - 200)/0.83)
        # avg_img = sum_img / num_images

    with PdfPages(output_filename) as pdf:
        with tifffile.TiffFile("microscope/track_detection/scan_depth/{}.tif".format(signal)) as tif:
            for i, page in enumerate(tif.pages):
                img = page.asarray().astype(np.float64)
                img = 0.11*(img - 200)/0.83
                img = img - avg_img
                        
                plt.imshow(img, cmap='hot', vmin=plot_range[0], vmax=plot_range[1])
                
                plt.xlim(0, 2304)
                plt.ylim(0, 4096)
                plt.xlabel("pixel")
                plt.ylabel("pixel")

                cbar = plt.colorbar()
                cbar.set_label("Photon Number")
                plt.title("Frame {}".format(i+1))
                pdf.savefig()  # 保存到PDF
                plt.close()




def plot_substract_background_photon_number_distribution_frame_by_frame(**kwargs):
    output_filename = kwargs.get('output_filename', '')
    background = kwargs.get('background', '')
    signal = kwargs.get('signal', '')

    with tifffile.TiffFile("microscope/track_detection/align/{}.tif".format(background)) as tif:
        sum_img = None
        num_images = 0
        for i, page in enumerate(tif.pages):
            img = page.asarray().astype(np.float64)
            if sum_img is None:
                sum_img = np.zeros_like(img, dtype=np.float64)
            sum_img += img
            num_images += 1

        avg_img = (0.11*(sum_img / num_images - 200)/0.83)

        
    with PdfPages(output_filename) as pdf:
        with tifffile.TiffFile("microscope/track_detection/align/{}.tif".format(signal)) as tif:
            for i, page in enumerate(tif.pages):
                img = page.asarray().astype(np.float64)
                img = 0.11*(img - 200)/0.83
                img = img - avg_img
                pixel_values = img.flatten()

                plt.figure(figsize=(8, 6))
                plt.hist(pixel_values, bins=50, color='gray', alpha=0.7)

                plt.yscale('log')

                plt.title("Photon Number Distribution (Substract Background) - Frame {}\n{}".format(i+1, signal))
                plt.xlabel("Photon Number")
                plt.ylabel("Pixel Count")
                plt.grid(True)

                pdf.savefig()  # 保存当前图形到 PDF
                plt.close()    # 关闭当前图形


                

def calculate_alpha_candidate(image, window_size=20):
    M, N = image.shape
    alpha_count = 0
    alpha_regions = []  # 存储符合条件的区域
    alpha_photons = []
    
    # 计算积分图，累积图像的和
    integral_image = np.cumsum(np.cumsum(image, axis=0), axis=1)

    # 创建一个标记矩阵，记录已扫描过的区域
    visited = np.zeros_like(image, dtype=bool)

    def get_window_sum(x1, y1, x2, y2):
         # 确保坐标合法
        if x1 < 0 or y1 < 0:
            return 0
        return integral_image[x2, y2] - (integral_image[x1-1, y2] if x1 > 0 else 0) - (integral_image[x2, y1-1] if y1 > 0 else 0) + (integral_image[x1-1, y1-1] if x1 > 0 and y1 > 0 else 0)

    for i in range(M - window_size + 1):
        for j in range(N - window_size + 1):
            # 如果当前窗口已经被访问过，则跳过
            if np.any(visited[i:i + window_size, j:j + window_size]):
                continue

            # 计算当前窗口的和
            window_sum = get_window_sum(i + window_size - 1, j + window_size - 1, i, j)

            if window_sum >= 200:
                alpha_count += 1
                alpha_regions.append((i, j, i + window_size - 1, j + window_size - 1))  # 存储符合条件的窗口的边界框
                alpha_photons.append(window_sum)
                
                # 标记该区域为已访问，避免后续扫描与其重叠
                visited[i:i + window_size, j:j + window_size] = True

                # 跳过窗口的右侧区域
                j += window_size - 1  # 跳过当前窗口的列

    return alpha_count, alpha_regions, alpha_photons




def analysis_alpha_candidate(**kwargs):
    output_filename = kwargs.get('output_filename', '')
    background = kwargs.get('background', '')
    signal = kwargs.get('signal', '')

    with tifffile.TiffFile("microscope/track_detection/scan_depth/{}.tif".format(background)) as tif:
        sum_img = None
        num_images = 0
        for i, page in enumerate(tif.pages):
            img = page.asarray().astype(np.float64)
            if sum_img is None:
                sum_img = np.zeros_like(img, dtype=np.float64)
            sum_img += img
            num_images += 1

        avg_img = (0.11*(sum_img / num_images - 200)/0.83)

    record_alpha_count = {}
    record_alpha_photons = {}
    with PdfPages(output_filename) as pdf:
        with tifffile.TiffFile("microscope/track_detection/scan_depth/{}.tif".format(signal)) as tif:
            for i, page in enumerate(tif.pages):
                # if i > 3:
                #     break
                img = page.asarray().astype(np.float64)
                img = 0.11*(img - 200)/0.83
                img = img - avg_img

                # 获取符合条件的区域及其数量
                alpha_count, alpha_regions, alpha_photons = calculate_alpha_candidate(img)
                record_alpha_count[i+1] = alpha_count
                record_alpha_photons[i+1] = alpha_photons
                # print("Frame {}, Alpha Candidate Count = {}".format(i+1, alpha_count))
                
                # 绘制图像
                plt.imshow(img, cmap='hot', vmin=0, vmax=1)
                plt.xlim(0, 2304)
                plt.ylim(0, 4096)
                plt.xlabel("pixel")
                plt.ylabel("pixel")

                # 标记符合条件的区域
                for region in alpha_regions:
                    x1, y1, x2, y2 = region
                    plt.gca().add_patch(plt.Rectangle((y1, x1), y2 - y1 + 1, x2 - x1 + 1, linewidth=0.3, edgecolor='green', facecolor='none'))

                # 添加色条
                cbar = plt.colorbar()
                cbar.set_label("Photon Number")
                plt.title("Frame {}".format(i+1))
                pdf.savefig()  # 保存到PDF
                plt.close()

    output_1 = "microscope/track_detection/scan_depth/alpha_candidate/count.{}.txt".format(signal)
    with open(output_1, "w") as f:
        for key, value in record_alpha_count.items():
            f.write("{} {}\n".format(key, value))

    output_2 = "microscope/track_detection/scan_depth/alpha_candidate/photons.{}.txt".format(signal)
    with open(output_2, "w") as f:
        for key, value in record_alpha_photons.items():
            f.write("{} {}\n".format(key, value))


def plot_alpha_candidate_vs_height():

    x = []
    y1 = []
    y2 = []
    print("===============================")
    print("=== Average Alpha Candidate ===")
    print("===============================")

    signal_background_pairs = [["60x.nosource.exp2s.height0um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height0um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height1um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height2um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height3um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height4um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height5um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height8um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height11um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height14um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height17um", "60x.nosource.exp2s.height0um"], ["60x.alpha.exp2s.height20um", "60x.nosource.exp2s.height0um"]]
    for signal_background_pair in signal_background_pairs:
        signal = signal_background_pair[0]
        frame_total_count = 0
        alpha_total_count = 0
        with open("microscope/track_detection/scan_depth/alpha_candidate/{}.txt".format(signal), 'r') as file:
            for line in file:
                frame, alpha_count = line.split()
                frame_total_count += 1
                alpha_total_count += int(alpha_count)
        print("{} = {}".format(signal, alpha_total_count / frame_total_count))

    # with PdfPages("/lustre/neutrino/chenzhangming/microscope/track_detection/LED_Fiber/Compare_Fiber_Camera.pdf") as pdf:

    #     plt.rcParams.update({'font.size': 13})
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # 创建两个子图

    #     ax1.plot(x, y1, label="Come out from Fiber", marker='o', linestyle='-', color='b')
    #     ax1.plot(x, y2, label="Arrive Camera", marker='s', linestyle='-', color='r')

    #     ax1.set_xlabel('LED Voltage (V)')
    #     ax1.set_ylabel('Photon Number')
    #     ax1.grid(True)
    #     ax1.legend()

    #     ratio = [y2_i / y1_i if y1_i != 0 else 0 for y2_i, y1_i in zip(y2, y1)]

    #     ax2.plot(x, ratio, label="Ratio", marker='x', linestyle='-', color='g')
    #     ax2.set_xlabel('LED Voltage (V)')
    #     ax2.set_ylabel('Ratio')
    #     ax2.set_ylim(0, 0.4)
    #     ax2.grid(True)
    #     ax2.legend()

    #     # 将当前图形保存到PDF文件中
    #     pdf.savefig(fig)  # 保存当前图形
    #     plt.close()  # 关闭图形，避免过多内存占用



def plot_substract_background_mean_total_photon_count():

    signal_background_pairs = [["60x.nosource.exp2s.height0um", "60x.nosource.exp2s.height0um"], 
                               ["60x.alpha.exp2s.height0um", "60x.nosource.exp2s.height0um"], 
                               ["60x.alpha.exp2s.height1um", "60x.nosource.exp2s.height0um"],
                               ["60x.alpha.exp2s.height2um", "60x.nosource.exp2s.height0um"], 
                               ["60x.alpha.exp2s.height3um", "60x.nosource.exp2s.height0um"],
                               ["60x.alpha.exp2s.height4um", "60x.nosource.exp2s.height0um"], 
                               ["60x.alpha.exp2s.height5um", "60x.nosource.exp2s.height0um"], 
                               ["60x.alpha.exp2s.height8um", "60x.nosource.exp2s.height0um"], 
                               ["60x.alpha.exp2s.height11um", "60x.nosource.exp2s.height0um"], 
                               ["60x.alpha.exp2s.height14um", "60x.nosource.exp2s.height0um"], 
                               ["60x.alpha.exp2s.height17um", "60x.nosource.exp2s.height0um"], 
                               ["60x.alpha.exp2s.height20um", "60x.nosource.exp2s.height0um"]]
    
    with tifffile.TiffFile("microscope/track_detection/scan_depth/60x.nosource.exp2s.height0um.tif") as tif:
        sum_img = None
        num_images = 0
        for i, page in enumerate(tif.pages):
            img = page.asarray().astype(np.float64)
            if sum_img is None:
                sum_img = np.zeros_like(img, dtype=np.float64)
            sum_img += img
            num_images += 1

        avg_img = (0.11*(sum_img / num_images - 200)/0.83)
        print("avg_img")
        print(avg_img)
        # avg_img = sum_img / num_images

    record_photon = {}
    for signal_background_pair in signal_background_pairs:
        signal = signal_background_pair[0]
        record_photon[signal] = []
        with tifffile.TiffFile("microscope/track_detection/scan_depth/{}.tif".format(signal)) as tif:
            for i, page in enumerate(tif.pages):
                img = page.asarray().astype(np.float64)
                img = 0.11*(img - 200)/0.83
                img = img - avg_img

                record_photon[signal].append(np.sum(img))


    means = [np.mean(record_photon[signal]) for signal in record_photon]

    # Plotting the results
    plt.figure(figsize=(15, 6))
    plt.plot(list(record_photon.keys()), means, marker='o', linestyle='-', color='b', label='Mean Total Photon Count')
    # plt.xlabel('Signal')
    plt.ylabel('Mean Total Photon Count')
    # plt.title('Mean Photon Count for Different Signals')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save or show the plot
    plt.savefig("microscope/track_detection/scan_depth/total_photon_count/TEST.pdf")


def plot_image_frame_by_frame(**kwargs):
    output_filename = kwargs.get('output_filename', '')
    signal = kwargs.get('signal', '')
    plot_range = kwargs.get('plot_range', '')

    with PdfPages(output_filename) as pdf:
        with tifffile.TiffFile("microscope/track_detection/qcmos_60x/{}.tif".format(signal)) as tif:
            for i, page in enumerate(tif.pages):
                img = page.asarray().astype(np.float64)
                img = 0.11*(img - 200)/0.83
         
                plt.imshow(img, cmap='hot', vmin=plot_range[0], vmax=plot_range[1])

                plt.xlim(0, 2304)
                plt.ylim(0, 4096)
                plt.xlabel("pixel")
                plt.ylabel("pixel")

                cbar = plt.colorbar()
                cbar.set_label("Photon Number")
                plt.title("Frame {}".format(i+1))
                pdf.savefig()  # 保存到PDF
                plt.close()


#last



# ========================================================================================
# ======================================= opencv =========================================
# =======================================================================================

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

# =======================================================================================
# =======================================================================================
                    
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


# plot_average_all_frames(
#         output_filename = "microscope/track_detection/cooling/average_all_frames.pdf"
# )


# plot_diff_experiment_all_frames_gray_distribution()


# 20250317
# experiment_types = ["alpha_scintillator.exposure_1min"]

# for experiment_type in experiment_types:
#     print("experiment type = {}".format(experiment_type))
#     plot_gray_distribution_frame_by_frame(
#         input_filename = "microscope/track_detection/qcmos/{}.tif".format(experiment_type),
#         output_filename = "microscope/track_detection/qcmos/{}.gray_distribution_frame_by_frame.pdf".format(experiment_type)
#         )


# plot_average_all_frames(
#     output_filename = "microscope/track_detection/qcmos/TEST.average_all_frames.pdf"
# )

# experiment_types = ["nosource_exp1s_2min", "alphasource_exp1s_2min", "betasource_exp1s_2min", "nosource_exp10s_1min", "betasource_exp10s_1min"]
# experiment_types = ["alpha_exp1min_10min"]

# for experiment_type in experiment_types:
#     print("experiment type = {}".format(experiment_type))
#     plot_all_frames_gray_distribution(
#         experiment_type = experiment_type,
#         input_filename = "microscope/track_detection/qcmos/{}.tif".format(experiment_type),
#         output_filename = "microscope/track_detection/qcmos/same_x_axis.{}.gray_distribution_all_frames.pdf".format(experiment_type)
#     )


#20250320
# plot_average_all_frames(
#     output_filename = "microscope/track_detection/qcmos_60x/TEST.average_all_frames.pdf"
# )


# experiment_type = "60x.2.99V.100kHz.pulse_width_30ns.exp_1s"
# calculate_photon_number(input_filename = "/lustre/neutrino/chenzhangming/microscope/track_detection/LED_Fiber/{}.tif".format(experiment_type),
#                         output_filename = "/lustre/neutrino/chenzhangming/microscope/track_detection/LED_Fiber/photon_number.{}.pdf".format(experiment_type)
#                         )



# plot_average_all_frames(
#     output_filename = "microscope/track_detection/LED_Fiber/TEST.average_all_frames.pdf"
# )

# experiment_type = "60x.2.99V.100kHz.pulse_width_30ns.exp_1s"
# experiment_type = "60x.background.exp_1s"
# plot_average_phton_number_all_frames(
#     output_filename = "microscope/track_detection/LED_Fiber/substract_background.average_photon_number_all_frames.{}.pdf".format(experiment_type)
# )

# plot_average_phton_number_all_frames(
#     output_filename = "microscope/track_detection/LED_Fiber/average_photon_number_all_frames.{}.pdf".format(experiment_type)
# )


# plot_average_phton_number_distribution(output_filename = "microscope/track_detection/LED_Fiber/average_phton_number_distribution.{}.pdf".format(experiment_type)
#                                        )

# plot_range = [0, 5]
# plot_substract_background_average_phton_number_all_frames(output_filename = "microscope/track_detection/LED_Fiber/substract_background.average_photon_number_all_frames.plot_range_{}_{}.pdf".format(plot_range[0], plot_range[1]),
#                                                           plot_range = plot_range
#                                                           )

# plot_diff_experiment_average_photon_number_all_frames_distribution()

# calculate_photon_number(output_filename = "/lustre/neutrino/chenzhangming/microscope/track_detection/LED_Fiber/Photon_Number_vs_Voltage.pdf")

# plot_cd_pmt_DCR_vs_run()

# plot_range_list = [[220, 225], [225, 230], [230, 235], [235, 240]]
# plot_range_list = [[200, 205], [205, 210], [210, 215], [215, 220]]
# plot_range_list = [[195, 197], [197, 199], [199, 201], [201, 203], [203, 205], [205, 207]]
# for plot_range in plot_range_list:
#     plot_image_frame_by_frame(output_filename = "microscope/track_detection/align/beta.exp10s.plot_range_{}_{}.pdf".format(plot_range[0], plot_range[1]),
#                               plot_range = plot_range
#                               )

# plot_substract_background_frames(output_filename = "/lustre/neutrino/chenzhangming/microscope/track_detection/qcmos_60x/nosource_substract_nosource.pdf")

# plot_substract_background_gray_distribution_frame_by_frame(output_filename = "/lustre/neutrino/chenzhangming/microscope/track_detection/qcmos_60x/photon_distribution.alpha_substract_nosource.pdf")

# plot_range_list = [[195, 197], [197, 199], [199, 201], [201, 203], [203, 205], [205, 207]]
# plot_range_list = [[0, 3]]
# for plot_range in plot_range_list:
#     plot_image_frame_by_frame(output_filename = "microscope/track_detection/align/nosource_exp10s.plot_range_{}_{}.pdf".format(plot_range[0], plot_range[1]),
#                               plot_range = plot_range
#                               )


# plot_substract_background_frames(output_filename = "/lustre/neutrino/chenzhangming/microscope/track_detection/align/nosource_exp10s.substract_nosource.pdf")

# calculate_photon_number(output_filename = "/lustre/neutrino/chenzhangming/microscope/track_detection/LED_Fiber/new.Photon_Number_vs_Voltage.pdf")

############
# 20250322 #
############

# signal_background_pairs = [["60x.nosource.exp10s", "60x.nosource.exp10s"],
#                            ["60x.alphasource.exp10s", "60x.nosource.exp10s"],
#                            ["60x.alphasource.exp10s.below.uppersurface.5um", "60x.nosource.exp10s"],
#                            ["60x.alphasource.exp10s.below.uppersurface.10um", "60x.nosource.exp10s"]]

# signal_background_pairs = [["60x_nosource_uppersurface", "60x_nosource_uppersurface"],
#                            ["60x_alphasource_uppersurface", "60x_nosource_uppersurface"],
#                            ["60x_betasource_uppersurface", "60x_nosource_uppersurface"]]

# signal_background_pairs = [["60x_film_nosource_uppersurface", "60x_film_nosource_uppersurface"],
#                            ["60x_film_alphasource_uppersurface", "60x_film_nosource_uppersurface"],
#                            ["60x_film_betasource_uppersurface", "60x_film_nosource_uppersurface"]]

# signal_background_pairs = [["60x_oil_nosource_uppersurface_upperbound", "60x_oil_nosource_uppersurface_upperbound"],
#                            ["60x_oil_alphasource_uppersurface_upperbound", "60x_oil_nosource_uppersurface_upperbound"],
#                            ["60x_oil_betasource_uppersurface_upperbound", "60x_oil_nosource_uppersurface_upperbound"]]

# signal_background_pairs = [["60x_oil_nosource_uppersurface_lowerbound", "60x_oil_nosource_uppersurface_lowerbound"],
#                            ["60x_oil_alphasource_uppersurface_lowerbound", "60x_oil_nosource_uppersurface_lowerbound"],
#                            ["60x_oil_betasource_uppersurface_lowerbound", "60x_oil_nosource_uppersurface_lowerbound"]]

# signal_background_pairs = [["10x.nosource.exp2s.height0um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height0um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height1um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height2um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height3um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height4um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height5um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height8um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height11um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height14um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height17um", "10x.nosource.exp2s.height0um"],
#                            ["10x.alpha.exp2s.height20um", "10x.nosource.exp2s.height0um"]]


# plot_range_list = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]
# plot_range_list = [[0, 1], [1, 2], [2, 3], [3, 4]]
# plot_range_list = [[0, 0.5], [0.5, 1], [0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]
# plot_range_list = [[0, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2], [0.2, 0.25]]
# for signal_background_pair in signal_background_pairs:
#     for plot_range in plot_range_list:
#         plot_substract_background_frame_by_frame(output_filename = "microscope/track_detection/scan_depth/substract_background_frame_by_frame/{}.plot_range_{}_{}.pdf".format(signal_background_pair[0], plot_range[0], plot_range[1]),
#                                                  signal = signal_background_pair[0],
#                                                  background = signal_background_pair[1],
#                                                  plot_range = plot_range
#                                                  )


# for signal_background_pair in signal_background_pairs:
#     plot_substract_background_photon_number_distribution_frame_by_frame(output_filename = "microscope/track_detection/align/substract_background_photon_number_distribution_frame_by_frame/{}.pdf".format(signal_background_pair[0]),
#                                                                         signal = signal_background_pair[0],
#                                                                         background = signal_background_pair[1]
#                                                                         )


# signal_background_pairs = [["60x.nosource.exp2s.height0um", "60x.nosource.exp2s.height0um"]]

# for signal_background_pair in signal_background_pairs:
#     analysis_alpha_candidate(output_filename = "microscope/track_detection/scan_depth/alpha_candidate/{}.pdf".format(signal_background_pair[0]),
#                              signal = signal_background_pair[0],
#                              background = signal_background_pair[1]
#                              )

# plot_substract_background_mean_total_photon_count()


# signal_background_pairs = [["60x.nosource.exp2s.height0um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height0um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height1um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height2um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height3um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height4um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height5um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height8um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height11um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height14um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height17um", "60x.nosource.exp2s.height0um"],
#                            ["60x.alpha.exp2s.height20um", "60x.nosource.exp2s.height0um"]]

# signal_background_pairs = [["60x.nosource.exp10s", "60x.nosource.exp10s"],
#                            ["60x.alphasource.exp10s", "60x.nosource.exp10s"],
#                            ["60x.alphasource.exp10s.below.uppersurface.5um", "60x.nosource.exp10s"],
#                            ["60x.alphasource.exp10s.below.uppersurface.10um", "60x.nosource.exp10s"]]

# signal_background_pairs = [["60x_nosource_uppersurface", "60x_nosource_uppersurface"],
#                            ["60x_alphasource_uppersurface", "60x_nosource_uppersurface"],
#                            ["60x_betasource_uppersurface", "60x_nosource_uppersurface"]]

signal_background_pairs = [["60x_film_nosource_uppersurface", "60x_film_nosource_uppersurface"],
                           ["60x_film_alphasource_uppersurface", "60x_film_nosource_uppersurface"],
                           ["60x_film_betasource_uppersurface", "60x_film_nosource_uppersurface"]]

# signal_background_pairs = [["60x_oil_nosource_uppersurface_upperbound", "60x_oil_nosource_uppersurface_upperbound"],
#                            ["60x_oil_alphasource_uppersurface_upperbound", "60x_oil_nosource_uppersurface_upperbound"],
#                            ["60x_oil_betasource_uppersurface_upperbound", "60x_oil_nosource_uppersurface_upperbound"]]

# signal_background_pairs = [["60x_oil_nosource_uppersurface_lowerbound", "60x_oil_nosource_uppersurface_lowerbound"],
#                            ["60x_oil_alphasource_uppersurface_lowerbound", "60x_oil_nosource_uppersurface_lowerbound"],
#                            ["60x_oil_betasource_uppersurface_lowerbound", "60x_oil_nosource_uppersurface_lowerbound"]]

plot_range_list = [[0, 0.5], [0.5, 1], [0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]
for signal_background_pair in signal_background_pairs:
    signal = signal_background_pair[0]
    for plot_range in plot_range_list:
        plot_image_frame_by_frame(output_filename = "microscope/track_detection/qcmos_60x/image_frame_by_frame/{}.plot_range_{}_{}.pdf".format(signal, plot_range[0], plot_range[1]),
                                  signal = signal,
                                  plot_range = plot_range
                                  )



        
