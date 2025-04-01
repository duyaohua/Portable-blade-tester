import sys
import math
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton,QScrollArea, QHBoxLayout,QLabel, QTextEdit, QFileDialog, QVBoxLayout, QWidget, QMessageBox
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtCore import Qt
from PIL import Image, ImageDraw
import cv2
from image_processor import ImageProcessor
from skimage.filters import threshold_local
from MJJS import calculate_image_stats
class ImageProcessorApp(QMainWindow):
    A4_WIDTH_CM = 21.0
    A4_HEIGHT_CM = 29.7
    REFERENCE_AREA_1 = 500.0  # 全树
    REFERENCE_AREA_2 = 160.0  # 上枝
    REFERENCE_AREA_3 = 113.4  # 枝干
    REFERENCE_AREA_4 = 66.6   # 右上枝
    REFERENCE_AREA_5 = 66.6   # 左上枝
    REFERENCE_AREA_6 = 46.6   # 右下枝
    REFERENCE_AREA_7 = 46.6   # 左下枝
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理工具")
        self.resize(800, 600)
        A4_WIDTH_CM = 21.0
        A4_HEIGHT_CM = 29.7
        # 初始化变量
        self.image_path = None
        self.image = None
        self.points = []
        self.results = []
        self.black_actual_areas = {} 
        self.temp_image = None
        self.scale_factor = 1.0  # 缩放比例

         # 修改主窗口布局为水平布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # 创建水平布局用于放置图像显示区域和分割区域
        self.image_layout = QHBoxLayout()
        self.layout.addLayout(self.image_layout)
        
        # 左侧图像显示区域
        self.label_image = QLabel("图像将显示在这里")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setFixedSize(600, 400)
        self.label_image.mousePressEvent = self.mouse_press_event
        self.label_image.wheelEvent = self.wheel_event
        self.image_layout.addWidget(self.label_image)
        
        # 右侧分割区域显示 - 添加滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(200)  # 设置固定宽度
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建容器widget来放置分割区域
        self.segments_widget = QWidget()
        self.segments_layout = QVBoxLayout(self.segments_widget)
        self.segments_layout.setSpacing(5)
        self.segments_layout.setAlignment(Qt.AlignTop)  # 确保从顶部开始排列
        
        # 将容器添加到滚动区域
        self.scroll_area.setWidget(self.segments_widget)
        self.image_layout.addWidget(self.scroll_area)
        self.current_view_image = None  # Store current view image
        self.is_segment_view = False    # Flag to track if viewing segment
        

        # 按钮：上传图像
        self.upload_button = QPushButton("上传图像")
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

        # 按钮：清除点
        self.clear_button = QPushButton("清除点")
        self.clear_button.clicked.connect(self.clear_points)
        self.layout.addWidget(self.clear_button)

        # 结果显示区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

        # 按钮：保存结果
        self.save_button = QPushButton("保存结果为 CSV")
        self.save_button.clicked.connect(self.save_results)
        self.layout.addWidget(self.save_button)

    def upload_image(self):
        """上传图像并显示"""
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.bmp)")
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.temp_image = self.image.copy()
            self.scale_factor = 1.0  # 重置缩放比例
            # 执行图像分割
            self.segment_image()
            # 显示处理后的图像
            self.display_image()
    

    def segment_image(self):
        """分割图像并标注顶点，计算原始面积和实际面积"""
        if self.image is None:
            return

        try:
            # Store original image first for scale calculation
            self.original_image = self.image.copy()
            
            # 使用ImageProcessor处理图像
            results = ImageProcessor.process_image(self.image_path)
            
            # 获取比例因子
            scale_factor = self._get_scale_factor()
            
            # 将处理后的图像转换为PIL格式
            result_image_rgb = cv2.cvtColor(results['processed'], cv2.COLOR_BGR2RGB)
            self.temp_image = Image.fromarray(result_image_rgb)

            # 清空之前的结果
            self.results.clear()
            self.result_text.clear()
            self.black_actual_areas.clear()  # Clear previous areas

            # 记录分割结果
            for quad in results['quads']:
                # Convert vertices to list of integer tuples
                vertices = [(int(x), int(y)) for x, y in quad['vertices']]
                pixel_area = quad['area']
                region_number = quad['index']

                # 提取并处理分割区域
                min_x = min(x[0] for x in vertices)
                max_x = max(x[0] for x in vertices)
                min_y = min(y[1] for y in vertices)
                max_y = max(y[1] for y in vertices)
                
                segment_array = np.array(self.temp_image)[min_y:max_y, min_x:max_x]
                rotated_segment = ImageProcessor.rotate_to_horizontal(
                    cv2.cvtColor(np.array(segment_array), cv2.COLOR_RGB2BGR),
                    [(x - min_x, y - min_y) for x, y in vertices]
                )

                # 处理图像并保存
                gray_segment = cv2.cvtColor(rotated_segment, cv2.COLOR_BGR2GRAY)
                T = threshold_local(gray_segment, 11, offset=1, method="gaussian")
                binary_segment = (gray_segment > T).astype("uint8") * 255
                save_path = f'scan_{region_number}.png'
                cv2.imwrite(save_path, binary_segment)
                
                # 计算黑白像素比例
                stats = calculate_image_stats(save_path)
                
                # 计算实际面积
                calibration_factor = self._calculate_calibration_factor({"Shape": region_number, "Area": pixel_area})
                region_actual_area = pixel_area * (scale_factor ** 2) * calibration_factor
                black_actual_area = region_actual_area * stats['black_ratio']
                
                # 保存黑色实际面积
                self.black_actual_areas[region_number] = black_actual_area

                # 更新结果
                self.results.append({
                    "Type": "Segment",
                    "Shape": region_number,
                    "Area": pixel_area,
                    "ActualArea": black_actual_area,
                    "TotalArea": region_actual_area,
                    "Vertices": vertices,
                    "BlackPixelRatio": stats['black_ratio'],
                    "WhitePixelRatio": stats['white_ratio'],
                    "BlackPixels": stats['black_pixels'],
                    "WhitePixels": stats['white_pixels']
                })
                
                # 添加结果文本
                self.result_text.append(
                    f"区域 {region_number}:\n"
                    f"  填充面积 = {black_actual_area:.2f} 平方厘米\n"
                )

            # 显示处理后的图像
            self.display_image()

        except Exception as e:
            import traceback
            print(traceback.format_exc())  # For debugging
            QMessageBox.critical(self, "错误", f"分割图像时发生错误：{str(e)}")
    def display_image(self):
        """显示主图像和分割区域"""
        if self.temp_image:
            # 显示主图像
            width = int(self.temp_image.width * self.scale_factor)
            height = int(self.temp_image.height * self.scale_factor)
            scaled_image = self.temp_image.resize((width, height), Image.LANCZOS)

            display_image = QImage(self.label_image.width(), self.label_image.height(), QImage.Format_RGB888)
            display_image.fill(Qt.white)

            x = (self.label_image.width() - width) // 2
            y = (self.label_image.height() - height) // 2

            qimage = QImage(scaled_image.tobytes(), scaled_image.width, scaled_image.height, QImage.Format_RGB888)
            painter = QPainter(display_image)
            painter.drawImage(x, y, qimage)
            painter.end()

            pixmap = QPixmap.fromImage(display_image)
            self.label_image.setPixmap(pixmap)

            # 清除之前的分割区域显示
            for i in reversed(range(self.segments_layout.count())): 
                self.segments_layout.itemAt(i).widget().setParent(None)

            # 显示分割区域
            if hasattr(self, 'results'):
                for result in self.results:
                    if result["Type"] == "Segment":
                        try:
                            # 创建分割区域标签
                            segment_label = QLabel()
                            segment_label.setAlignment(Qt.AlignCenter)
                            
                            # 从主图像中提取分割区域
                            vertices = result["Vertices"]
                            min_x = min(x[0] for x in vertices)
                            max_x = max(x[0] for x in vertices)
                            min_y = min(y[1] for y in vertices)
                            max_y = max(y[1] for y in vertices)
                            
                            # 提取分割区域并旋转
                            segment_array = np.array(self.temp_image)[min_y:max_y, min_x:max_x]
                            rotated_segment = ImageProcessor.rotate_to_horizontal(
                                cv2.cvtColor(np.array(segment_array), cv2.COLOR_RGB2BGR),
                                [(x - min_x, y - min_y) for x, y in vertices]
                            )

                            # 处理图像
                            gray_segment = cv2.cvtColor(rotated_segment, cv2.COLOR_BGR2GRAY)
                            T = threshold_local(gray_segment, 11, offset=1, method="gaussian")
                            binary_segment = (gray_segment > T).astype("uint8") * 255
                            
                            # 保存处理后的图像
                            region_number = result["Shape"]
                            save_path = f'scan_{region_number}.png'
                            cv2.imwrite(save_path, binary_segment)
                            stats = calculate_image_stats(save_path)

                            # 计算实际面积
                            A4_AREA = self.A4_WIDTH_CM * self.A4_HEIGHT_CM
                            calibration_factor = self._calculate_calibration_factor(result)
                            region_area_ratio = result['Area'] / (self.image.width * self.image.height)
                            region_actual_area = result['Area'] * (self._get_scale_factor() ** 2) * calibration_factor
                            black_actual_area = region_actual_area * stats['black_ratio']
                            #实际面积添加到字典中
                            self.black_actual_areas[region_number] = black_actual_area
                            # 更新结果
                            self.results = [r for r in self.results if r["Shape"] != region_number]
                            self.results.append({
                                "Type": "Segment",
                                "Shape": region_number,
                                "Area": result['Area'],
                                "ActualArea": black_actual_area,
                                "TotalArea": region_actual_area,  # Add total area including white space
                                "Vertices": result["Vertices"],
                                "BlackPixelRatio": stats['black_ratio'],
                                "WhitePixelRatio": stats['white_ratio'],
                                "BlackPixels": stats['black_pixels'],
                                "WhitePixels": stats['white_pixels']
                            })

                            # 转换为PIL图像用于显示
                            segment_image = Image.fromarray(cv2.cvtColor(rotated_segment, cv2.COLOR_BGR2RGB))

                            # 计算缩放比例
                            max_width = 180
                            scale = min(1.0, max_width / segment_image.width)
                            new_width = int(segment_image.width * scale)
                            new_height = int(segment_image.height * scale)
                            
                            # 按比例缩放
                            if scale < 1.0:
                                segment_image = segment_image.resize((new_width, new_height), Image.LANCZOS)

                            # 显示在右侧面板
                            segment_qimage = QImage(
                                segment_image.tobytes(), 
                                segment_image.width, 
                                segment_image.height, 
                                QImage.Format_RGB888
                            )
                            segment_pixmap = QPixmap.fromImage(segment_qimage)
                            segment_label.setPixmap(segment_pixmap)

                            # 添加标题和区域信息
                            title_label = QLabel(
                                f"区域 {result['Shape']}\n"
                                f"总面积: {region_actual_area:.1f} cm²\n"
                                f"填充面积: {black_actual_area:.1f} cm²"
                            )
                            title_label.setAlignment(Qt.AlignCenter)

                            # 创建容器并添加到布局
                            container = QWidget()
                            container_layout = QVBoxLayout(container)
                            container_layout.addWidget(title_label)
                            container_layout.addWidget(segment_label)
                            self.segments_layout.addWidget(container)

                        except Exception as e:
                            print(f"处理区域 {result['Shape']} 时发生错误: {str(e)}")
                        
                        # 添加点击事件
                        def create_click_handler(segment_img,original_segment ):
                            def handle_click():
                                # 保存当前状态
                                self.original_image = self.temp_image
                                # 更新主显示区域
                                self.temp_image = original_segment 
                                # Set flags and store current view
                                self.is_segment_view = True
                                self.current_view_image = segment_img
                                # 更新缩放因子为1.0开始
                                self.scale_factor = 1.0
                                # 刷新显示
                                self._display_main_image()
                                # 恢复原始图像用于右侧显示
                                #self.temp_image = self.original_image
                            return handle_click
                        
                        segment_label.mousePressEvent = lambda _, img=segment_image, orig=segment_image.copy(): create_click_handler(img, orig)()
                        
                        self.segments_layout.addWidget(container)


    def _display_main_image(self):
        """更新主图像显示"""
        if self.temp_image:
            width = int(self.temp_image.width * self.scale_factor)
            height = int(self.temp_image.height * self.scale_factor)
            scaled_image = self.temp_image.resize((width, height), Image.LANCZOS)

            display_image = QImage(self.label_image.width(), self.label_image.height(), QImage.Format_RGB888)
            display_image.fill(Qt.white)

            x = (self.label_image.width() - width) // 2
            y = (self.label_image.height() - height) // 2

            qimage = QImage(scaled_image.tobytes(), scaled_image.width, scaled_image.height, QImage.Format_RGB888)
            painter = QPainter(display_image)
            painter.drawImage(x, y, qimage)
            painter.end()

            pixmap = QPixmap.fromImage(display_image)
            self.label_image.setPixmap(pixmap)

    def _get_scale_factor(self):
        """获取像素到厘米的转换比例"""
        if not hasattr(self, 'original_image'):
            return 1.0
            
        # Get image dimensions in pixels
        img_width = self.original_image.width
        img_height = self.original_image.height
        
        # Calculate scale factors based on proportion
        # For a region's pixels / total pixels * A4 dimension
        width_scale = self.A4_WIDTH_CM / img_width   # cm per pixel in width
        height_scale = self.A4_HEIGHT_CM / img_height  # cm per pixel in height
        
        # Use the smaller scale to ensure measurements fit within A4
        scale_factor = min(width_scale, height_scale)
        
        return scale_factor
    def mouse_press_event(self, event):
        """捕获鼠标点击事件"""
        if self.temp_image is None:
            return

        # Calculate clicked point in current view
        x = int((event.position().x() - (
                self.label_image.width() - self.temp_image.width * self.scale_factor) // 2) / self.scale_factor)
        y = int((event.position().y() - (
                self.label_image.height() - self.temp_image.height * self.scale_factor) // 2) / self.scale_factor)

        if not (0 <= x < self.temp_image.width and 0 <= y < self.temp_image.height):
            return

        # Check if this is the third point
        if len(self.points) == 2:
            reply = QMessageBox.question(
                self,
                "确认",
                "是否要重新计算两点之间的距离？\n"
                "选择'是'将重置图像，\n"
                "选择'否'将清除已有点。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                # Clear points and reset image to original segmented view
                self.points.clear()
                if hasattr(self, 'original_image'):
                    self.temp_image = self.original_image.copy()
                    # Reset view flags
                    self.is_segment_view = False
                    self.current_view_image = None
                    self.scale_factor = 1.0
                    # Display the segmented original image
                    self.segment_image()
            else:
                # Just clear points without resetting image
                self.points.clear()
                self.temp_image = self.temp_image.copy()
                self._display_main_image()
            return

        # Store clicked coordinates
        self.points.append((x, y))
        self.result_text.append(f"点击点: ({x}, {y})")

        # Draw point at clicked location
        draw = ImageDraw.Draw(self.temp_image)
        radius = 5
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i ** 2 + j ** 2 <= radius ** 2:
                    new_x = x + i
                    new_y = y + j
                    if 0 <= new_x < self.temp_image.width and 0 <= new_y < self.temp_image.height:
                        draw.point((new_x, new_y), fill=(255, 0, 0))

        # Calculate and display distance if we have two points
        if len(self.points) == 2:
            # Draw line between points
            draw.line(self.points, fill=(255, 0, 0), width=4)
            
            # Calculate distance
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            real_distance = pixel_distance * self._get_scale_factor()  # Convert to cm

            # Save results
            self.results.append({
                "Type": "Distance",
                "Value": real_distance,
                "PixelValue": pixel_distance,
                "Points": self.points.copy()
            })
            self.result_text.append(f"距离: {real_distance:.2f} 厘米")

        self._display_main_image()
    def _get_region_info(self, region_number):
        """获取指定区域的信息"""
        if not region_number:
            return None
            
        for result in self.results:
            if result["Type"] == "Segment" and result["Shape"] == region_number:
                return {
                    'vertices': result["Vertices"],
                    'area': result["Area"],
                    'actual_area': result["ActualArea"]
                }
        return None

    def wheel_event(self, event):
        """鼠标滚轮事件，用于放大缩小图像"""
        if self.temp_image:
            # 根据滚轮方向调整缩放比例
            delta = event.angleDelta().y()
            if delta > 0:
                self.scale_factor *= 1.1  # 放大
            else:
                self.scale_factor *= 0.9  # 缩小

            # 限制缩放比例范围，改为 0.1 到 10.0
            self.scale_factor = max(0.1, min(self.scale_factor, 10.0))

            if self.is_segment_view and self.current_view_image:
                temp = self.temp_image
                self.temp_image = self.current_view_image
                self._display_main_image()
                self.temp_image = temp
            else:
                self._display_main_image()
    def reset_view(self):
        """重置视图到原始状态"""
        self.is_segment_view = False
        self.current_view_image = None
        self.scale_factor = 1.0
        if hasattr(self, 'original_image'):
            self.temp_image = self.original_image
            self._display_main_image()
    '''def calculate_distance(self, point1, point2):
        """计算两点之间的距离"""
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)'''

    def _calculate_calibration_factor(self, result):
        """Calculate area calibration factor based on reference areas"""
        shape = result["Shape"]
        reference_areas = {
            1: self.REFERENCE_AREA_1,
            2: self.REFERENCE_AREA_2,
            3: self.REFERENCE_AREA_3,
            4: self.REFERENCE_AREA_4,
            5: self.REFERENCE_AREA_5,
            6: self.REFERENCE_AREA_6,
            7: self.REFERENCE_AREA_7
        }
        
        if shape in reference_areas:
            return reference_areas[shape] / (result['Area'] * (self._get_scale_factor() ** 2))
        else:
            # 如果是未知区域，使用区域1和2的平均校准因子
            calibration_1 = self.REFERENCE_AREA_1 / (self.results[0]['Area'] * (self._get_scale_factor() ** 2))
            calibration_2 = self.REFERENCE_AREA_2 / (self.results[1]['Area'] * (self._get_scale_factor() ** 2))
            return (calibration_1 + calibration_2) / 2

    def clear_points(self):
        """清除已选择的点"""
        self.points.clear()
        self.temp_image = self.image.copy()
        self.display_image()
        self.result_text.append("已清除所有点")

    def save_results(self):
        """保存结果为 CSV 文件"""
        if not self.results:
            QMessageBox.warning(self, "警告", "没有结果可保存！")
            return

        try:
            # 定义区域类型映射
            region_types = {
                1: "全树",
                2: "上枝",
                3: "枝干",
                4: "右上枝",
                5: "左上枝",
                6: "右下枝",
                7: "左下枝"
            }

            formatted_results = []
            for result in self.results:
                if result["Type"] == "Segment":
                    region_number = result["Shape"]
                    region_type = region_types.get(region_number, "未知区域")
                    formatted_results.append({
                        "类型": region_type,
                        "编号": result["Shape"],
                        "像素面积": f"{result['Area']:.2f}",
                        "实际面积(cm²)": f"{self.black_actual_areas.get(result['Shape'], 0):.2f}",
                        "顶点坐标": ';'.join([f"({x},{y})" for x, y in result["Vertices"]])
                    })
                elif result["Type"] == "Distance":
                    formatted_results.append({
                        "类型": "距离测量",
                        "编号": "-",
                        "像素距离": f"{result['PixelValue']:.2f}",
                        "实际距离(cm)": f"{result['Value']:.2f}",
                        "测量点": ';'.join([f"({x},{y})" for x, y in result["Points"]])
                    })
                

            # 将格式化后的结果转换为DataFrame
            df = pd.DataFrame(formatted_results)

            # 获取保存路径
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(
                self,
                "保存文件",
                "",
                "CSV Files (*.csv)"
            )

            if file_path:
                try:
                    # 尝试以写入模式打开文件以测试权限
                    with open(file_path, 'w', encoding='utf-8-sig', newline='') as test_file:
                        pass

                    # 如果文件可写，则保存数据
                    df.to_csv(
                        file_path,
                        index=False,
                        encoding='utf-8-sig',
                        na_rep='-'
                    )
                    QMessageBox.information(self, "成功", f"结果已保存到 {file_path}")

                except PermissionError:
                    QMessageBox.critical(
                        self,
                        "错误",
                        "无法保存文件，请确保：\n"
                        "1. 选择的位置有写入权限\n"
                        "2. 文件未被其他程序占用\n"
                        "3. 尝试保存到其他位置"
                    )
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"保存文件时发生错误：{str(e)}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理数据时发生错误：{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())