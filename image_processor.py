import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local

class ImageProcessor:
    @staticmethod
    def process_image(image_path):
        """处理图像并检测四边形区域"""
        # 读取原始图像
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise Exception("无法读取图像")

        # 调整图像大小
        ratio = original_img.shape[0] / 500.0
        img_resize = imutils.resize(original_img, height=550)

        # 图像预处理
        gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # 使用更严格的Canny参数
        edged_img = cv2.Canny(blurred_image, 50, 200)

        # 先进行形态学闭运算确保轮廓闭合
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_img = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, kernel)

        # 查找图像中的轮廓，使用RETR_EXTERNAL只检测全部轮廓
        cnts, hierarchy = cv2.findContours(closed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选闭合的轮廓
        closed_contours = []
        for cnt in cnts:
            # 计算轮廓的周长
            perimeter = cv2.arcLength(cnt, True)
            # 计算轮廓的面积
            area = cv2.contourArea(cnt)
            
            # 使用形状指标来判断是否为闭合区域
            # 圆度 = 4π×面积/周长²，完美的圆形圆度为1
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 计算轮廓的凸包
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
            # 计算轮廓与其凸包的面积比
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # 同时满足以下条件的被认为是有效的闭合区域:
            # 1. 圆度大于0.2（排除过于复杂或不规则的形状）
            # 2. 与凸包面积比大于0.8（说明是相对完整的形状）
            # 3. 面积大于阈值（排除太小的区域）
            if circularity > 0.5 and solidity > 0.8 and area > 500:
                closed_contours.append(cnt)

        # 根据轮廓面积排序
        closed_contours = sorted(closed_contours, key=cv2.contourArea, reverse=True)[:20]

        # 创建结果图像
        result_image = img_resize.copy()
        
        # 使用已经确认为闭合的轮廓继续处理
        cnts = closed_contours
            
        # 生成随机颜色
        colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) 
                for _ in range(len(cnts))]

        # 存储检测到的四边形
        quadrilaterals = []

        # 检测四边形
        for i, c in enumerate(cnts):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 降低多边形近似参数以提高精度

            if len(approx) == 4 and cv2.isContourConvex(approx):  # 添加凸性检查
                # 计算轮廓面积
                area = cv2.contourArea(approx)
                
                # 排除太小的区域
                if area > 500:
                    # 绘制轮廓
                    cv2.drawContours(result_image, [approx], -1, (0, 255, 0), 2)

                    # 精确定位顶点
                    refined_points = []
                    for point in approx:
                        pt = tuple(point[0])
                        
                        # 在原始边缘图像中查找更精确的角点位置
                        window_size = 5
                        x, y = pt
                        window = edged_img[
                            max(0, y-window_size):min(edged_img.shape[0], y+window_size+1),
                            max(0, x-window_size):min(edged_img.shape[1], x+window_size+1)
                        ]
                        
                        if window.size > 0:
                            # 使用质心法找到更精确的角点
                            moments = cv2.moments(window)
                            if moments["m00"] != 0:
                                cx = int(moments["m10"] / moments["m00"])
                                cy = int(moments["m01"] / moments["m00"])
                                
                                # 调整回原始坐标系
                                refined_x = max(0, x-window_size) + cx
                                refined_y = max(0, y-window_size) + cy
                                
                                # 绘制精确的角点
                                cv2.circle(result_image, (refined_x, refined_y), 2, (0, 0, 255), -1)
                                cv2.circle(result_image, (refined_x, refined_y), 4, (255, 255, 255), 1)
                                
                                refined_points.append((refined_x, refined_y))
                            else:
                                refined_points.append(pt)
                        else:
                            refined_points.append(pt)

                    # 使用精确的顶点
                    quadrilaterals.append(np.array(refined_points))
        # 计算每个四边形的属性
        quad_results = []
        for i, quad in enumerate(quadrilaterals):
            area = cv2.contourArea(quad)
            # Convert numpy array vertices to list of tuples
            vertices = quad.reshape(-1, 2).tolist()
            
            quad_results.append({
                'index': i + 1,
                'vertices': vertices,  # Now it's a list of [x,y] coordinates
                'area': area
            })

        return {
            'original': original_img,
            'processed': result_image,
            'quads': quad_results,
            'ratio': ratio
        }
    @staticmethod
    def rotate_to_horizontal(image, vertices):
        """将图像旋转至水平方向，如果已经水平则不旋转"""
        # 将顶点转换为numpy数组
        vertices = np.array(vertices)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(vertices)
        angle = rect[-1]
        
        # 检查是否已经水平
        # 计算水平和垂直方向的边长
        width = rect[1][0]
        height = rect[1][1]
        
        # 计算长宽比
        aspect_ratio = max(width, height) / (min(width, height) if min(width, height) > 0 else 1)
        
        # 如果长宽比接近1或角度接近0/90度（允许5度误差），认为是水平的
        is_horizontal = (abs(angle) < 5 or abs(angle - 90) < 5) and aspect_ratio > 1.2
        
        if is_horizontal:
            return image  # 如果已经水平，直接返回原图
        
        # 否则进行旋转处理
        if angle < -45:
            angle = 90 + angle
        
        # 获取图像中心点
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # 获取旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 执行旋转
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated