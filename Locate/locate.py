import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from Locate.Unet import OptimizedUNet  # 模型结构
"""
此代码定义了一个函数，接受一张图片（非图片路径），从原图中定位出车牌位置并对车牌进行矫正处理，输出带车牌检测框原图和矫正后的车牌位置图片
"""
def order_points(pts):
    """将输入的四个点排序为左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上：x+y最小
    rect[2] = pts[np.argmax(s)]  # 右下：x+y最大

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上：x-y最小
    rect[3] = pts[np.argmax(diff)]  # 左下：x-y最大
    return rect


def rectify_license_plate(image, plate_corners):
    # 将输入的四个点转换为numpy数组
    pts = np.array(plate_corners, dtype=np.float32)

    # 对点进行排序
    ordered_pts = order_points(pts)
    (tl, tr, br, bl) = ordered_pts

    # 计算目标矩形的宽度和高度
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    # 定义目标矩形的四个点
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

    # 应用变换
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

def predict_license_plate(model_path, image, device='cuda'):
    """
    使用模型预测车牌位置并矫正车牌区域。

    Args:
        model_path (str): 已训练模型的路径。
        image (numpy.ndarray): 输入图片。
        device (str): 使用的设备 ('cuda' 或 'cpu')。

    Returns:
        original_image_with_box (numpy.ndarray): 带有车牌框选区域的原图。
        corrected_plate (numpy.ndarray): 矫正后的车牌区域图片。
    """
    # 设备设置
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = OptimizedUNet(in_ch=3, out_ch=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # 加载输入图片
    original_image = image
    assert original_image is not None, f"图片加载失败"
    orig_h, orig_w, _ = original_image.shape

    # 预处理图片
    resized_image = cv2.resize(original_image, (640, 640), interpolation=cv2.INTER_LINEAR)
    normalized_image = resized_image.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(normalized_image.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        pred_mask = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()

    # 后处理掩码
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

    # 调整掩码回到原图尺寸
    processed_mask_resized = cv2.resize(processed_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 检测轮廓并提取车牌区域
    contours, _ = cv2.findContours(processed_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到面积最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 使用凸包确保为四边形
        hull = cv2.convexHull(max_contour)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        # 如果逼近的点不足四个，强制拟合为四边形
        if len(approx) != 4:
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            approx = np.int0(box)

        vertices = approx.reshape(4, 2)
        original_image_with_box = original_image.copy()
        cv2.drawContours(original_image_with_box, [vertices], -1, (0, 255, 0), 2)

        # 矫正车牌区域
        corrected_plate = rectify_license_plate(original_image, vertices)
        warped_resized = cv2.resize(corrected_plate, (160, 32), interpolation=cv2.INTER_LINEAR)

        return original_image_with_box, warped_resized

    print("未检测到车牌或无法拟合四边形")
    return original_image, None


# 测试函数
if __name__ == "__main__":
    model_path = "../weights/best_model_mix.pth"  # 替换为您的模型路径
    image_path = "../data/CCPD2020/ccpd_green/test/03125-89_263-177&502_477&597-464&581_177&597_185&513_477&502-0_0_3_27_29_25_33_33-102-60.jpg"  # 测试图片路径
    image = cv2.imread(image_path)
    original_image_with_box, corrected_plate = predict_license_plate(model_path, image)
    cv2.imwrite("1.jpg",corrected_plate)
    cv2.imwrite("2.jpg",original_image_with_box)

    # 使用matplotlib显示结果
    plt.figure(figsize=(10, 5))

    # 显示带框的原图
    plt.subplot(1, 2, 1)
    plt.title("Original Image with Box")
    plt.imshow(cv2.cvtColor(original_image_with_box, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 显示矫正后的车牌
    if corrected_plate is not None:
        plt.subplot(1, 2, 2)
        plt.title("Corrected License Plate")
        plt.imshow(cv2.cvtColor(corrected_plate, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.tight_layout()
    plt.show()