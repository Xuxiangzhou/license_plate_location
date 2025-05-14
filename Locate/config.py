import albumentations as A
"""
此处用于配制Unet模型训练参数
"""
class Config:
    # 训练数据集
    train_image_dir = "../data/CCPD2020/ccpd_green/train"
    train_mask_dir = "../data/CCPD2020/ccpd_green/train-mask"
    # 验证数据集
    val_image_dir = "../data/CCPD2020/ccpd_green/val"
    val_mask_dir = "../data/CCPD2020/ccpd_green/val-mask"
    val_results_dir = "../data/results/val_results"
    # 测试数据集
    test_image_dir = "../data/CCPD2019/ccpd_fn"
    test_mask_dir = "../data/CCPD2019/ccpd_fn-mask"
    # 模型路径
    model_path = "../weights/best_model_mix.pth"
    # 模型训练检查点
    model_dir = "../model"

    result_path = "data/result"     #结果保存路径
    norm_mean = [0.485, 0.456, 0.406]  # ImageNet均值
    norm_std = [0.229, 0.224, 0.225]  # ImageNet标准差
    # GUI界面背景图片
    background_path = "../ui/BG.png"
    # 图像参数
    in_channels = 3
    out_channels = 1            # 二值分割建议单通道
    img_size = (640,640)       # 输入尺寸

    # 训练参数
    augment = True    #数据增强
    num_workers = 4          #多进程
    compile_model = False
    batch_size =8        #单次传递给模型的数据量
    accum_steps =2        # 梯度累积步数
    epochs = 100                 # 总训练轮次
    lr= 1e-3                    #初始学习率
    reconize_model = "../Recognition/checkpoints/best_model.pth"     #用于GUI识别时采用车牌定位模型