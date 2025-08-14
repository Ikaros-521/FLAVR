import os
import torch
import cv2
import numpy as np
from PIL import Image
import threading
import queue
from collections import deque
import argparse
import time

# 导入FLAVR相关模块
from model.FLAVR_arch import UNet_3D_3D
from torchvision import transforms

class SimpleRealtimeFLAVR:
    def __init__(self, model_path, factor=2, device='cuda', scale_factor=1.0):
        self.model_path = model_path
        self.factor = factor
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scale_factor = scale_factor
        
        # 加载模型
        self.load_model()
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=10)
        self.output_queue = queue.Queue(maxsize=20)
        self.original_size = None  # 存储原始尺寸
        
        # 处理线程
        self.running = False
        self.processing_thread = None
        
        # 统计
        self.fps = 0
        self.fps_counter = 0
        self.fps_timer = time.time()
        
        print(f"实时FLAVR初始化完成 - 设备: {self.device}, 倍数: {factor}x, 缩放: {scale_factor}x")
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise Exception(f"模型文件不存在: {self.model_path}")
        
        # 创建模型
        self.model = UNet_3D_3D(
            "unet_18", 
            n_inputs=4, 
            n_outputs=self.factor-1, 
            joinType="concat", 
            upmode="transpose"
        )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        saved_state_dict = checkpoint['state_dict']
        saved_state_dict = {k.partition("module.")[-1]: v for k, v in saved_state_dict.items()}
        self.model.load_state_dict(saved_state_dict)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("模型加载成功!")
    
    def preprocess_frame(self, frame):
        """预处理帧"""
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # 获取原始尺寸
        original_size = pil_image.size
        
        # 应用缩放因子
        scaled_width = int(original_size[0] * self.scale_factor)
        scaled_height = int(original_size[1] * self.scale_factor)
        
        # 调整到8的倍数（FLAVR要求）
        new_width = 8 * (scaled_width // 8)
        new_height = 8 * (scaled_height // 8)
        
        # 调整尺寸并转换为tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((new_height, new_width))
        ])
        
        return transform(pil_image)
    
    def postprocess_frame(self, tensor, original_size=None):
        """后处理帧"""
        img = tensor.data.mul(255.).clamp(0, 255).round()
        im = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        # 根据缩放因子决定显示尺寸
        if original_size is not None:
            current_height, current_width = im.shape[:2]
            
            if self.scale_factor != 1.0:
                # 如果设置了缩放，显示尺寸也相应缩放
                display_width = int(original_size[0] * self.scale_factor)
                display_height = int(original_size[1] * self.scale_factor)
                
                # 调整到8的倍数
                display_width = 8 * (display_width // 8)
                display_height = 8 * (display_height // 8)
                
                if current_width != display_width or current_height != display_height:
                    print(f"补帧画面调整显示尺寸: {current_width}x{current_height} -> {display_width}x{display_height}")
                    im = cv2.resize(im, (display_width, display_height), interpolation=cv2.INTER_LANCZOS4)
            else:
                # 如果缩放因子是1.0，恢复到原始尺寸
                target_width, target_height = original_size
                if current_width != target_width or current_height != target_height:
                    print(f"补帧画面恢复到原始尺寸: {current_width}x{current_height} -> {target_width}x{target_height}")
                    im = cv2.resize(im, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        return im
    
    def interpolate_frames(self, frames):
        """插值处理"""
        if len(frames) < 4:
            return []
        
        # 准备输入
        inputs = [frame.to(self.device).unsqueeze(0) for frame in frames[-4:]]
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        return [of.squeeze(0).cpu() for of in outputs]
    
    def processing_worker(self):
        """处理线程"""
        print("处理线程启动")
        
        while self.running:
            if len(self.frame_buffer) >= 4:
                frames = list(self.frame_buffer)[-4:]
                interpolated = self.interpolate_frames(frames)
                
                for frame in interpolated:
                    if not self.output_queue.full():
                        self.output_queue.put(frame)
                    else:
                        try:
                            self.output_queue.get_nowait()
                            self.output_queue.put(frame)
                        except queue.Empty:
                            pass
            
            time.sleep(0.001)
        
        print("处理线程结束")
    
    def start(self):
        """启动处理"""
        self.running = True
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("处理已启动")
    
    def stop(self):
        """停止处理"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        print("处理已停止")
    
    def add_frame(self, frame):
        """添加帧"""
        # 记录原始尺寸（仅在第一次）
        if self.original_size is None:
            height, width = frame.shape[:2]
            self.original_size = (width, height)
            print(f"记录原始尺寸: {width}x{height}")
        
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
    
    def get_interpolated_frame(self):
        """获取插值帧"""
        try:
            if not self.output_queue.empty():
                frame_tensor = self.output_queue.get_nowait()
                return self.postprocess_frame(frame_tensor, self.original_size)
        except queue.Empty:
            pass
        return None
    
    def update_fps(self):
        """更新FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time

def main():
    parser = argparse.ArgumentParser(description="实时FLAVR补帧")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--factor", type=int, default=2, choices=[2, 4, 8], help="插值倍数")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--width", type=int, help="手动设置摄像头宽度")
    parser.add_argument("--height", type=int, help="手动设置摄像头高度")
    parser.add_argument("--scale", type=float, default=1.0, help="缩放因子 (1.0=原始尺寸, 0.5=一半尺寸, 2.0=两倍尺寸)")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    args = parser.parse_args()
    
    try:
        # 创建插值器
        interpolator = SimpleRealtimeFLAVR(
            model_path=args.model,
            factor=args.factor,
            device=args.device,
            scale_factor=args.scale
        )
        
        # 启动处理
        interpolator.start()
        
        # 打开摄像头
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {args.camera}")
            return
        
        # 如果指定了目标尺寸，则设置
        if args.width and args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            print(f"设置目标分辨率: {args.width}x{args.height}")
        else:
            # 尝试设置常见的高分辨率
            resolutions = [
                (1920, 1080),  # 1080p
                (1280, 720),   # 720p
                (800, 600),    # SVGA
                (640, 480),    # VGA
            ]
            
            for width, height in resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if actual_width >= width and actual_height >= height:
                    print(f"设置分辨率: {actual_width}x{actual_height}")
                    break
        
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 检查最终设置
        final_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        final_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        final_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"摄像头最终设置: {final_width}x{final_height} @ {final_fps:.1f}fps")
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {args.camera}")
            return
        
        print("摄像头已打开，按 'q' 退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 添加帧
            interpolator.add_frame(frame)
            
            # 获取插值帧
            interpolated = interpolator.get_interpolated_frame()
            
            # 根据缩放因子调整原始帧显示尺寸
            if args.scale != 1.0:
                display_width = int(frame.shape[1] * args.scale)
                display_height = int(frame.shape[0] * args.scale)
                
                # 调整到8的倍数
                display_width = 8 * (display_width // 8)
                display_height = 8 * (display_height // 8)
                
                if frame.shape[1] != display_width or frame.shape[0] != display_height:
                    frame_display = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LANCZOS4)
                else:
                    frame_display = frame
            else:
                frame_display = frame
            
            # 显示原始帧
            cv2.imshow('Original', frame_display)
            
            # 显示插值帧
            if interpolated is not None:
                cv2.imshow(f'Interpolated ({args.factor}x)', interpolated)
                interpolator.update_fps()
                
                # 显示FPS
                cv2.putText(frame, f"FPS: {interpolator.fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理
        interpolator.stop()
        cap.release()
        cv2.destroyAllWindows()
        
        print("程序已退出")
        
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        if 'interpolator' in locals():
            interpolator.stop()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 