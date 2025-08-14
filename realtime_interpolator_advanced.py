import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image
import threading
import queue
from collections import deque
import argparse

# 导入FLAVR相关模块
from model.FLAVR_arch import UNet_3D_3D
from dataset.transforms import ToTensorVideo, Resize
from torchvision import transforms

class AdvancedRealtimeFLAVRInterpolator:
    def __init__(self, model_path, factor=2, buffer_size=15, device='cuda', 
                 scale_factor=1.0, enable_optimization=True):
        self.model_path = model_path
        self.factor = factor
        self.buffer_size = buffer_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scale_factor = scale_factor
        self.enable_optimization = enable_optimization
        
        # 初始化模型
        self.model = None
        self.load_model()
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=buffer_size)
        self.output_buffer = queue.Queue(maxsize=buffer_size * factor * 2)
        self.original_size = None  # 存储原始尺寸
        
        # 处理线程
        self.processing_thread = None
        self.running = False
        
        # 性能统计
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.avg_fps = 0
        self.processing_times = deque(maxlen=100)
        
        # 模型参数
        self.nbr_frame = 4
        self.n_outputs = factor - 1
        
        # 性能优化
        if self.enable_optimization:
            self.setup_optimization()
        
        print(f"高级实时FLAVR插值器初始化完成:")
        print(f"  - 设备: {self.device}")
        print(f"  - 插值倍数: {factor}x")
        print(f"  - 缓冲区大小: {buffer_size}")
        print(f"  - 缩放因子: {scale_factor}x")
    
    def setup_optimization(self):
        """设置性能优化"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            print("CUDA优化已启用")
    
    def load_model(self):
        """加载FLAVR模型"""
        try:
            print(f"正在加载模型: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise Exception(f"模型文件不存在: {self.model_path}")
            
            # 模型参数
            model_arch = "unet_18"
            nbr_frame = 4
            joinType = "concat"
            n_outputs = self.factor - 1
            
            # 创建模型
            self.model = UNet_3D_3D(
                model_arch.lower(), 
                n_inputs=nbr_frame, 
                n_outputs=n_outputs, 
                joinType=joinType, 
                upmode="transpose"
            )
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            saved_state_dict = checkpoint['state_dict']
            saved_state_dict = {k.partition("module.")[-1]: v for k, v in saved_state_dict.items()}
            self.model.load_state_dict(saved_state_dict)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 预热模型
            self.warmup_model()
            
            print("模型加载成功!")
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
    
    def warmup_model(self):
        """预热模型以提高首次推理速度"""
        try:
            print("正在预热模型...")
            
            # 创建随机输入进行预热 - 使用较小的尺寸以节省内存
            dummy_inputs = []
            for _ in range(4):  # 使用固定的4帧
                dummy_tensor = torch.randn(1, 3, 128, 128)  # 使用较小尺寸预热
                dummy_inputs.append(dummy_tensor.to(self.device))
            
            with torch.no_grad():
                for _ in range(3):  # 运行几次预热
                    _ = self.model(dummy_inputs)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("模型预热完成")
            
        except Exception as e:
            print(f"模型预热失败: {str(e)}")
    
    def preprocess_frame(self, frame):
        """预处理单帧"""
        try:
            # 转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(frame_rgb)
            
            # 获取原始尺寸
            original_size = pil_image.size
            
            # 应用缩放因子
            scaled_width = int(original_size[0] * self.scale_factor)
            scaled_height = int(original_size[1] * self.scale_factor)
            
            # 调整到8的倍数（FLAVR要求）
            new_width = 8 * (scaled_width // 8)
            new_height = 8 * (scaled_height // 8)
            
            # 应用变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((new_height, new_width), antialias=True)
            ])
            
            tensor = transform(pil_image)
            return tensor
            
        except Exception as e:
            print(f"帧预处理失败: {str(e)}")
            return None
    
    def postprocess_frame(self, tensor, original_size=None):
        """后处理张量为图像"""
        try:
            # 转换为numpy数组
            img = tensor.data.mul(255.).clamp(0, 255).round()
            im = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            # 转换回BGR格式
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
            
        except Exception as e:
            print(f"帧后处理失败: {str(e)}")
            return None
    
    def interpolate_frames(self, frames):
        """对帧序列进行插值"""
        try:
            start_time = time.time()
            
            if len(frames) < self.nbr_frame:
                # 帧数不足，复制最后一帧
                while len(frames) < self.nbr_frame:
                    frames.append(frames[-1])
            
            # 准备输入
            inputs = [frame.to(self.device).unsqueeze(0) for frame in frames]
            
            with torch.no_grad():
                output_frames = self.model(inputs)
            
            # 处理输出
            outputs = []
            for of in output_frames:
                output_frame = of.squeeze(0).cpu()
                outputs.append(output_frame)
            
            # 记录处理时间
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return outputs
            
        except Exception as e:
            print(f"插值处理失败: {str(e)}")
            return []
    
    def processing_worker(self):
        """处理线程工作函数"""
        print("处理线程启动")
        
        while self.running:
            try:
                # 检查缓冲区是否有足够的帧
                if len(self.frame_buffer) >= self.nbr_frame:
                    # 获取最新的4帧
                    frames = list(self.frame_buffer)[-self.nbr_frame:]
                    
                    # 进行插值
                    interpolated_frames = self.interpolate_frames(frames)
                    
                    # 将插值帧放入输出缓冲区
                    for frame in interpolated_frames:
                        if not self.output_buffer.full():
                            self.output_buffer.put(frame)
                        else:
                            # 缓冲区满，丢弃最旧的帧
                            try:
                                self.output_buffer.get_nowait()
                                self.output_buffer.put(frame)
                            except queue.Empty:
                                pass
                
                # 动态调整休眠时间
                if len(self.processing_times) > 0:
                    avg_time = np.mean(self.processing_times)
                    sleep_time = max(0.001, min(0.01, avg_time * 0.1))
                else:
                    sleep_time = 0.001
                
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"处理线程错误: {str(e)}")
                time.sleep(0.1)
        
        print("处理线程结束")
    
    def start_processing(self):
        """启动处理线程"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.running = True
            self.processing_thread = threading.Thread(target=self.processing_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("处理线程已启动")
    
    def stop_processing(self):
        """停止处理线程"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
            print("处理线程已停止")
    
    def add_frame(self, frame):
        """添加新帧到缓冲区"""
        try:
            # 记录原始尺寸（仅在第一次）
            if self.original_size is None:
                height, width = frame.shape[:2]
                self.original_size = (width, height)
                print(f"记录原始尺寸: {width}x{height}")
            
            # 预处理帧
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is not None:
                self.frame_buffer.append(processed_frame)
        except Exception as e:
            print(f"添加帧失败: {str(e)}")
    
    def get_interpolated_frame(self):
        """获取插值后的帧"""
        try:
            if not self.output_buffer.empty():
                frame_tensor = self.output_buffer.get_nowait()
                frame = self.postprocess_frame(frame_tensor, self.original_size)
                return frame
            else:
                return None
        except queue.Empty:
            return None
        except Exception as e:
            print(f"获取插值帧失败: {str(e)}")
            return None
    
    def update_fps(self):
        """更新FPS统计"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            self.avg_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def get_status_info(self):
        """获取详细状态信息"""
        buffer_usage = len(self.frame_buffer) / self.buffer_size * 100
        output_queue_size = self.output_buffer.qsize()
        
        # 计算平均处理时间
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # 获取内存使用情况
        memory_info = self.get_memory_usage()
        
        status = {
            'fps': f"{self.avg_fps:.1f}",
            'buffer_usage': f"{buffer_usage:.1f}%",
            'output_queue': output_queue_size,
            'avg_processing_time': f"{avg_processing_time*1000:.1f}ms",
            'device': str(self.device),
            'factor': self.factor,
            'memory': memory_info
        }
        
        return status
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        try:
            # GPU内存
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                return f"GPU:{allocated:.1f}GB/{reserved:.1f}GB"
            else:
                return "CPU模式"
        except:
            return "N/A"

def create_video_source(source_type, source_path=None, camera_index=0, target_width=None, target_height=None):
    """创建视频源"""
    if source_type == "camera":
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头 {camera_index}")
        
        # 如果指定了目标尺寸，则设置
        if target_width and target_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            print(f"设置目标分辨率: {target_width}x{target_height}")
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
        
        return cap
    elif source_type == "video":
        cap = cv2.VideoCapture(source_path)
        return cap
    elif source_type == "rtsp":
        cap = cv2.VideoCapture(source_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    else:
        raise ValueError(f"不支持的视频源类型: {source_type}")

def main():
    parser = argparse.ArgumentParser(description="高级实时FLAVR视频插值")
    parser.add_argument("--factor", type=int, default=2, choices=[2, 4, 8], help="插值倍数")
    parser.add_argument("--source", type=str, default="camera", 
                       choices=["camera", "video", "rtsp"], help="视频源类型")
    parser.add_argument("--source_path", type=str, help="视频文件路径或RTSP URL")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--buffer", type=int, default=15, help="缓冲区大小")
    parser.add_argument("--width", type=int, help="手动设置摄像头宽度")
    parser.add_argument("--height", type=int, help="手动设置摄像头高度")
    parser.add_argument("--scale", type=float, default=1.0, help="缩放因子 (1.0=原始尺寸, 0.5=一半尺寸, 2.0=两倍尺寸)")
 
    parser.add_argument("--device", type=str, default="cuda", help="计算设备 (cuda/cpu)")
    parser.add_argument("--no_optimization", action="store_true", help="禁用性能优化")
    parser.add_argument("--save_output", type=str, help="保存输出视频路径")
    
    args = parser.parse_args()
    
    try:
        if args.factor == 2:
            model_path = "models/flavr_2x.pth"
        elif args.factor == 4:
            model_path = "models/flavr_4x.pth"
        else:
            model_path = "models/flavr_8x.pth"

        # 创建插值器
        interpolator = AdvancedRealtimeFLAVRInterpolator(
            model_path=model_path,
            factor=args.factor,
            buffer_size=args.buffer,
            device=args.device,
            scale_factor=args.scale,
            enable_optimization=not args.no_optimization
        )
        
        # 启动处理线程
        interpolator.start_processing()
        
        # 创建视频源
        cap = create_video_source(
            args.source, 
            args.source_path, 
            args.camera,
            args.width,
            args.height
        )
        
        if not cap.isOpened():
            print(f"无法打开视频源: {args.source}")
            return
        
        print(f"视频源已打开")
        print("按 'q' 退出，按 's' 显示/隐藏状态信息，按 'r' 重置统计")
        
        show_status = True
        
        # 视频保存设置
        video_writer = None
        if args.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 使用原始尺寸，但需要先获取一帧来确定尺寸
            ret, test_frame = cap.read()
            if ret:
                height, width = test_frame.shape[:2]

                print(f"原始尺寸: {width}x{height}")
                # 调整到8的倍数
                width = 8 * (width // 8)
                height = 8 * (height // 8)
                video_writer = cv2.VideoWriter(
                    args.save_output, fourcc, 30, 
                    (width, height)
                )
                # 重新设置摄像头位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break
            
            frame_count += 1
            
            # 添加帧到插值器
            interpolator.add_frame(frame)
            
            # 获取插值后的帧
            interpolated_frame = interpolator.get_interpolated_frame()
            
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
            
            # 显示插值后的帧
            if interpolated_frame is not None:
                cv2.imshow(f'Interpolated ({args.factor}x)', interpolated_frame)
                interpolator.update_fps()
                
                # 打印尺寸信息（调试用）
                if frame_count == 1:
                    print(f"原始帧尺寸: {frame.shape}")
                    print(f"插值帧尺寸: {interpolated_frame.shape}")
                
                # 保存视频
                if video_writer:
                    video_writer.write(interpolated_frame)
            
            # 显示状态信息
            if show_status:
                status = interpolator.get_status_info()
                status_text = [
                    f"FPS: {status['fps']}",
                    f"Buffer: {status['buffer_usage']}",
                    f"Queue: {status['output_queue']}",
                    f"Process: {status['avg_processing_time']}",
                    f"Memory: {status['memory']}",
                    f"Factor: {status['factor']}x"
                ]
                
                # 在原始帧上显示状态
                for i, text in enumerate(status_text):
                    cv2.putText(frame, text, (10, 30 + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_status = not show_status
            elif key == ord('r'):
                # 重置统计
                interpolator.fps_counter = 0
                interpolator.fps_timer = time.time()
                interpolator.avg_fps = 0
                interpolator.processing_times.clear()
                print("统计已重置")
        
        # 清理资源
        interpolator.stop_processing()
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # 显示最终统计
        total_time = time.time() - start_time
        print(f"\n最终统计:")
        print(f"  - 总帧数: {frame_count}")
        print(f"  - 总时间: {total_time:.2f}秒")
        print(f"  - 平均FPS: {frame_count/total_time:.2f}")
        
        print("程序已退出")
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'interpolator' in locals():
            interpolator.stop_processing()
        if 'cap' in locals():
            cap.release()
        if 'video_writer' in locals() and video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 