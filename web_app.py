import os
import sys
import time
import tempfile
import torch
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from torchvision.io import read_video, write_video
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import traceback

# 导入FLAVR相关模块
from model.FLAVR_arch import UNet_3D_3D
from dataset.transforms import ToTensorVideo, Resize

# 模型配置
MODEL_CONFIGS = {
    "flavr_2x": {
        "path": "models/flavr_2x.pth",
        "factor": 2,
        "description": "2x插值 (30FPS → 60FPS)"
    },
    "flavr_4x": {
        "path": "models/flavr_4x.pth", 
        "factor": 4,
        "description": "4x插值 (30FPS → 120FPS)"
    },
    "flavr_8x": {
        "path": "models/flavr_8x.pth",
        "factor": 8, 
        "description": "8x插值 (30FPS → 240FPS)"
    }
}



class FLAVRInterpolator:
    def __init__(self):
        self.models = {}  # 缓存所有加载的模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_model_name = None
        

        
    def load_model(self, model_name):
        """加载FLAVR模型"""
        try:
            print(f"🔄 正在加载模型: {model_name}")
            if model_name not in MODEL_CONFIGS:
                return False, f"未知模型: {model_name}"
            
            # 检查是否已经加载了相同的模型
            if model_name in self.models:
                self.current_model_name = model_name
                config = MODEL_CONFIGS[model_name]
                return True, f"模型已加载! {config['description']} (设备: {self.device})"
            
            config = MODEL_CONFIGS[model_name]
            model_path = config["path"]
            
            if not os.path.exists(model_path):
                return False, f"模型文件不存在: {model_path}\n请下载模型文件到 {model_path}"
            
            # 模型参数
            model_arch = "unet_18"
            nbr_frame = 4
            joinType = "concat"
            n_outputs = config["factor"] - 1
            
            # 创建模型
            model = UNet_3D_3D(
                model_arch.lower(), 
                n_inputs=nbr_frame, 
                n_outputs=n_outputs, 
                joinType=joinType, 
                upmode="transpose"
            )
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)
            saved_state_dict = checkpoint['state_dict']
            saved_state_dict = {k.partition("module.")[-1]: v for k, v in saved_state_dict.items()}
            model.load_state_dict(saved_state_dict)
            
            model = model.to(self.device)
            model.eval()

            print("模型加载成功")
            
            # 缓存模型
            self.models[model_name] = model
            self.current_model_name = model_name
            
            return True, f"模型加载成功! {config['description']} (设备: {self.device})"
            
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"
    
    def get_model_factor(self):
        """获取当前模型的插值倍数"""
        if self.current_model_name and self.current_model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[self.current_model_name]["factor"]
        return 2  # 默认2x
    
    def get_loaded_models_info(self):
        """获取已加载模型的信息"""
        info = []
        for model_name in self.models.keys():
            if model_name in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_name]
                info.append(f"✅ {config['description']} (已加载)")
        return info
    
    def clear_models(self):
        """清理所有模型缓存"""
        for model_name in list(self.models.keys()):
            if model_name != self.current_model_name:  # 保留当前使用的模型
                del self.models[model_name]
                print(f"🗑️  已释放模型: {model_name}")
        
        torch.cuda.empty_cache()
    
    def switch_model(self, new_model_name):
        """切换模型时释放其他模型"""
        # 确保new_model_name是字符串
        if not isinstance(new_model_name, str):
            return False, f"模型名称必须是字符串，当前类型: {type(new_model_name)}"
        
        if new_model_name == self.current_model_name and new_model_name in self.models:
            return True, "模型未变化"
        
        # 释放除新模型外的所有其他模型
        models_to_remove = []
        for model_name in list(self.models.keys()):  # 使用list()避免在迭代时修改字典
            if model_name != new_model_name:
                models_to_remove.append(model_name)
        
        for model_name in models_to_remove:
            del self.models[model_name]
            print(f"🗑️  切换模型时释放: {model_name}")
        
        # 如果新模型未加载，则加载它
        if new_model_name not in self.models:
            success, msg = self.load_model(new_model_name)
            if not success:
                return False, msg
        
        self.current_model_name = new_model_name
        return True, f"已切换到 {new_model_name}"
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        import psutil
        
        # 获取系统内存使用
        memory = psutil.virtual_memory()
        system_memory = f"系统内存: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent}%)"
        
        if torch.cuda.is_available():
            gpu_memory = f"GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB"
            return f"{system_memory} | {gpu_memory}"
        else:
            return f"{system_memory} | CPU模式"
    
    def print_memory_status(self, stage=""):
        """打印内存状态"""
        import psutil
        
        # 系统内存
        memory = psutil.virtual_memory()
        print(f"{stage} 系统内存: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent}%)")
        
        # GPU内存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"{stage} GPU内存: {allocated:.2f}GB / {reserved:.2f}GB")
        
        # Python进程内存
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024**3
        print(f"{stage} 进程内存: {process_memory:.2f}GB")
    
    def video_to_tensor(self, video_path):
        """将视频转换为张量 - 改进版本"""
        try:
            # 检查文件是否存在
            if not os.path.exists(video_path):
                raise Exception(f"视频文件不存在: {video_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                raise Exception("视频文件为空")
            
            print(f"视频文件大小: {file_size / (1024*1024):.2f} MB")
            
            # 尝试使用torchvision读取
            try:
                video_tensor, _, metadata = read_video(video_path)
                fps = metadata["video_fps"]
                duration = metadata.get("duration", 0)
                
                # 计算视频长度（秒）
                if duration > 0:
                    video_length = duration
                else:
                    video_length = len(video_tensor) / fps if fps > 0 else 0
                
                print(f"视频长度: {video_length:.2f}秒")
                print(f"视频帧数: {len(video_tensor)}")
                print(f"视频FPS: {fps}")
                print(f"视频尺寸: {video_tensor.shape}")
                
                return video_tensor, fps
                
            except Exception as e:
                print(f"torchvision读取失败: {str(e)}")
                # 尝试使用OpenCV读取
                return self._video_to_tensor_opencv(video_path)
                
        except Exception as e:
            raise Exception(f"视频读取失败: {str(e)}")
    
    def _video_to_tensor_opencv(self, video_path):
        """使用OpenCV读取视频"""
        try:
            print("尝试使用OpenCV读取视频...")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"OpenCV读取 - FPS: {fps}, 帧数: {frame_count}, 尺寸: {width}x{height}")
            
            # 读取所有帧
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                raise Exception("没有读取到任何帧")
            
            # 转换为tensor格式 (T, H, W, C)
            video_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
            
            print(f"OpenCV读取成功 - 帧数: {len(frames)}")
            return video_tensor, fps
            
        except Exception as e:
            raise Exception(f"OpenCV读取也失败: {str(e)}")
    
    def _get_video_info(self, video_path):
        """获取视频基本信息"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height
            }
            
        except Exception as e:
            raise Exception(f"获取视频信息失败: {str(e)}")
    
    def _load_video_segment(self, video_path, start_frame, end_frame):
        """流式加载视频段"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            # 设置起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 读取指定范围的帧
            frames = []
            for i in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                raise Exception(f"无法读取帧 {start_frame}-{end_frame}")
            
            # 转换为tensor格式 (T, H, W, C)
            video_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
            
            # 预处理当前段
            video_tensor, (new_h, new_w) = self.video_transform(video_tensor)
            
            print(f"流式加载段 {start_frame}-{end_frame}: {len(frames)}帧, 尺寸: {new_h}x{new_w}")
            
            # 清理frames列表，释放内存
            del frames
            
            return video_tensor, (new_w, new_h)
            
        except Exception as e:
            raise Exception(f"流式加载视频段失败: {str(e)}")
    
    def video_transform(self, video_tensor, target_size=None):
        """视频预处理"""
        T, H, W = video_tensor.size(0), video_tensor.size(1), video_tensor.size(2)
        
        if target_size is None:
            # 自动调整到8的倍数
            new_h = 8 * (H // 8)
            new_w = 8 * (W // 8)
        else:
            new_h, new_w = target_size
        
        print(f"原始尺寸: {H}x{W}, 调整后尺寸: {new_h}x{new_w}")
        
        # 检查内存需求并给出警告
        estimated_memory_gb = T * new_h * new_w * 3 * 4 / (1024**3)  # 转换为GB
        print(f"预估内存需求: {estimated_memory_gb:.2f} GB")
        
        if estimated_memory_gb > 4:  # 超过4GB给出警告
            print(f"⚠️  警告: 预估内存需求 {estimated_memory_gb:.2f} GB 较高，可能导致内存不足")
        
        transform = transforms.Compose([
            ToTensorVideo(),
            Resize((new_h, new_w))
        ])
        
        try:
            video_tensor = transform(video_tensor)
        except RuntimeError as e:
            if "not enough memory" in str(e):
                # 计算建议的分辨率
                available_memory_gb = 2  # 假设可用内存2GB
                scale_factor = (available_memory_gb / estimated_memory_gb) ** 0.5
                suggested_h = int(H * scale_factor)
                suggested_w = int(W * scale_factor)
                suggested_h = 8 * (suggested_h // 8)
                suggested_w = 8 * (suggested_w // 8)
                
                error_msg = f"""
❌ 内存不足错误!

当前视频信息:
- 原始尺寸: {H}x{W}
- 目标尺寸: {new_h}x{new_w}
- 帧数: {T}
- 预估内存需求: {estimated_memory_gb:.2f} GB

💡 解决方案:
1. 使用更小的视频文件
2. 手动设置较小的目标尺寸 (建议: {suggested_h}x{suggested_w})
3. 裁剪视频长度
4. 增加系统内存

请重新上传较小的视频或手动设置目标尺寸。
                """
                raise Exception(error_msg)
            else:
                raise e
        
        return video_tensor, (new_h, new_w)
    
    def make_image(self, img):
        """将张量转换为图像"""
        try:
            q_im = img.data.mul(255.).clamp(0, 255).round()
            im = q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            # 注意：OpenCV读取的是BGR格式，to_tensor保持BGR格式
            # 所以这里不需要颜色空间转换，直接返回BGR格式
            # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # 删除这行，保持BGR格式
            
            # 验证图像数据
            if im is None or im.size == 0:
                print("警告: 生成的图像为空")
                return None
            
            if np.any(np.isnan(im)) or np.any(np.isinf(im)):
                print("警告: 图像包含无效数据")
                return None
            
            return im
            
        except Exception as e:
            print(f"图像转换失败: {str(e)}")
            return None
    
    def validate_frame(self, frame):
        """验证帧数据"""
        if frame is None:
            return False
        
        if not isinstance(frame, np.ndarray):
            return False
        
        if len(frame.shape) != 3:
            return False
        
        if frame.size == 0:
            return False
        
        if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
            return False
        
        # 验证颜色通道
        if frame.shape[2] != 3:
            print(f"警告: 帧颜色通道数错误: {frame.shape[2]}")
            return False
        
        # 验证颜色值范围
        if frame.min() < 0 or frame.max() > 255:
            print(f"警告: 帧颜色值超出范围: [{frame.min()}, {frame.max()}]")
            return False
        
        return True
    
    def interpolate_video(self, video_path, segment_duration=10):
        """视频插值主函数 - 真正的分段存储处理"""
        if not self.current_model_name or self.current_model_name not in self.models:
            raise Exception("请先加载模型!")
        
        factor = self.get_model_factor()
        
        # 获取视频基本信息（不加载全部数据）
        video_info = self._get_video_info(video_path)
        original_fps = video_info['fps']
        total_frames = video_info['frame_count']
        width = video_info['width']
        height = video_info['height']
        
        print(f"视频信息: {width}x{height}, {original_fps}FPS, {total_frames}帧")
        
        # 检查视频帧数是否足够
        if total_frames < 4:
            raise Exception(f"视频帧数不足! 需要至少4帧，当前只有{total_frames}帧")
        
        # 简化分段策略：根据4的倍数和时长分段
        min_frames_per_segment = 4  # FLAVR需要至少4帧
        target_frames_per_segment = int(segment_duration * original_fps)
        
        # 确保每段帧数是4的倍数
        frames_per_segment = 4 * (target_frames_per_segment // 4)  # 向下取整到4的倍数
        if frames_per_segment < min_frames_per_segment:
            frames_per_segment = min_frames_per_segment
        
        print(f"分段计算: 目标时长{segment_duration}秒, 视频{original_fps}FPS, 总帧数{total_frames}")
        print(f"每段帧数: {frames_per_segment}帧 (4的倍数)")
        
        # 计算分段数
        num_segments = (total_frames + frames_per_segment - 1) // frames_per_segment
        
        # 检查最后一段是否需要补帧
        last_segment_frames = total_frames - (num_segments - 1) * frames_per_segment
        if last_segment_frames > 0 and last_segment_frames < min_frames_per_segment:
            print(f"最后一段帧数不足({last_segment_frames}帧 < {min_frames_per_segment}帧)，将补帧到{min_frames_per_segment}帧")
        
        print(f"分段策略: {num_segments}段，每段{frames_per_segment}帧")
        
        # 显示每段的帧数分布
        for i in range(num_segments):
            start_frame = i * frames_per_segment
            end_frame = min(start_frame + frames_per_segment, total_frames)
            segment_frames = end_frame - start_frame
            print(f"  第{i+1}段: 帧{start_frame}-{end_frame-1} (共{segment_frames}帧)")
            
            if segment_frames < min_frames_per_segment and segment_frames > 0:
                print(f"    ⚠️  最后一段帧数不足，将使用复制策略补足")
        
        print(f"分段存储处理: {num_segments}段，每段约{segment_duration}秒 ({frames_per_segment}帧)")
        
        # 显示初始内存状态
        self.print_memory_status("初始")
        
        # 创建临时目录存储分段视频
        temp_dir = tempfile.mkdtemp(prefix="flavr_segments_")
        segment_files = []
        
        try:
            # 分段处理并立即保存
            total_processed_frames = 0
            for segment_idx in tqdm(range(num_segments), desc="分段处理进度"):
                start_frame = segment_idx * frames_per_segment
                end_frame = min(start_frame + frames_per_segment, total_frames)
                segment_frame_count = end_frame - start_frame
                
                print(f"处理第 {segment_idx + 1}/{num_segments} 段: 帧 {start_frame}-{end_frame} (共{segment_frame_count}帧)")
                
                # 流式读取当前段
                segment_tensor, segment_size = self._load_video_segment(video_path, start_frame, end_frame)
                
                # 检查最后一段是否需要补帧
                if segment_idx == num_segments - 1 and segment_frame_count < 4:
                    print(f"第{segment_idx + 1}段帧数不足({segment_frame_count}帧)，补帧到4帧")
                    # 复制最后一帧来补足到4帧
                    last_frame = segment_tensor[:, -1:, :, :]  # 取最后一帧
                    while segment_tensor.size(1) < 4:
                        segment_tensor = torch.cat([segment_tensor, last_frame], dim=1)
                    print(f"补帧后: {segment_tensor.size(1)}帧")
                
                # 处理当前段
                segment_outputs = self._interpolate_segment(segment_tensor, factor, segment_idx, num_segments)
                
                if segment_outputs:
                    # 立即转换为图像并保存当前段
                    segment_images = []
                    for i, im_ in enumerate(segment_outputs):
                        frame = self.make_image(im_)
                        if self.validate_frame(frame):
                            segment_images.append(frame)
                        else:
                            print(f"警告: 第{segment_idx + 1}段第{i}帧无效，跳过")
                    
                    if segment_images:
                        # 保存当前段到临时文件
                        segment_file = os.path.join(temp_dir, f"segment_{segment_idx:03d}.mp4")
                        output_fps = original_fps * factor
                        
                        # 保存当前段视频
                        self._save_segment_video(segment_images, segment_size, output_fps, segment_file)
                        segment_files.append(segment_file)
                        
                        expected_output_frames = segment_frame_count * factor
                        print(f"第{segment_idx + 1}段保存完成: 输入{segment_frame_count}帧 -> 输出{len(segment_images)}帧 (期望{expected_output_frames}帧)")
                        total_processed_frames += len(segment_images)
                    else:
                        print(f"警告: 第{segment_idx + 1}段没有有效帧，跳过")
                else:
                    print(f"警告: 第{segment_idx + 1}段没有输出，跳过")
                
                # 立即清理内存
                del segment_tensor, segment_outputs, segment_images
                
                # 清理frames列表
                if 'frames' in locals():
                    del frames
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # 确保GPU操作完成
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                print(f"第{segment_idx + 1}段内存清理完成")
                
                # 显示当前内存使用情况
                self.print_memory_status(f"第{segment_idx + 1}段后")
            
            print(f"分段处理完成，累计输出帧数: {total_processed_frames}帧")
            
            if not segment_files:
                raise Exception(f"没有生成任何输出段! 请检查视频是否有效，或尝试调整分段时长参数。")
            
            # 合并所有分段视频
            print("合并分段视频...")
            final_video_path = self._merge_segment_videos(segment_files, original_fps * factor)
            
            # 验证最终视频的帧数
            try:
                cap = cv2.VideoCapture(final_video_path)
                final_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                final_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                expected_frames = total_frames * factor
                print(f"帧数验证: 原视频{total_frames}帧 -> 期望{expected_frames}帧 -> 实际{final_frame_count}帧")
                print(f"帧率验证: 原视频{original_fps}FPS -> 期望{original_fps * factor}FPS -> 实际{final_fps}FPS")
                
                # 详细分析每段的帧数
                print("分段帧数分析:")
                segment_total = 0
                for i, segment_file in enumerate(segment_files):
                    if os.path.exists(segment_file):
                        cap = cv2.VideoCapture(segment_file)
                        segment_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        segment_total += segment_frames
                        print(f"  第{i+1}段: {segment_frames}帧")
                    else:
                        print(f"  第{i+1}段: 文件不存在")
                
                print(f"  分段总计: {segment_total}帧")
                
                if final_frame_count != expected_frames:
                    print(f"❌ 错误: 输出帧数({final_frame_count})与期望帧数({expected_frames})不符")
                    print(f"   差异: {final_frame_count - expected_frames}帧")
                    
                    # 如果帧数不对，尝试修复
                    if abs(final_frame_count - expected_frames) <= 5:  # 允许5帧的误差
                        print("⚠️  帧数差异较小，可能是合并过程中的误差")
                    else:
                        print("❌  帧数差异较大，可能存在处理错误")
                else:
                    print(f"✅ 帧数验证通过: {final_frame_count}帧")
                    
            except Exception as e:
                print(f"帧数验证失败: {str(e)}")
            
            print(f"分段存储处理完成: {num_segments}段，每段约{segment_duration}秒")
            
            return final_video_path, segment_size, original_fps * factor
            
        finally:
            # 清理临时文件
            for segment_file in segment_files:
                try:
                    if os.path.exists(segment_file):
                        os.remove(segment_file)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
    
    def _interpolate_segment(self, segment_tensor, factor, segment_idx=0, total_segments=1):
        """处理单个视频段 - 按照官方demo逻辑"""
        nbr_frame = 4
        n_outputs = factor - 1
        
        # 获取时间维度
        time_dim = segment_tensor.size(1)
        
        # 将视频张量按时间维度分割成帧
        frames = [segment_tensor[:, i, :, :] for i in range(time_dim)]
        
        # 目标输出帧数：输入帧数 * factor
        target_output_frames = time_dim * factor
        print(f"第{segment_idx + 1}段目标: 输入{time_dim}帧 -> 输出{target_output_frames}帧")
        
        # 处理不足4帧的情况
        if time_dim < nbr_frame:
            print(f"第{segment_idx + 1}段帧数不足({time_dim}帧)，使用复制策略")
            # 简单复制策略
            outputs = []
            for frame in frames:
                outputs.extend([frame] * factor)  # 每帧复制factor次
            print(f"第{segment_idx + 1}段复制处理完成: 输入{time_dim}帧 -> 输出{len(outputs)}帧")
            return outputs
        
        # 正常处理（4帧或以上）- 按照官方逻辑
        idxs = torch.arange(time_dim).view(1, -1).unfold(1, size=nbr_frame, step=1).squeeze(0)
        
        print(f"第{segment_idx + 1}段调试: {time_dim}帧, {nbr_frame}帧窗口, {len(idxs)}个索引组")
        
        outputs = []  # 存储输入和插值帧
        current_model = self.models[self.current_model_name]
        
        # 按照官方逻辑：添加第一帧（第2帧）
        outputs.append(frames[idxs[0][1]])
        
        # 处理每个4帧窗口
        for i in range(len(idxs)):
            idx_set = idxs[i]
            inputs = [frames[idx_].to(self.device).unsqueeze(0) for idx_ in idx_set]
            
            with torch.no_grad():
                output_frames = current_model(inputs)
            
            output_frames = [of.squeeze(0).cpu().data for of in output_frames]
            outputs.extend(output_frames)  # 添加插值帧
            outputs.append(inputs[2].squeeze(0).cpu().data)  # 添加第3帧
            
            # 清理GPU内存
            del inputs, output_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"第{segment_idx + 1}段处理完成: 输入{time_dim}帧 -> 输出{len(outputs)}帧")
        return outputs
    
    def save_video(self, frames, size, fps, output_path):
        """保存视频 - 改进版本"""
        try:
            # 尝试使用H.264编码，更兼容
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_path, fourcc, fps, size)
            
            if not out.isOpened():
                # 如果H.264不可用，尝试mp4v
                print("H.264编码不可用，尝试mp4v编码")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, size)
                
                if not out.isOpened():
                    # 如果mp4v也不可用，尝试XVID
                    print("mp4v编码不可用，尝试XVID编码")
                    output_path = output_path.replace('.mp4', '.avi')
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(output_path, fourcc, fps, size)
            
            if not out.isOpened():
                raise Exception("无法创建视频写入器，请检查编码器支持")
            
            print(f"使用编码器: {chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)}")
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            return output_path
            
        except Exception as e:
            print(f"视频保存失败: {str(e)}")
            # 尝试使用PIL和ffmpeg保存
            return self._save_video_alternative(frames, size, fps, output_path)
    
    def _save_video_alternative(self, frames, size, fps, output_path):
        """备用视频保存方法"""
        try:
            print("使用备用方法保存视频...")
            
            # 使用PIL保存帧为临时文件
            temp_dir = tempfile.mkdtemp()
            frame_files = []
            
            for i, frame in enumerate(frames):
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                pil_image.save(frame_path)
                frame_files.append(frame_path)
            
            # 使用ffmpeg合成视频
            import subprocess
            
            # 构建ffmpeg命令
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # 覆盖输出文件
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                output_path
            ]
            
            print(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"ffmpeg错误: {result.stderr}")
                raise Exception(f"ffmpeg执行失败: {result.stderr}")
            
            # 清理临时文件
            for frame_file in frame_files:
                try:
                    os.remove(frame_file)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
            
            return output_path
            
        except Exception as e:
            print(f"备用保存方法也失败: {str(e)}")
            raise Exception(f"视频保存失败: {str(e)}")
    
    def _save_segment_video(self, frames, size, fps, output_path):
        """保存分段视频"""
        try:
            # 直接使用ffmpeg保存，避免OpenCV的编码问题
            return self._save_video_ffmpeg(frames, size, fps, output_path)
            
        except Exception as e:
            print(f"分段视频保存失败: {str(e)}")
            # 备用方案：尝试OpenCV
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, size)
                
                if out.isOpened():
                    for frame in frames:
                        if frame is not None:
                            out.write(frame)
                    out.release()
                    
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        return output_path
            except Exception as e2:
                print(f"OpenCV备用保存也失败: {str(e2)}")
            
            raise e
    
    def _save_video_ffmpeg(self, frames, size, fps, output_path):
        """使用ffmpeg保存视频"""
        try:
            # 创建临时目录保存帧
            temp_dir = tempfile.mkdtemp()
            frame_files = []
            
            for i, frame in enumerate(frames):
                if frame is None:
                    print(f"跳过空帧 {i}")
                    continue
                
                try:
                    # 验证帧数据
                    if not isinstance(frame, np.ndarray):
                        print(f"帧{i}不是numpy数组，跳过")
                        continue
                    
                    if len(frame.shape) != 3:
                        print(f"帧{i}维度错误: {frame.shape}，跳过")
                        continue
                    
                    # 确保帧尺寸正确
                    if frame.shape[:2] != (size[1], size[0]):
                        print(f"调整帧{i}尺寸: {frame.shape[:2]} -> {size[1]}x{size[0]}")
                        frame = cv2.resize(frame, size)
                    
                    # 确保数据类型正确
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    # 注意：frame已经是BGR格式，需要转换为RGB给PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    pil_image.save(frame_path)
                    frame_files.append(frame_path)
                    
                except Exception as e:
                    print(f"保存帧{i}失败: {str(e)}")
                    continue
            
            if not frame_files:
                raise Exception("没有成功保存任何帧")
            
            # 使用ffmpeg合成视频
            import subprocess
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'fast',
                '-movflags', '+faststart',  # 优化网络播放
                '-profile:v', 'baseline',   # 使用基础配置，提高兼容性
                '-level', '3.0',            # 设置编码级别
                output_path
            ]
            
            print(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"ffmpeg错误输出: {result.stderr}")
                print(f"ffmpeg标准输出: {result.stdout}")
                raise Exception(f"ffmpeg执行失败 (返回码: {result.returncode}): {result.stderr}")
            
            print(f"ffmpeg执行成功，输出文件: {output_path}")
            
            # 清理临时文件
            for frame_file in frame_files:
                try:
                    os.remove(frame_file)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
            
            return output_path
            
        except Exception as e:
            print(f"ffmpeg保存失败: {str(e)}")
            # 尝试使用torchvision作为最后的备用方案
            try:
                print("尝试使用torchvision保存...")
                return self._save_video_torchvision(frames, size, fps, output_path)
            except Exception as e2:
                print(f"torchvision保存也失败: {str(e2)}")
                raise e
    
    def _merge_segment_videos(self, segment_files, fps):
        """合并分段视频"""
        try:
            if len(segment_files) == 1:
                return segment_files[0]
            
            # 创建合并文件列表
            merge_list_path = os.path.join(os.path.dirname(segment_files[0]), 'merge_list.txt')
            with open(merge_list_path, 'w') as f:
                for segment_file in segment_files:
                    f.write(f"file '{segment_file}'\n")
            
            # 使用ffmpeg合并
            import subprocess
            output_path = os.path.join(os.path.dirname(segment_files[0]), 'merged_output.mp4')
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', merge_list_path,
                '-c', 'copy',
                output_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            # 清理合并列表文件
            try:
                os.remove(merge_list_path)
            except:
                pass
            
            if result.returncode != 0:
                raise Exception(f"视频合并失败: {result.stderr}")
            
            return output_path
            
        except Exception as e:
            print(f"视频合并失败: {str(e)}")
            # 如果合并失败，返回第一个分段文件
            return segment_files[0] if segment_files else None

# 创建全局插值器实例
interpolator = FLAVRInterpolator()

def check_model_files():
    """检查模型文件是否存在"""
    status = {}
    for model_name, config in MODEL_CONFIGS.items():
        exists = os.path.exists(config["path"])
        status[model_name] = {
            "exists": exists,
            "path": config["path"],
            "description": config["description"]
        }
    return status

def ensure_compatible_format(video_path):
    """确保视频格式兼容"""
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(video_path)[1].lower()
        
        # 如果已经是MP4格式，直接返回
        if file_ext == '.mp4':
            return video_path
        
        # 尝试使用OpenCV读取视频
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            cap.release()
            return video_path  # 可以读取，不需要转换
        
        # 如果无法读取，尝试转换为MP4
        print(f"视频格式 {file_ext} 可能不兼容，尝试转换为MP4...")
        
        # 创建临时MP4文件
        temp_mp4 = video_path.replace(file_ext, '_converted.mp4')
        
        try:
            import subprocess
            
            # 使用ffmpeg转换
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                temp_mp4
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(temp_mp4):
                print(f"视频转换成功: {temp_mp4}")
                return temp_mp4
            else:
                print(f"视频转换失败: {result.stderr}")
                return video_path  # 转换失败，返回原文件
                
        except Exception as e:
            print(f"转换过程出错: {str(e)}")
            return video_path  # 转换失败，返回原文件
            
    except Exception as e:
        print(f"格式检查失败: {str(e)}")
        return video_path
        
def process_video(video_file, model_name, segment_duration):
    """Gradio处理函数"""
    if video_file is None:
        return None, "请上传视频文件"
    
    if model_name is None:
        return None, "请选择模型"
    
    try:
        # 切换模型（会自动释放其他模型）
        success, msg = interpolator.switch_model(model_name)
        if not success:
            return None, msg
        
        # 获取视频文件路径
        if isinstance(video_file, str):
            video_path = video_file
        elif hasattr(video_file, 'name'):
            video_path = video_file.name
        elif hasattr(video_file, 'file_path'):
            video_path = video_file.file_path
        else:
            return None, "无法获取视频文件路径"
        
        # 检查文件是否存在
        if not os.path.exists(video_path):
            return None, f"视频文件不存在: {video_path}"
        
        # 检查文件大小
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return None, "视频文件为空"
        
        print(f"处理视频文件: {video_path}, 大小: {file_size / (1024*1024):.2f} MB")
        
        # 检查视频格式，如果格式不兼容则转换
        video_path = ensure_compatible_format(video_path)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_output = tmp_file.name
        
        print(f"开始处理视频，分段时长: {segment_duration}秒")
        
        # 显示处理信息
        status_msg = f"🔄 开始处理视频...\n📊 分段时长: {segment_duration}秒\n"
        
        # 处理视频
        output_path, size, fps = interpolator.interpolate_video(video_path, segment_duration)
        print("视频处理完成")
        
        # 复制到最终输出位置
        import shutil
        shutil.copy2(output_path, temp_output)

        print(f"视频处理完成，输出路径: {temp_output}")
        
        factor = interpolator.get_model_factor()
        memory_usage = interpolator.get_memory_usage()
        return output_path, f"处理完成! {factor}x插值, 分段时长: {segment_duration}秒, 输出FPS: {fps}, 尺寸: {size[0]}x{size[1]} | {memory_usage}"
        
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        
        # 视频格式相关错误
        if "视频帧数不足" in error_msg or "视频太短" in error_msg:
            return None, f"❌ {error_msg}\n💡 请上传更长的视频文件（至少需要4帧）"
        elif "PyAV is not installed" in error_msg:
            return None, f"❌ {error_msg}\n💡 请安装PyAV: pip install av"
        elif "Video not playable" in error_msg or "无法打开视频文件" in error_msg:
            return None, f"❌ 视频格式不支持\n💡 请尝试以下解决方案:\n1. 系统会自动转换格式，请稍等\n2. 如果仍有问题，请手动转换为MP4格式\n3. 使用H.264编码\n4. 检查视频文件是否损坏\n5. 尝试其他视频文件"
        elif "视频读取失败" in error_msg:
            return None, f"❌ {error_msg}\n💡 请检查:\n1. 视频文件是否完整\n2. 格式是否支持(MP4, AVI, MOV等)\n3. 文件是否损坏"
        elif "视频保存失败" in error_msg:
            return None, f"❌ {error_msg}\n💡 请检查:\n1. 磁盘空间是否充足\n2. 是否有写入权限\n3. 尝试重新处理"
        else:
            return None, f"❌ 处理失败: {error_msg}"

def clear_model_cache():
    """清理模型缓存"""
    interpolator.clear_models()
    return interpolator.get_memory_usage(), "模型缓存已清理"

def update_model_status():
    """更新模型状态显示"""
    model_status = check_model_files()
    model_choices = []
    model_descriptions = []
    
    for model_name, config in MODEL_CONFIGS.items():
        status = model_status[model_name]
        if status["exists"]:
            model_choices.append(model_name)
            model_descriptions.append(f"✅ {config['description']}")
        else:
            model_choices.append(model_name)
            model_descriptions.append(f"❌ {config['description']} (未下载)")
    
    loaded_models_info = interpolator.get_loaded_models_info()
    status_text = "### 模型状态:\n" + "\n".join([f"- {desc}" for desc in model_descriptions]) + \
                  "\n\n### 已加载模型:\n" + "\n".join([f"- {info}" for info in loaded_models_info])
    
    return model_choices, status_text, interpolator.get_memory_usage()

def check_video_format(video_file):
    """检查视频格式"""
    if video_file is None:
        return "请上传视频文件"
    
    try:
        # 获取视频文件路径
        if isinstance(video_file, str):
            video_path = video_file
        elif hasattr(video_file, 'name'):
            video_path = video_file.name
        elif hasattr(video_file, 'file_path'):
            video_path = video_file.file_path
        else:
            return "无法获取视频文件路径"
        
        # 检查文件扩展名
        file_ext = os.path.splitext(video_path)[1].lower()
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        if file_ext not in supported_formats:
            return f"⚠️ 不支持的格式: {file_ext}\n💡 建议转换为MP4格式"
        
        # 检查文件大小
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return "❌ 视频文件为空"
        
        # 尝试读取视频信息
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return "❌ 无法打开视频文件，可能格式不支持或文件损坏"
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            if frame_count < 4:
                return f"❌ 视频帧数不足: {frame_count}帧 (需要至少4帧)"
            
            return f"✅ 视频格式正常\n📊 信息: {width}x{height}, {fps:.1f}FPS, {frame_count}帧, {file_size/(1024*1024):.1f}MB"
            
        except Exception as e:
            return f"⚠️ 视频信息读取失败: {str(e)}"
            
    except Exception as e:
        return f"❌ 视频检查失败: {str(e)}"

def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="FLAVR 视频补帧工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 FLAVR 视频补帧工具
        
        FLAVR是一种快速、无流的帧插值方法，能够进行单次多帧预测。上传您的视频，选择插值模型，即可生成高帧率视频。
        
        ### 使用说明:
        1. 上传视频文件 (支持mp4, avi等格式，至少需要4帧)
        2. 选择预训练模型 (插值倍数自动确定)
        3. 点击开始处理
        
        ### 模型说明:
        - **2x插值**: 适用于30FPS→60FPS
        - **4x插值**: 适用于30FPS→120FPS  
        - **8x插值**: 适用于30FPS→240FPS
        
        ### 🚀 新功能:
        - **视频预览**: 支持视频上传前预览
        - **智能内存管理**: 按需加载模型，切换时自动释放内存
        - **实时状态监控**: 显示内存使用和模型状态
        - **分段存储处理**: 真正的分段处理，每段处理完成后立即保存到本地，彻底解决内存问题
        
        ### ⚠️ 内存使用提示:
        - 处理大分辨率视频时可能需要大量内存
        - 系统会自动显示预估内存需求
        - 如遇内存不足，请使用较小的视频文件或降低分辨率
        - **分段处理**: 可通过调整分段时长来控制内存使用
        
        ### 📹 支持的视频格式:
        - **推荐格式**: MP4 (H.264编码)
        - **其他格式**: AVI, MOV, MKV, WMV
        - **自动转换**: 系统会自动转换不兼容的格式为MP4
        - **注意事项**: 确保视频文件完整且未损坏
        """)
        
        # 检查模型文件状态
        model_status = check_model_files()
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入部分
                video_input = gr.Video(
                    label="上传视频文件"
                )
                
                # 视频格式检查
                video_check_btn = gr.Button(
                    "检查视频格式",
                    variant="secondary",
                    size="sm"
                )
                
                video_status = gr.Textbox(
                    label="视频状态",
                    interactive=False,
                    lines=3
                )
                
                # 模型选择
                model_choices = []
                model_descriptions = []
                for model_name, config in MODEL_CONFIGS.items():
                    status = model_status[model_name]
                    if status["exists"]:
                        model_choices.append(model_name)
                        model_descriptions.append(f"✅ {config['description']}")
                    else:
                        model_choices.append(model_name)
                        model_descriptions.append(f"❌ {config['description']} (未下载)")
                
                model_input = gr.Dropdown(
                    choices=model_choices,
                    label="选择模型",
                    info="插值倍数由模型自动确定",
                    value=model_choices[0] if model_choices else None
                )
                
                # 分段时长控制
                segment_duration = gr.Slider(
                    minimum=1,
                    maximum=60,
                    value=5,
                    step=1,
                    label="分段时长 (秒)",
                    info="控制分段处理的段长度，每段处理完成后立即保存，推荐5-10秒"
                )
                
                # 显示模型状态
                model_status_text = gr.Markdown(
                    value="### 模型状态:\n" + "\n".join([f"- {desc}" for desc in model_descriptions])
                )
                
                process_btn = gr.Button(
                    "开始处理",
                    variant="primary",
                    size="lg"
                )
                
                # 内存管理
                memory_info = gr.Textbox(
                    label="内存使用情况",
                    value=interpolator.get_memory_usage(),
                    interactive=False
                )
                
                clear_btn = gr.Button(
                    "清理模型缓存",
                    variant="secondary",
                    size="sm"
                )
                
                refresh_btn = gr.Button(
                    "刷新状态",
                    variant="secondary",
                    size="sm"
                )
            
            with gr.Column(scale=1):
                # 输出部分
                video_output = gr.Video(
                    label="处理结果"
                )
                
                status_output = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                    lines=3
                )
        
        # 处理逻辑
        process_btn.click(
            fn=process_video,
            inputs=[video_input, model_input, segment_duration],
            outputs=[video_output, status_output]
        )
        
        # 清理缓存
        clear_btn.click(
            fn=clear_model_cache,
            inputs=[],
            outputs=[memory_info, status_output]
        )
        
        # 刷新状态
        refresh_btn.click(
            fn=update_model_status,
            inputs=[],
            outputs=[model_input, model_status_text, memory_info]
        )
        
        # 视频格式检查
        video_check_btn.click(
            fn=check_video_format,
            inputs=[video_input],
            outputs=[video_status]
        )
        
        # 添加模型下载说明
        gr.Markdown("""
        ### 📥 模型下载说明:
        
        请将下载的模型文件重命名并放置到以下路径:
        
        ```
        models/
        ├── flavr_2x.pth  # 2x插值模型
        ├── flavr_4x.pth  # 4x插值模型
        └── flavr_8x.pth  # 8x插值模型
        ```
        
        ### 🔗 模型下载链接:
        - [2x插值模型](https://drive.google.com/file/d/1IZe-39ZuXy3OheGJC-fT3shZocGYuNdH/view?usp=sharing) → 重命名为 `flavr_2x.pth`
        - [4x插值模型](https://drive.google.com/file/d/1GARJK0Ti1gLH_O0spxAEqzbMwUKqE37S/view?usp=sharing) → 重命名为 `flavr_4x.pth`
        - [8x插值模型](https://drive.google.com/file/d/1xoZqWJdIOjSaE2DtH4ifXKlRwFySm5Gq/view?usp=sharing) → 重命名为 `flavr_8x.pth`
        
        ### ⚡ 性能优化:
        - **智能切换**: 切换模型时自动释放其他模型，节省内存
        - **内存管理**: 可手动清理不需要的模型缓存
        - **GPU加速**: 支持CUDA加速，大幅提升处理速度
        - **实时状态**: 显示当前内存使用和模型加载状态
        
        ### 注意事项:
        - 处理时间取决于视频长度和分辨率
        - 建议使用GPU加速处理
        - 输出视频将自动调整到8的倍数分辨率
        - 支持常见视频格式: MP4, AVI, MOV等
        - 切换模型时会自动释放其他模型内存，提高系统性能
        - **视频要求**: 至少需要4帧才能进行插值处理
        - **帧率说明**: 输出帧率 = 原始帧率 × 插值倍数 (如30FPS输入，2x插值输出60FPS)
        - **内存要求**: 大分辨率视频处理需要充足内存，系统会显示预估内存需求
        - **内存不足时**: 请使用较小的视频文件或降低视频分辨率
        - **分段存储处理**: 长视频会分段处理并立即保存到本地，彻底解决内存问题
        - **分段时长建议**: 内存较小建议3-5秒，内存充足可设置8-15秒
        - **内存优化**: 每段处理完成后立即保存并清理内存，支持处理较长的视频
        """)
    
    return demo

if __name__ == "__main__":
    # 创建并启动界面
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    ) 