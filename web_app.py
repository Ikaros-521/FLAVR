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
        if new_model_name == self.current_model_name and new_model_name in self.models:
            return True, "模型未变化"
        
        # 释放除新模型外的所有其他模型
        models_to_remove = []
        for model_name in self.models.keys():
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
        if torch.cuda.is_available():
            return f"GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB"
        else:
            return "CPU模式"
    
    def video_to_tensor(self, video_path):
        """将视频转换为张量"""
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
            
            return video_tensor, fps
        except Exception as e:
            raise Exception(f"视频读取失败: {str(e)}")
    
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
        q_im = img.data.mul(255.).clamp(0, 255).round()
        im = q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        return im
    
    def interpolate_video(self, video_path):
        """视频插值主函数"""
        if not self.current_model_name or self.current_model_name not in self.models:
            raise Exception("请先加载模型!")
        
        factor = self.get_model_factor()
        
        # 读取视频
        video_tensor, original_fps = self.video_to_tensor(video_path)
        print(f"原始视频FPS: {original_fps}")
        print(f"视频帧数: {len(video_tensor)}")
        
        # 检查视频帧数是否足够
        if len(video_tensor) < 4:
            raise Exception(f"视频帧数不足! 需要至少4帧，当前只有{len(video_tensor)}帧")
        
        # 预处理
        video_tensor, (new_h, new_w) = self.video_transform(video_tensor)
        print(f"调整后尺寸: {new_h}x{new_w}")
        print(f"视频张量形状: {video_tensor.shape}")
        
        # 检查张量维度
        if len(video_tensor.shape) != 4:
            raise Exception(f"视频张量维度错误! 期望4维，实际{len(video_tensor.shape)}维")
        
        # 准备帧索引
        nbr_frame = 4
        n_outputs = factor - 1
        
        # 获取时间维度（经过ToTensorVideo后，时间维度在第1维）
        time_dim = video_tensor.size(1)
        
        # 确保有足够的帧进行插值
        if time_dim < nbr_frame:
            raise Exception(f"视频太短，无法进行插值! 需要至少{nbr_frame}帧")
        
        # 计算可以生成的插值片段数量
        num_segments = time_dim - nbr_frame + 1
        print(f"可生成插值片段: {num_segments}")
        
        if num_segments <= 0:
            raise Exception(f"视频太短，无法进行插值! 需要至少{nbr_frame}帧")
        
        idxs = torch.arange(time_dim).view(1, -1).unfold(1, size=nbr_frame, step=1).squeeze(0)
        
        # 将视频张量按时间维度分割成帧
        frames = [video_tensor[:, i, :, :] for i in range(time_dim)]
        outputs = []
        
        # 添加第一帧
        if len(idxs) > 0:
            outputs.append(frames[idxs[0][1]])
        else:
            raise Exception("无法生成插值索引，视频可能太短")
        
        # 逐帧插值
        current_model = self.models[self.current_model_name]
        for i in tqdm(range(len(idxs)), desc="插值进度"):
            idx_set = idxs[i]
            inputs = [frames[idx_].to(self.device).unsqueeze(0) for idx_ in idx_set]
            
            with torch.no_grad():
                output_frames = current_model(inputs)
            
            output_frames = [of.squeeze(0).cpu().data for of in output_frames]
            outputs.extend(output_frames)
            outputs.append(frames[idx_set[2]].cpu().data)
        
        # 转换为图像列表
        new_video = [self.make_image(im_) for im_ in outputs]
        
        # 计算输出帧率 = 原始帧率 * 插值倍数
        output_fps = original_fps * factor
        print(f"插值后输出帧率: {output_fps} FPS")
        
        return new_video, (new_w, new_h), output_fps
    
    def save_video(self, frames, size, fps, output_path):
        """保存视频"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return output_path

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

def process_video(video_file, model_name):
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
        video_path = video_file if isinstance(video_file, str) else video_file.name
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_output = tmp_file.name
        
        print("开始处理视频")
        # 处理视频
        frames, size, fps = interpolator.interpolate_video(video_path)
        print("视频处理完成")
        
        # 保存视频
        output_path = interpolator.save_video(frames, size, fps, temp_output)
        
        factor = interpolator.get_model_factor()
        memory_usage = interpolator.get_memory_usage()
        return output_path, f"处理完成! {factor}x插值, 输出FPS: {fps}, 尺寸: {size[0]}x{size[1]} | {memory_usage}"
        
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        if "视频帧数不足" in error_msg or "视频太短" in error_msg:
            return None, f"❌ {error_msg}\n💡 请上传更长的视频文件（至少需要4帧）"
        elif "PyAV is not installed" in error_msg:
            return None, f"❌ {error_msg}\n💡 请安装PyAV: pip install av"
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
        
        ### ⚠️ 内存使用提示:
        - 处理大分辨率视频时可能需要大量内存
        - 系统会自动显示预估内存需求
        - 如遇内存不足，请使用较小的视频文件或降低分辨率
        """)
        
        # 检查模型文件状态
        model_status = check_model_files()
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入部分
                video_input = gr.Video(
                    label="上传视频文件",
                    format="mp4"
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
                    label="处理结果",
                    format="mp4"
                )
                
                status_output = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                    lines=3
                )
        
        # 处理逻辑
        process_btn.click(
            fn=process_video,
            inputs=[video_input, model_input],
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