# 🎬 实时FLAVR使用指南（更新版）

## 参数说明

### 基础参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--factor` | 插值倍数 (2,4,8) | 2 |
| `--camera` | 摄像头索引 | 0 |
| `--width` | 手动设置摄像头宽度 | 自动检测 |
| `--height` | 手动设置摄像头高度 | 自动检测 |
| `--device` | 计算设备 (cuda/cpu) | cuda |
| `--buffer` | 缓冲区大小 | 15 |
| `--scale` | 缩放因子 | 1.0 |
| `--source` | 视频源类型 | camera |
| `--source_path` | 视频文件路径 | - |
| `--save_output` | 保存输出路径 | - |

### 缩放因子说明
- `1.0` - 原始尺寸（默认）
- `0.5` - 一半尺寸（性能优化）
- `0.25` - 四分之一尺寸（最高性能）
- `2.0` - 两倍尺寸（高质量）

## 使用示例

### 1. 基础使用（保持原始尺寸）
```bash
# 简化版本
python realtime_simple.py --factor 2

# 高级版本
python realtime_interpolator_advanced.py --factor 2
```

### 2. 性能优化
```bash
# 一半尺寸处理和显示
python realtime_simple.py --factor 2 --scale 0.5

# 四分之一尺寸处理和显示
python realtime_simple.py --factor 2 --scale 0.25
```

### 3. 高质量模式
```bash
# 两倍尺寸处理和显示
python realtime_interpolator_advanced.py --factor 2 --scale 2.0
```

### 4. 不同插值倍数
```bash
# 4x插值
python realtime_simple.py --model models/flavr_4x.pth --factor 4

# 8x插值
python realtime_simple.py --model models/flavr_8x.pth --factor 8
```

### 5. CPU模式
```bash
# 使用CPU处理
python realtime_simple.py --factor 2 --device cpu
```

### 6. 手动设置分辨率
```bash
# 手动设置1080p
python realtime_simple.py --factor 2 --width 1920 --height 1080

# 手动设置720p
python realtime_simple.py --factor 2 --width 1280 --height 720
```

### 7. 保存输出
```bash
# 保存处理结果
python realtime_interpolator_advanced.py --factor 2 --save_output output.mp4
```

## 重要说明

### 尺寸处理
- **处理尺寸**: 根据缩放因子调整（内部处理）
- **显示尺寸**: 根据缩放因子调整（原始和补帧画面都缩放）
- **智能调整**: 自动调整到8的倍数以满足模型要求

### 性能建议
- **低端设备**: `--scale 0.25` + `--device cpu` (处理和显示都缩小)
- **中等设备**: `--scale 0.5` (处理和显示都缩小)
- **高端设备**: `--scale 1.0` (处理和显示都是原始尺寸)
- **质量优先**: `--scale 2.0` (处理和显示都放大)

### 摄像头设置
- **自动检测**: 系统会自动尝试设置最高分辨率（1080p → 720p → SVGA → VGA）
- **手动设置**: 可以使用 `--width` 和 `--height` 参数手动指定分辨率
- **智能匹配**: 如果指定分辨率不支持，会自动降级到最接近的支持分辨率

## 故障排除

### 常见问题
1. **内存不足**: 降低缩放因子 `--scale 0.25`
2. **性能不足**: 使用CPU模式 `--device cpu`
3. **摄像头无法打开**: 尝试不同索引 `--camera 1`
4. **模型文件不存在**: 检查模型路径

### 调试信息
程序会显示以下信息：
- 原始尺寸记录
- 尺寸调整过程
- 处理性能统计

## 快速开始

```bash
# 一键启动（推荐新手）
python start_realtime.py --factor 2

# 性能优化启动
python start_realtime.py --factor 2 --scale 0.5
```

现在系统会自动处理所有尺寸相关的问题，您只需要关注插值倍数和缩放因子即可！🎯 