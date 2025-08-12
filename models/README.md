# 模型文件目录

请将下载的FLAVR预训练模型文件放置在此目录中，并按照以下命名规则重命名：

## 📁 文件结构

```
models/
├── flavr_2x.pth  # 2x插值模型
├── flavr_4x.pth  # 4x插值模型
└── flavr_8x.pth  # 8x插值模型
```

## 🔗 下载链接

| 模型 | 原始文件名 | 重命名后 | 下载链接 |
|------|------------|----------|----------|
| **2x插值** | 原始文件名 | `flavr_2x.pth` | [下载](https://drive.google.com/file/d/1IZe-39ZuXy3OheGJC-fT3shZocGYuNdH/view?usp=sharing) |
| **4x插值** | 原始文件名 | `flavr_4x.pth` | [下载](https://drive.google.com/file/d/1GARJK0Ti1gLH_O0spxAEqzbMwUKqE37S/view?usp=sharing) |
| **8x插值** | 原始文件名 | `flavr_8x.pth` | [下载](https://drive.google.com/file/d/1xoZqWJdIOjSaE2DtH4ifXKlRwFySm5Gq/view?usp=sharing) |

## 📋 使用说明

1. 从上述链接下载模型文件
2. 将文件重命名为对应的名称（如 `flavr_2x.pth`）
3. 放置到此 `models/` 目录中
4. 启动Web应用，系统会自动检测可用的模型

## ⚠️ 注意事项

- 确保文件名完全匹配（区分大小写）
- 模型文件较大，下载可能需要一些时间
- 每个模型对应特定的插值倍数，不能混用 