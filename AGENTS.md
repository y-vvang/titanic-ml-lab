# AGENTS.md - Titanic ML Lab

## 项目概览
泰坦尼克号生还预测机器学习科普网站，用户通过交互式操作体验完整的ML流程：数据探索→数据清洗→模型训练→模型评估→自定义预测。支持模型版本管理（存储/切换/重命名/删除/导入/导出）。后端使用 Python sklearn 实时训练模型，前端 Next.js 多页面架构。

## 版本技术栈
- **Framework**: Next.js 16 (App Router)
- **Core**: React 19
- **Language**: TypeScript 5
- **UI**: shadcn/ui + Tailwind CSS 4 (暗色科技毛玻璃主题)
- **ML Backend**: Python 3 + scikit-learn + pandas + numpy

## 目录结构
```
├── public/dataset.csv          # 泰坦尼克号原始数据
├── scripts/ml_engine.py        # Python ML 引擎（数据探索/清洗/训练/预测）
├── src/
│   ├── app/
│   │   ├── page.tsx            # 首页
│   │   ├── layout.tsx          # 根布局（metadata）
│   │   ├── globals.css         # 全局样式 + @theme 设计令牌
│   │   ├── explore/page.tsx    # 数据探索
│   │   ├── clean/page.tsx      # 数据清洗
│   │   ├── train/page.tsx      # 模型训练
│   │   ├── evaluate/page.tsx   # 模型评估（含版本选择器）
│   │   ├── predict/page.tsx    # 自定义预测（含版本选择器）
│   │   ├── versions/page.tsx   # 版本管理（重命名/删除/导出/导入）
│   │   └── api/
│   │       ├── data/info/route.ts      # GET 数据概览
│   │       ├── data/explore/route.ts   # GET 数据探索
│   │       ├── data/clean/route.ts     # POST 数据清洗
│   │       ├── data/preview/route.ts   # GET 数据预览
│   │       ├── model/train/route.ts    # POST SSE 模型训练
│   │       └── model/predict/route.ts  # POST 模型预测
│   ├── components/
│   │   ├── ClientLayout.tsx    # 客户端布局（Provider + Nav）
│   │   └── StepNav.tsx         # 顶部导航（含版本管理入口）
│   └── lib/
│       ├── MLContext.tsx       # 全局ML状态 Context（含版本管理）
│       └── utils.ts            # 工具函数
```

## 构建与测试命令
- 开发：`pnpm dev`（端口5000，HMR）
- 构建：`pnpm run build`
- 静态检查：`pnpm lint`、`pnpm ts-check`
- Python 测试：`python3 scripts/ml_engine.py info '{}'`

## 代码风格指南
- 暗色主题：主色 #7C5CFF，强调色 #69E7FF，成功色 #62FAD3
- 毛玻璃卡片：`.glass-card` class
- 渐变按钮：`.btn-gradient` class
- 所有页面组件使用 `'use client'`
- API 路由通过 child_process 调用 Python 引擎

## 数据流
1. 用户操作前端 → 调用 Next.js API Route
2. API Route 通过 `spawn` 执行 `python3 scripts/ml_engine.py <command> <json>`
3. Python 引擎返回 JSON 结果（训练为 SSE 流式）
4. 前端更新 MLContext 全局状态

## 版本管理数据结构
```typescript
interface ModelVersion {
  versionId: number;       // 自增ID
  name: string;            // 可编辑名称，默认 "模型 v1"
  createdAt: string;       // ISO 时间戳
  trainConfig: TrainConfig; // 训练配置（含 cleanConfig/modelType/hyperparams/testSize）
  trainResult: TrainResult; // 训练结果（含 metrics/confusionMatrix/featureImportance/groupAccuracy/modelInfo/hasUnfilledMissing）
}
```
- **存储**：localStorage key `titanic-ml-versions`，`activeVersionId` 同步存储
- **版本选择器**：Evaluate/Predict 页面顶部下拉切换 `activeVersionId`
- **逻辑回归约束**：Predict 页当逻辑回归模型有未填充缺失值时，禁用 CabinDeck/Title 输入

## 关键约束
- ML 训练为真实 sklearn 计算，非预制结果
- Predict 通过重训实现（Python predict 命令接收 cleanConfig 重新训练后预测）
- 数据集路径：public/dataset.csv（Python 使用相对路径）
- 导出格式为 JSON（含 `format: "titanic-ml-model"` 标识 + ModelVersion 数据）
