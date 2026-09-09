# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

泰坦尼克号生还预测机器学习科普网站：五步交互式 ML 流程（数据探索 → 数据清洗 → 模型训练 → 模型评估 → 自定义预测），外加模型版本管理。前端 Next.js 16 (App Router) + React 19 + shadcn/ui + Tailwind CSS 4（暗色毛玻璃主题，设计规范见 DESIGN.md）；ML 计算是**真实的**：API 路由通过 `child_process` 调用 `scripts/ml_engine.py`（Python + sklearn/pandas/numpy），非预制结果。

## 部署架构：Vercel（生产）+ 本地 spawn（开发）双模式

项目已迁移为可部署 Vercel。ML 逻辑（Python sklearn）按环境走两条路径，由 `src/lib/ml-engine.ts` 的 `isVercel()`（`process.env.VERCEL`）切换：

- **本地开发**：Next.js 路由 spawn `python3 scripts/ml_engine.py <cmd> <json>`（Windows 上自动探测 `python`/`python3`），train 逐行流式转发 stdout。
- **Vercel 生产**：Next.js 路由 fetch `/api/py/<cmd>`（`api/py/<cmd>/index.py` Python Serverless Functions），train 由路由把返回的事件数组合成 SSE 流，**前端契约与本地完全一致**。

关键约束与设计决策：

- **Serverless 不能运行时 pip install**——Python 依赖在根目录 `requirements.txt` 声明，Vercel 构建时安装（版本与本地对齐并固定，`random_state=42` 的可复现性依赖库版本）。
- **Python 函数只打包 api/ 目录可达文件**——`public/dataset.csv` 在函数文件系统不可达，数据集在 `api/py/shared/data/dataset.csv` 有一份拷贝（`public/` 原文件保留给首页下载链接，**两份必须保持一致**）。
- **共享模块**：ML 核心在 `api/py/shared/ml_engine.py`（单一事实来源），`scripts/ml_engine.py` 是薄 CLI 包装，`api/py/<cmd>/index.py` 通过 `api/py/shared/endpoint.py` 的 `make_handler()` 工厂生成入口。
- **兜底 handler**：`ml_engine.py` 和 `endpoint.py` 末尾各有 404 `handler`——Vercel 可能把 api/ 下任意 .py 注册为函数入口，没有 handler 会构建失败；定义为 404 后两种情况都安全。
- Python 版本固定在 `.python-version`（3.12）；Python 函数 maxDuration 在 `vercel.json` 设为 60s（Hobby 上限 300s，sklearn 冷启动 + 学习曲线计算 60s 足够）；Next 路由内用 `export const maxDuration` 设置。
- Vercel Python runtime 打包**无 tree-shaking**（全量打包）；若部署时报体积超限（Python 上限 500MB），在 vercel.json `functions` 里给 `api/py/**/*.py` 加 `excludeFiles` 裁剪。

## 常用命令

包管理器**只允许 pnpm**（`preinstall` 钩子强制）。

```bash
pnpm install                  # 安装 Node 依赖
bash ./scripts/prepare.sh     # 首次运行前：pip install scikit-learn pandas numpy
pnpm dev                      # 开发服务器（端口 5000，tsx watch src/server.ts）
pnpm build                    # pnpm install → pip install → next build → tsup 打包 server.ts
pnpm start                    # 生产模式启动（node dist/server.js，端口 5000）
pnpm lint                     # ESLint
pnpm ts-check                 # TypeScript 检查
pnpm validate                 # 并行跑 ts-check + lint:build
```

- 开发环境需先跑 `prepare.sh` 装 Python 依赖，否则 API 路由全部 500。Python 依赖的权威清单是根目录 `requirements.txt`（与 Vercel 构建共用；`prepare.sh`/`build.sh` 里的 pip install 命令保持与之同步）。
- Python 引擎直接测试：`python3 scripts/ml_engine.py info '{}'`（五个命令：`info` / `explore` / `clean` / `train` / `predict`，第二个参数为 JSON 字符串，stdout 输出 JSON）。Windows 上如无 `python3` 命令需用 `python`。
- Shell 脚本为 bash 语法，Windows 下需在 Git Bash / WSL 中运行。

## 架构

### 数据流（核心链路）

```
页面 (src/app/{explore,clean,train,evaluate,predict,versions}/page.tsx)
  → fetch Next.js API 路由 (src/app/api/**/route.ts)
    → 本地: spawn python3 scripts/ml_engine.py <command> <json-config>
      → Vercel: fetch /api/py/<command>（Python Serverless Function）
      → 引擎读数据集，返回 JSON
  → 更新 MLContext 全局状态 (src/lib/MLContext.tsx)
```

- `src/lib/ml-engine.ts`：双模式调用层（`runEngine()` / `isVercel()` / `pythonFunctionUrl()` / `resolvePython()`），所有路由统一经它调引擎。
- `api/model/train/route.ts`：SSE 流式。本地模式逐行转发 Python stdout；Vercel 模式把 Python 函数返回的 `{"events": [...]}` 逐个转成 `data: ...\n\n`。事件类型：`log` / `lc_progress` / `learning_curve` / `model_structure` / `result` / `error`，收到 `type === 'result'` 关闭流。
- `api/model/predict/route.ts`：**预测通过重新训练实现**——predict 命令接收完整 cleanConfig，重训后对该乘客打分，前端不持久化模型。
- 其余 data 路由（info/explore/clean/preview）为一次性 JSON 调用。

### ml_engine 契约（api/py/shared/ml_engine.py）

- 对外入口：`run_command(command, config)`（非 train，返回 dict）、`run_train_collect(config)`（返回事件数组）、`run_train(config, emit)`（emit 回调输出事件）。
- 配置结构：`cleanConfig`（missingValueStrategy 的 Age/Embarked/Cabin 策略、selectedFeatures、nameStrategy）+ `modelType`（logistic_regression / decision_tree / random_forest）+ `hyperparams` + `testSize`。
- 特征工程：Name 提取 Title（低频合并为 Rare）、Cabin 提取 Deck 首字母。
- 所有输出经 `_sanitize_nan` 递归清洗 NaN/Infinity → null，保证 JSON 合法——新增返回字段时必须走这套机制。
- 该模块不得直接 print / sys.exit（两个消费方：CLI 与 serverless 函数）。

### 前端状态与版本管理

- `MLContext.tsx`（React Context）持有全部流程状态；所有页面组件为 `'use client'`。
- 模型版本存 **localStorage**（key `titanic-ml-versions` + `titanic-ml-active-version`），`ModelVersion = {versionId, name, createdAt, trainConfig, trainResult}`；evaluate/predict 页顶部下拉切换版本，versions 页支持重命名/删除/导出(JSON)/导入。
- 逻辑回归 + 未填充缺失值时，predict 页禁用 CabinDeck/Title 输入（对应 ml_engine 的特征可用性约束）。

### COZE 遗留物（保留不动，Vercel 上不生效）

- `src/server.ts`：自定义 HTTP 服务器包 Next.js，由 `dev.sh`/`start.sh` 启动，`build.sh` 用 tsup 打包到 `dist/`。仅本地/COZE 用；Vercel 直接用 `next build` 产物 + `api/` 函数。
- `.coze` / `.cozeproj/`（原型 HTML） / `scripts/*.sh`：平台配置与容器脚本，本地开发流程仍依赖。
- `coze-coding-dev-sdk`、`.npmrc` 中的 npmmirror 镜像、`next.config.ts` 的 `allowedDevOrigins: ['*.dev.coze.site']` 均为 COZE 环境产物。
