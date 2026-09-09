#!/usr/bin/env python3
"""
Titanic ML Engine — 核心机器学习引擎（共享模块）
基于 sklearn 实现：数据加载、清洗、特征工程、模型训练、评估、预测

两个消费方共用本模块，勿在此处直接 print / sys.exit：
- CLI：scripts/ml_engine.py（本地开发，spawn 调用）
- Vercel Python Functions：api/py/<command>/index.py（生产环境）

train 通过 emit 回调输出事件（log/lc_progress/learning_curve/model_structure/result），
其余命令通过 run_command() 返回 dict。
"""

import sys
import json
import os
import math
import time
import warnings
import traceback


def _sanitize_nan(obj):
    """递归将 NaN/Infinity 替换为 None，确保输出合法 JSON"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_nan(v) for v in obj)
    return obj


def json_safe_dumps(obj, **kwargs):
    """json.dumps 的安全版本，自动处理 NaN/Infinity"""
    return json.dumps(_sanitize_nan(obj), **kwargs)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ── 数据路径 ──
# Vercel serverless 函数只打包 api/ 目录，public/ 在函数文件系统中不可达，
# 因此数据集在本模块 data/ 下有一份拷贝（须与 public/dataset.csv 保持一致）。
# 本地开发时若 data/ 拷贝缺失，回退到仓库 public/ 下的原文件。
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_CANDIDATES = [
    os.path.join(_MODULE_DIR, "data", "dataset.csv"),
    os.path.join(_MODULE_DIR, "..", "..", "..", "public", "dataset.csv"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])


def load_data():
    """加载原始数据"""
    df = pd.read_csv(DATA_PATH)
    return df


def _extract_title(name):
    """从姓名中提取头衔"""
    import re
    match = re.search(r',\s*([^\.]+)\.', str(name))
    if match:
        title = match.group(1).strip()
        # 合并低频头衔
        rare = {'Lady', 'Countess', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                'Rev', 'Sir', 'Jonkheer', 'Dona'}
        if title in rare:
            return 'Rare'
        if title == 'Mlle':
            return 'Miss'
        if title == 'Ms':
            return 'Miss'
        if title == 'Mme':
            return 'Mrs'
        return title
    return 'Unknown'


def clean_data(df, config):
    """
    根据用户配置清洗数据
    config: {
      "missingValueStrategy": {"Age": "median", "Embarked": "mode", "Cabin": "drop"},
      "selectedFeatures": ["Pclass", "Sex", "Age", ...],
      "nameStrategy": "extract_title" | "drop",
      "cabinStrategy": "extract_deck" | "drop",
    }
    """
    df = df.copy()
    missing_strategy = config.get("missingValueStrategy", {})
    
    # Age 填充
    age_strategy = missing_strategy.get("Age", "median")
    if age_strategy == "mean":
        df["Age"] = df["Age"].fillna(df["Age"].mean())
    elif age_strategy == "median":
        df["Age"] = df["Age"].fillna(df["Age"].median())
    elif age_strategy == "mode":
        df["Age"] = df["Age"].fillna(df["Age"].mode()[0])
    # "none" = 保留缺失值不填充
    
    # Embarked 填充
    embarked_strategy = missing_strategy.get("Embarked", "mode")
    if embarked_strategy == "mode":
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    elif embarked_strategy == "drop":
        df = df.dropna(subset=["Embarked"])
    # "none" = 保留缺失值不填充
    
    # Cabin 处理
    # Cabin strategy: accept from cabinStrategy or missingValueStrategy.Cabin
    # Frontend sends missingValueStrategy.Cabin, map values: "deck"→"extract_deck", "drop"→"drop", "keep"→"keep", "none"→"none"
    cabin_raw = missing_strategy.get("Cabin", "drop")
    cabin_map = {"deck": "extract_deck", "drop": "drop", "keep": "keep", "none": "none",
                 "extract_deck": "extract_deck"}  # also accept direct values
    cabin_strategy = config.get("cabinStrategy", cabin_map.get(cabin_raw, "drop"))
    if cabin_strategy == "extract_deck":
        # 提取CabinDeck首字母，缺失值保留为NaN而非填充'U'
        # 这样缺失值检测能正确识别未填充的CabinDeck
        df["CabinDeck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) and str(x).strip() != "" else None)
        df = df.drop(columns=["Cabin"])
    elif cabin_strategy == "keep":
        # 保留Cabin原始列，不提取也不删除（用于后续选择是否丢弃）
        pass
    else:
        df = df.drop(columns=["Cabin"])
    
    # Name 处理
    name_strategy = config.get("nameStrategy", "drop")
    if name_strategy == "extract_title":
        df["Title"] = df["Name"].apply(_extract_title)
        df = df.drop(columns=["Name"])
    else:
        df = df.drop(columns=["Name"])
    
    return df


def encode_features(df, selected_features):
    """特征编码"""
    df = df.copy()
    
    # Sex 编码
    if "Sex" in selected_features:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    
    # Embarked 编码 (one-hot)
    if "Embarked" in selected_features:
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
        df = pd.concat([df, embarked_dummies], axis=1)
        selected_features = [f for f in selected_features if f != "Embarked"]
        selected_features.extend(embarked_dummies.columns.tolist())
    
    # CabinDeck 编码 (one-hot)
    if "CabinDeck" in selected_features:
        deck_dummies = pd.get_dummies(df["CabinDeck"], prefix="Deck")
        df = pd.concat([df, deck_dummies], axis=1)
        selected_features = [f for f in selected_features if f != "CabinDeck"]
        selected_features.extend(deck_dummies.columns.tolist())
    
    # Cabin 编码 (one-hot, 仅当保留且被选中时)
    if "Cabin" in selected_features and "Cabin" in df.columns:
        cabin_dummies = pd.get_dummies(df["Cabin"], prefix="Cabin")
        df = pd.concat([df, cabin_dummies], axis=1)
        selected_features = [f for f in selected_features if f != "Cabin"]
        selected_features.extend(cabin_dummies.columns.tolist())
    
    # Title 编码 (one-hot)
    if "Title" in selected_features:
        title_dummies = pd.get_dummies(df["Title"], prefix="Title")
        df = pd.concat([df, title_dummies], axis=1)
        selected_features = [f for f in selected_features if f != "Title"]
        selected_features.extend(title_dummies.columns.tolist())
    
    return df, selected_features


def get_data_info():
    """获取数据概览信息"""
    df = load_data()
    total = len(df)
    survived = int(df["Survived"].sum())
    
    # 缺失值
    missing = {}
    for col in df.columns:
        count = int(df[col].isnull().sum())
        if count > 0:
            missing[col] = {"count": count, "percentage": round(count / total * 100, 1)}
    
    return {
        "total": total,
        "survived": survived,
        "died": total - survived,
        "survivalRate": round(survived / total * 100, 1),
        "missingValues": missing,
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }


def get_explore_data():
    """获取可视化所需的聚合数据"""
    df = load_data()
    total = len(df)
    
    # 生还比例
    survived = int(df["Survived"].sum())
    
    # 性别 vs 生还
    sex_survival = {}
    for sex in ["male", "female"]:
        subset = df[df["Sex"] == sex]
        s = int(subset["Survived"].sum())
        d = len(subset) - s
        sex_survival[sex] = {"survived": s, "died": d, "rate": round(s / len(subset) * 100, 1)}
    
    # 舱等 vs 生还
    pclass_survival = {}
    for pclass in sorted(df["Pclass"].unique()):
        subset = df[df["Pclass"] == pclass]
        s = int(subset["Survived"].sum())
        d = len(subset) - s
        pclass_survival[str(pclass)] = {"survived": s, "died": d, "rate": round(s / len(subset) * 100, 1)}
    
    # 登船港 vs 生还
    embarked_survival = {}
    for port in ["S", "C", "Q"]:
        subset = df[df["Embarked"] == port]
        if len(subset) > 0:
            s = int(subset["Survived"].sum())
            d = len(subset) - s
            embarked_survival[port] = {"survived": s, "died": d, "rate": round(s / len(subset) * 100, 1)}
    
    # 年龄分布（按生还分组的直方图数据）
    age_bins = list(range(0, 85, 5))
    age_hist = {"bins": age_bins, "survived": [], "died": []}
    for i in range(len(age_bins) - 1):
        lo, hi = age_bins[i], age_bins[i + 1]
        age_hist["survived"].append(int(((df["Age"] >= lo) & (df["Age"] < hi) & (df["Survived"] == 1)).sum()))
        age_hist["died"].append(int(((df["Age"] >= lo) & (df["Age"] < hi) & (df["Survived"] == 0)).sum()))
    
    # 票价统计
    fare_stats = {
        "survived": {"mean": round(float(df[df["Survived"] == 1]["Fare"].mean()), 1),
                      "median": round(float(df[df["Survived"] == 1]["Fare"].median()), 1),
                      "q25": round(float(df[df["Survived"] == 1]["Fare"].quantile(0.25)), 1),
                      "q75": round(float(df[df["Survived"] == 1]["Fare"].quantile(0.75)), 1)},
        "died": {"mean": round(float(df[df["Survived"] == 0]["Fare"].mean()), 1),
                 "median": round(float(df[df["Survived"] == 0]["Fare"].median()), 1),
                 "q25": round(float(df[df["Survived"] == 0]["Fare"].quantile(0.25)), 1),
                 "q75": round(float(df[df["Survived"] == 0]["Fare"].quantile(0.75)), 1)},
    }
    
    # 特征相关性（包含 CabinDeck 和 Title 编码）
    numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    df_numeric = df[numeric_cols].copy()
    df_numeric["SexCode"] = df["Sex"].map({"male": 0, "female": 1})
    # CabinDeck: 提取首字母并编码
    df_numeric["CabinDeckCode"] = df["Cabin"].apply(lambda x: ord(str(x)[0]) - ord('A') + 1 if pd.notna(x) and str(x).strip() != "" else 0)
    # Title: 提取头衔并编码
    df_numeric["TitleCode"] = df["Name"].apply(_extract_title).map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).fillna(0).astype(int)
    corr = df_numeric.corr()
    corr_data = {
        "columns": list(corr.columns),
        "matrix": [[round(v, 3) for v in row] for row in corr.values.tolist()]
    }
    
    # 特征频数分布（排除 Cabin, Ticket, Name, PassengerId）
    feature_distributions = {}
    
    # 提取 CabinDeck 和 Title 列用于分布统计
    df["CabinDeck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) and str(x).strip() != "" else None)
    df["Title"] = df["Name"].apply(_extract_title)
    
    # 类别型特征
    categorical_features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "CabinDeck", "Title"]
    for feat in categorical_features:
        if feat not in df.columns:
            continue
        values = sorted(df[feat].dropna().unique(), key=lambda x: str(x))
        # 映射显示名称
        display_map = {}
        if feat == "Pclass":
            display_map = {"1": "1等舱", "2": "2等舱", "3": "3等舱"}
        elif feat == "Sex":
            display_map = {"male": "男性", "female": "女性"}
        elif feat == "Embarked":
            display_map = {"S": "南安普顿", "C": "瑟堡", "Q": "皇后镇"}
        elif feat == "Survived":
            display_map = {"0": "遇难", "1": "生还"}
        elif feat == "CabinDeck":
            display_map = {d: f"{d}甲板" for d in "ABCDEFGT"}
        elif feat == "Title":
            display_map = {"Mr": "Mr先生", "Miss": "Miss小姐", "Mrs": "Mrs女士", "Master": "Master少爷", "Rare": "Rare稀有"}
        
        counts = {}
        survived_counts = {}
        for v in values:
            key = str(v)
            subset = df[df[feat] == v]
            counts[key] = int(len(subset))
            survived_counts[key] = int(subset["Survived"].sum())
        
        feature_distributions[feat] = {
            "type": "categorical",
            "values": [str(v) for v in values],
            "labels": {str(v): display_map.get(str(v), str(v)) for v in values},
            "counts": counts,
            "survivedCounts": survived_counts,
        }
    
    # 数值型特征：分箱直方图
    numerical_features = {"Age": (0, 80, 8), "Fare": (0, 520, 10)}
    for feat, (lo, hi, n_bins) in numerical_features.items():
        if feat not in df.columns:
            continue
        bin_width = (hi - lo) / n_bins
        bins = [round(lo + i * bin_width) for i in range(n_bins + 1)]
        bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(n_bins)]
        counts = []
        survived_counts = []
        for i in range(n_bins):
            mask = (df[feat] >= bins[i]) & (df[feat] < bins[i + 1])
            if i == n_bins - 1:
                mask = (df[feat] >= bins[i]) & (df[feat] <= bins[i + 1])
            counts.append(int(mask.sum()))
            survived_counts.append(int((mask & (df["Survived"] == 1)).sum()))
        
        feature_distributions[feat] = {
            "type": "numerical",
            "bins": bin_labels,
            "counts": counts,
            "survivedCounts": survived_counts,
        }
    
    return {
        "total": total,
        "survived": survived,
        "died": total - survived,
        "survivalRate": round(survived / total * 100, 1),
        "sexSurvival": sex_survival,
        "pclassSurvival": pclass_survival,
        "embarkedSurvival": embarked_survival,
        "ageHistogram": age_hist,
        "fareStats": fare_stats,
        "correlation": corr_data,
        "featureDistributions": feature_distributions,
    }


def _tree_to_json(tree_model, feature_names):
    """Convert sklearn decision tree to JSON for frontend visualization"""
    tree = tree_model.tree_
    
    def recurse(node_id, depth=0):
        if tree.children_left[node_id] == -1:  # leaf
            value = tree.value[node_id][0]
            pred_class = int(value.argmax())
            total = int(value.sum())
            return {
                "type": "leaf",
                "samples": total,
                "value": [int(v) for v in value],
                "prediction": pred_class,
                "depth": depth,
            }
        feature = feature_names[tree.feature[node_id]] if tree.feature[node_id] < len(feature_names) else f"feature_{tree.feature[node_id]}"
        threshold = round(float(tree.threshold[node_id]), 4)
        return {
            "type": "split",
            "feature": feature,
            "threshold": threshold,
            "samples": int(tree.n_node_samples[node_id]),
            "depth": depth,
            "left": recurse(tree.children_left[node_id], depth + 1),
            "right": recurse(tree.children_right[node_id], depth + 1),
        }
    
    # Only return top 3 levels to keep it manageable
    def recurse_limited(node_id, depth=0, max_depth=3):
        if depth > max_depth:
            return {"type": "truncated", "depth": depth}
        if tree.children_left[node_id] == -1:  # leaf
            value = tree.value[node_id][0]
            pred_class = int(value.argmax())
            total = int(value.sum())
            return {
                "type": "leaf",
                "samples": total,
                "value": [int(v) for v in value],
                "prediction": pred_class,
                "depth": depth,
            }
        feat_idx = int(tree.feature[node_id])
        feature = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
        threshold = round(float(tree.threshold[node_id]), 4)
        return {
            "type": "split",
            "feature": feature,
            "threshold": threshold,
            "samples": int(tree.n_node_samples[node_id]),
            "depth": depth,
            "left": recurse_limited(int(tree.children_left[node_id]), depth + 1, max_depth),
            "right": recurse_limited(int(tree.children_right[node_id]), depth + 1, max_depth),
        }
    
    return recurse_limited(0)


def run_train(config, emit):
    """
    训练模型，通过 emit 回调输出事件流
    config: {
      "cleanConfig": {...},
      "modelType": "logistic_regression" | "decision_tree" | "random_forest",
      "hyperparams": {...},
      "testSize": 0.3,
      "featureWeights": {...}
    }
    emit: callable(dict) — CLI 用逐行打印到 stdout，serverless 函数收集为数组
    """
    # 发送带时间戳和类型的详细日志
    import time as _time
    _t0 = _time.time()

    def log(msg, log_type="info"):
        ts = _time.strftime("%H:%M:%S")
        emit({"type": "log", "message": msg, "logType": log_type, "timestamp": ts})

    def result(data):
        emit({"type": "result", "data": data})

    def lc_progress(current, total, data):
        emit({"type": "lc_progress", "current": current, "total": total, "data": data})
    
    try:
        log("正在加载数据集...", "info")
        df = load_data()
        missing_info = {col: {"count": int(df[col].isnull().sum()), "pct": round(df[col].isnull().sum() / len(df) * 100, 1)} for col in df.columns if df[col].isnull().sum() > 0}
        log(f"数据集加载完成: {len(df)}条记录, {len(df.columns)}个字段 ({', '.join(df.columns[:6])}...)", "success")
        if missing_info:
            for col, info in missing_info.items():
                log(f"  缺失值 → {col}: {info['count']}个 ({info['pct']}%)", "detail")
        
        # 清洗数据
        clean_config = config.get("cleanConfig", {})
        log("开始数据清洗...", "info")
        df = clean_data(df, clean_config)
        
        # 详细清洗日志
        mvs = clean_config.get("missingValueStrategy", {})
        age_strategy = mvs.get("Age", clean_config.get("ageStrategy", "median"))
        age_missing = missing_info.get("Age", {})
        if age_missing:
            if age_strategy == "median":
                log(f"  Age缺失值: {age_missing.get('count', 177)}个 ({age_missing.get('pct', 19.9)}%), 使用中位数填充", "detail")
            elif age_strategy == "mean":
                log(f"  Age缺失值: {age_missing.get('count', 177)}个, 使用均值填充", "detail")
            elif age_strategy == "mode":
                log(f"  Age缺失值: {age_missing.get('count', 177)}个, 使用众数填充", "detail")
            else:
                log(f"  Age缺失值: {age_missing.get('count', 177)}个, 未填充", "detail")
        
        embarked_strategy = mvs.get("Embarked", clean_config.get("embarkedStrategy", "mode"))
        emb_missing = missing_info.get("Embarked", {})
        if emb_missing:
            strategy_desc = {"mode": "使用众数 'S' 填充", "drop": "删除对应行", "none": "未填充"}.get(embarked_strategy, embarked_strategy)
            log(f"  Embarked缺失值: {emb_missing.get('count', 2)}个, {strategy_desc}", "detail")
        
        cabin_raw = mvs.get("Cabin", clean_config.get("cabinStrategy", "drop"))
        cabin_map = {"deck": "extract_deck", "drop": "删除Cabin列", "keep": "保留原始值", "none": "未处理", "extract_deck": "提取甲板层"}
        cabin_strategy = clean_config.get("cabinStrategy", cabin_raw)
        cab_missing = missing_info.get("Cabin", {})
        if cab_missing:
            strategy_desc = cabin_map.get(cabin_strategy, cabin_strategy)
            log(f"  Cabin缺失值: {cab_missing.get('count', 687)}个 ({cab_missing.get('pct', 77.1)}%), 策略: {strategy_desc}", "detail")
        
        name_strategy = clean_config.get("nameStrategy", "drop")
        if name_strategy == "title":
            log(f"  姓名处理: 提取头衔特征 (Mr, Miss, Mrs, Master, Rare)", "detail")
        
        log(f"数据清洗完成: {len(df)}条记录, 缺失值已处理", "success")
        
        # 获取选中的特征
        selected_features = clean_config.get("selectedFeatures", ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])
        log(f"选中特征: {', '.join(selected_features)}", "info")
        
        # 在编码前检测缺失值
        pre_encode_nan_cols = [f for f in selected_features if f in df.columns and df[f].isna().any()]
        pre_encode_nan_count = sum(int(df[f].isna().sum()) for f in pre_encode_nan_cols)
        has_pre_encode_nan = pre_encode_nan_count > 0
        if has_pre_encode_nan:
            log(f"编码前特征中存在缺失值: {', '.join(pre_encode_nan_cols)} (共 {pre_encode_nan_count} 个)", "warning")
        
        # 特征编码
        log("特征编码: 独热编码分类特征...", "info")
        n_cols_before = len(df.columns)
        df_raw = df.copy()
        df, selected_features = encode_features(df, selected_features)
        n_new_cols = len(df.columns) - n_cols_before
        log(f"独热编码 → 扩展为 {len(df.columns)} 个特征列 (+{n_new_cols} 列)", "detail")
        
        # 确保特征列存在
        available_features = [f for f in selected_features if f in df.columns]
        
        # 构建特征矩阵
        X = df[available_features].copy()
        y = df["Survived"].copy()
        
        # 编码后也检测缺失值
        post_encode_nan_count = int(X.isna().sum().sum())
        has_nan = post_encode_nan_count > 0
        if post_encode_nan_count > 0:
            nan_cols = [col for col in X.columns if X[col].isna().any()]
            log(f"编码后特征中存在缺失值: {', '.join(nan_cols)} (共 {post_encode_nan_count} 个)", "warning")
        
        # 逻辑回归不支持缺失值
        model_type = config.get("modelType", "logistic_regression")
        if model_type == "logistic_regression" and has_nan:
            log("逻辑回归不支持缺失值，请先填充所有缺失值，或选择决策树/随机森林模型", "error")
            result({"error": "逻辑回归不支持缺失值。请先填充所有缺失值，或选择决策树/随机森林模型。", "hasUnfilledMissing": True})
            return
        
        # 划分训练集和测试集
        test_size = config.get("testSize", 0.3)
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        train_n, test_n = len(X_train), len(X_test)
        train_survived = int(y_train.sum())
        train_died = train_n - train_survived
        test_survived = int(y_test.sum())
        test_died = test_n - test_survived
        log(f"训练/测试集划分: {train_n}/{test_n} ({1-test_size:.0%}/{test_size:.0%}), 分层抽样", "info")
        log(f"  训练集: 生还 {train_survived} ({train_survived/train_n*100:.1f}%), 遇难 {train_died} ({train_died/train_n*100:.1f}%)", "detail")
        log(f"  测试集: 生还 {test_survived} ({test_survived/test_n*100:.1f}%), 遇难 {test_died} ({test_died/test_n*100:.1f}%)", "detail")
        
        # 标准化（仅对逻辑回归）
        hyperparams = config.get("hyperparams", {})
        scaler = None
        
        if model_type == "logistic_regression":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            C = hyperparams.get("C", 1.0)
            max_iter = hyperparams.get("maxIter", 1000)
            solver = hyperparams.get("solver", "lbfgs")
            penalty = hyperparams.get("penalty", "l2")
            # Validate solver-penalty compatibility
            if solver == "lbfgs" and penalty not in ("l2", None, "none"):
                penalty = "l2"
            elif solver == "liblinear" and penalty not in ("l1", "l2"):
                penalty = "l2"
            elif solver == "saga" and penalty not in ("l1", "l2", "elasticnet", None, "none"):
                penalty = "l2"
            if penalty == "none":
                penalty = None
            log(f"创建逻辑回归模型: C={C}, max_iter={max_iter}, solver={solver}, penalty={penalty}", "info")
            lr_kwargs = {"C": C, "max_iter": max_iter, "solver": solver, "random_state": random_state}
            if penalty is not None:
                lr_kwargs["penalty"] = penalty
            else:
                lr_kwargs["penalty"] = None
            model = LogisticRegression(**lr_kwargs)
            
            log("开始训练...", "warning")
            _t_train = _time.time()
            model.fit(X_train_scaled, y_train)
            _train_time = _time.time() - _t_train
            log(f"训练完成! 耗时 {_train_time:.2f}s", "success")
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
            
        elif model_type == "decision_tree":
            max_depth = hyperparams.get("maxDepth", 5)
            min_samples_split = hyperparams.get("minSamplesSplit", 2)
            min_samples_leaf = hyperparams.get("minSamplesLeaf", 1)
            criterion = hyperparams.get("criterion", "gini")
            if max_depth == 0 or max_depth is None:
                max_depth = None
            log(f"创建决策树模型: max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, criterion={criterion}", "info")
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=random_state
            )
            
            log("开始训练...", "warning")
            _t_train = _time.time()
            model.fit(X_train, y_train)
            _train_time = _time.time() - _t_train
            log(f"训练完成! 耗时 {_train_time:.2f}s", "success")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
        elif model_type == "random_forest":
            n_estimators = hyperparams.get("nEstimators", 100)
            max_depth = hyperparams.get("maxDepth", 5)
            min_samples_split = hyperparams.get("minSamplesSplit", 2)
            if max_depth == 0 or max_depth is None:
                max_depth = None
            log(f"创建随机森林模型: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}", "info")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
            
            log("开始训练...", "warning")
            _t_train = _time.time()
            model.fit(X_train, y_train)
            _train_time = _time.time() - _t_train
            log(f"训练完成! 耗时 {_train_time:.2f}s", "success")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
        
        # 计算学习曲线 - 流式推送20个采样点
        log("计算学习曲线 (5-fold CV, 20个采样点)...", "info")
        from sklearn.model_selection import learning_curve as sk_learning_curve
        X_for_lc = X_train_scaled if scaler else X_train
        
        # 生成20个均匀分布的采样比例
        lc_ratios = [round(0.05 + i * 0.95 / 19, 4) for i in range(20)]
        lc_n_points = len(lc_ratios)
        
        # 手动逐点计算学习曲线，每完成一个点就推送
        lc_train_sizes = []
        lc_train_scores = []
        lc_val_scores = []
        lc_train_stds = []
        lc_val_stds = []
        
        for idx, ratio in enumerate(lc_ratios):
            ts_abs, ts_scores, vs_scores = sk_learning_curve(
                model, X_for_lc, y_train,
                train_sizes=[ratio],
                cv=5, scoring='accuracy', random_state=42,
            )
            train_mean = round(float(ts_scores.mean()), 4)
            val_mean = round(float(vs_scores.mean()), 4)
            train_std = round(float(ts_scores.std()), 4)
            val_std = round(float(vs_scores.std()), 4)
            sample_pct = round(ratio * 100)
            
            lc_train_sizes.append(sample_pct)
            lc_train_scores.append(train_mean)
            lc_val_scores.append(val_mean)
            lc_train_stds.append(train_std)
            lc_val_stds.append(val_std)
            
            # 流式推送当前进度和累积数据
            current_data = {
                "trainSizes": lc_train_sizes,
                "trainScores": lc_train_scores,
                "valScores": lc_val_scores,
                "trainScoresStd": lc_train_stds,
                "valScoresStd": lc_val_stds,
            }
            lc_progress(idx + 1, lc_n_points, current_data)
        
        # 最终发送完整的学习曲线
        learning_curve_data = {
            "trainSizes": lc_train_sizes,
            "trainScores": lc_train_scores,
            "valScores": lc_val_scores,
            "trainScoresStd": lc_train_stds,
            "valScoresStd": lc_val_stds,
        }
        emit({"type": "learning_curve", "data": learning_curve_data})
        log(f"学习曲线计算完成 ({lc_n_points}个采样点)", "success")
        
        # 模型结构可视化数据
        model_structure = {}
        if model_type == "logistic_regression":
            # 逻辑回归：系数权重条形图
            coefs = model.coef_[0].tolist()
            model_structure = {
                "type": "coefficients",
                "features": available_features,
                "coefficients": [round(c, 4) for c in coefs],
                "intercept": round(float(model.intercept_[0]), 4),
            }
        elif model_type == "decision_tree":
            # 决策树：文本结构导出
            from sklearn.tree import export_text
            tree_text = export_text(model, feature_names=available_features, max_depth=4)
            # 导出树结构 JSON
            tree_data = _tree_to_json(model, available_features)
            model_structure = {
                "type": "tree",
                "maxDepth": int(model.get_depth()),
                "nNodes": int(model.tree_.node_count),
                "nLeaves": int(model.get_n_leaves()),
                "treeText": tree_text[:2000],  # 限制长度
                "treeData": tree_data,
            }
        elif model_type == "random_forest":
            # 随机森林：OOB 曲线 + 各树深度
            model_oob = RandomForestClassifier(
                n_estimators=hyperparams.get("nEstimators", 100),
                max_depth=hyperparams.get("maxDepth", 5) if hyperparams.get("maxDepth", 5) not in (0, None) else None,
                min_samples_split=hyperparams.get("minSamplesSplit", 2),
                random_state=42, oob_score=True, warm_start=True,
            )
            oob_errors = []
            n_trees_list = list(range(1, hyperparams.get("nEstimators", 100) + 1))
            # 每隔几棵树采样一次
            step = max(1, len(n_trees_list) // 20)
            sampled_trees = n_trees_list[::step]
            if n_trees_list[-1] not in sampled_trees:
                sampled_trees.append(n_trees_list[-1])
            for i in sampled_trees:
                model_oob.set_params(n_estimators=i)
                model_oob.fit(X_train, y_train)
                oob_errors.append(round(1 - model_oob.oob_score_, 4))
            model_structure = {
                "type": "forest",
                "nTrees": hyperparams.get("nEstimators", 100),
                "oobCurve": {"nTrees": sampled_trees, "oobErrors": oob_errors},
                "treeDepths": [int(t.get_depth()) for t in model.estimators_[:20]],
            }
        
        emit({"type": "model_structure", "data": model_structure})
        
        # 模型结构日志
        if model_type == "decision_tree":
            log(f"决策树结构: 深度{model_structure.get('maxDepth', '?')}, {model_structure.get('nNodes', '?')}个节点, {model_structure.get('nLeaves', '?')}个叶子节点", "detail")
        elif model_type == "logistic_regression":
            log("逻辑回归: 特征权重已解析", "detail")
        elif model_type == "random_forest":
            log(f"随机森林: {model_structure.get('nTrees', '?')}棵树, OOB曲线已生成", "detail")
        
        log("正在评估模型性能...", "info")
        
        # 计算评估指标
        acc = round(float(accuracy_score(y_test, y_pred)), 4)
        prec = round(float(precision_score(y_test, y_pred)), 4)
        rec = round(float(recall_score(y_test, y_pred)), 4)
        f1 = round(float(f1_score(y_test, y_pred)), 4)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        
        log(f"准确率: {acc:.2%} | 精确率: {prec:.2%} | 召回率: {rec:.2%} | F1: {f1:.2%}", "success")
        
        _total_time = _time.time() - _t0
        log(f"训练流程全部完成! 总耗时 {_total_time:.2f}s", "success")
        
        # 特征重要性
        feature_importance = []
        if model_type == "logistic_regression":
            importances = abs(model.coef_[0])
        else:
            importances = model.feature_importances_
        
        for i, feat in enumerate(available_features):
            feature_importance.append({
                "feature": feat,
                "importance": round(float(importances[i]), 4)
            })
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        
        # 分组准确率（基于编码前的原始数据，对所有训练特征分组）
        group_accuracy = {}
        df_test_raw = df_raw.loc[X_test.index]
        
        # 按性别
        sex_acc = {}
        for sex_name in ["male", "female"]:
            mask = df_test_raw["Sex"] == sex_name
            if mask.sum() > 0:
                sex_acc[sex_name] = round(float(accuracy_score(
                    y_test[mask], pd.Series(y_pred, index=y_test.index)[mask]
                )), 4)
        group_accuracy["bySex"] = sex_acc
        
        # 按舱等
        if "Pclass" in df_test_raw.columns:
            pclass_acc = {}
            for pclass in sorted(df_test_raw["Pclass"].unique()):
                mask = df_test_raw["Pclass"] == pclass
                if mask.sum() > 0:
                    pclass_acc[str(pclass)] = round(float(accuracy_score(
                        y_test[mask], pd.Series(y_pred, index=y_test.index)[mask]
                    )), 4)
            group_accuracy["byPclass"] = pclass_acc
        
        # 按年龄段
        if "Age" in df_test_raw.columns:
            age_acc = {}
            age_groups = {"child": (0, 12), "adult": (13, 60), "elder": (61, 100)}
            for group_name, (lo, hi) in age_groups.items():
                mask = (df_test_raw["Age"] >= lo) & (df_test_raw["Age"] <= hi)
                if mask.sum() > 0:
                    age_acc[group_name] = round(float(accuracy_score(
                        y_test[mask], pd.Series(y_pred, index=y_test.index)[mask]
                    )), 4)
            group_accuracy["byAgeGroup"] = age_acc
        
        # 按登船港口
        if "Embarked" in df_test_raw.columns:
            embarked_acc = {}
            for port in sorted(df_test_raw["Embarked"].dropna().unique()):
                mask = df_test_raw["Embarked"] == port
                if mask.sum() > 0:
                    embarked_acc[port] = round(float(accuracy_score(
                        y_test[mask], pd.Series(y_pred, index=y_test.index)[mask]
                    )), 4)
            if embarked_acc:
                group_accuracy["byEmbarked"] = embarked_acc
        
        # 按甲板层
        if "CabinDeck" in df_test_raw.columns:
            deck_acc = {}
            for deck in sorted(df_test_raw["CabinDeck"].dropna().unique()):
                mask = df_test_raw["CabinDeck"] == deck
                if mask.sum() >= 5:  # 至少5个样本才统计
                    deck_acc[deck] = round(float(accuracy_score(
                        y_test[mask], pd.Series(y_pred, index=y_test.index)[mask]
                    )), 4)
            if deck_acc:
                group_accuracy["byCabinDeck"] = deck_acc
        
        # 按头衔
        if "Title" in df_test_raw.columns:
            title_acc = {}
            for title in sorted(df_test_raw["Title"].dropna().unique()):
                mask = df_test_raw["Title"] == title
                if mask.sum() >= 5:  # 至少5个样本才统计
                    title_acc[title] = round(float(accuracy_score(
                        y_test[mask], pd.Series(y_pred, index=y_test.index)[mask]
                    )), 4)
            if title_acc:
                group_accuracy["byTitle"] = title_acc
        
        # 按SibSp分组
        if "SibSp" in df_test_raw.columns:
            sibsp_acc = {}
            for val in sorted(df_test_raw["SibSp"].unique()):
                mask = df_test_raw["SibSp"] == val
                if mask.sum() >= 5:
                    sibsp_acc[str(val)] = round(float(accuracy_score(
                        y_test[mask], pd.Series(y_pred, index=y_test.index)[mask]
                    )), 4)
            if sibsp_acc:
                group_accuracy["bySibSp"] = sibsp_acc
        
        # 按Parch分组
        if "Parch" in df_test_raw.columns:
            parch_acc = {}
            for val in sorted(df_test_raw["Parch"].unique()):
                mask = df_test_raw["Parch"] == val
                if mask.sum() >= 5:
                    parch_acc[str(val)] = round(float(accuracy_score(
                        y_test[mask], pd.Series(y_pred, index=y_test.index)[mask]
                    )), 4)
            if parch_acc:
                group_accuracy["byParch"] = parch_acc
        
        # 按Fare分组（低/中/高）
        if "Fare" in df_test_raw.columns:
            fare_acc = {}
            fare_groups = {"low": (0, 15), "medium": (15, 50), "high": (50, 1000)}
            for group_name, (lo, hi) in fare_groups.items():
                mask = (df_test_raw["Fare"] >= lo) & (df_test_raw["Fare"] < hi)
                if mask.sum() >= 5:
                    fare_acc[group_name] = round(float(accuracy_score(
                        y_test[mask], pd.Series(y_pred, index=y_test.index)[mask]
                    )), 4)
            if fare_acc:
                group_accuracy["byFare"] = fare_acc
        
        log("评估完成!")
        
        # 保存模型和预处理信息到临时文件
        model_info = {
            "modelType": model_type,
            "availableFeatures": available_features,
            "scalerUsed": scaler is not None,
        }
        
        # 输出结果
        result({
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            },
            "confusionMatrix": {
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            },
            "featureImportance": feature_importance,
            "groupAccuracy": group_accuracy,
            "trainSize": len(X_train),
            "testSize": len(X_test),
            "modelInfo": model_info,
            "learningCurve": learning_curve_data,
            "hasUnfilledMissing": has_nan,
        })
        
    except Exception as e:
        log(f"ERROR: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        result({"error": str(e)})


def predict(config):
    """
    对单条数据进行预测
    config: {
      "passenger": {"Pclass": 1, "Sex": "female", "Age": 25, ...},
      "modelInfo": {...},  // 来自训练结果
      "cleanConfig": {...}
    }
    """
    try:
        passenger = config.get("passenger", {})
        model_info = config.get("modelInfo", {})
        clean_config = config.get("cleanConfig", {})
        
        # 重新训练模型（因模型对象无法序列化传递）
        df = load_data()
        df = clean_data(df, clean_config)
        selected_features = clean_config.get("selectedFeatures", ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])
        df_encoded, selected_features = encode_features(df, selected_features)
        available_features = [f for f in selected_features if f in df_encoded.columns]
        
        X = df_encoded[available_features].copy()
        y = df_encoded["Survived"].copy()
        
        model_type = model_info.get("modelType", "logistic_regression")
        hyperparams = config.get("hyperparams", {})
        
        scaler = None
        if model_type == "logistic_regression":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            C = hyperparams.get("C", 1.0)
            max_iter = hyperparams.get("maxIter", 1000)
            solver = hyperparams.get("solver", "lbfgs")
            penalty = hyperparams.get("penalty", "l2")
            # Validate solver-penalty compatibility
            if solver == "lbfgs" and penalty not in ("l2", None, "none"):
                penalty = "l2"
            elif solver == "liblinear" and penalty not in ("l1", "l2"):
                penalty = "l2"
            elif solver == "saga" and penalty not in ("l1", "l2", "elasticnet", None, "none"):
                penalty = "l2"
            if penalty == "none":
                penalty = None
            lr_kwargs = {"C": C, "max_iter": max_iter, "solver": solver, "random_state": 42}
            if penalty is not None:
                lr_kwargs["penalty"] = penalty
            else:
                lr_kwargs["penalty"] = None
            model = LogisticRegression(**lr_kwargs)
            model.fit(X_scaled, y)
        elif model_type == "decision_tree":
            max_depth = hyperparams.get("maxDepth", 5)
            min_samples_split = hyperparams.get("minSamplesSplit", 2)
            min_samples_leaf = hyperparams.get("minSamplesLeaf", 1)
            criterion = hyperparams.get("criterion", "gini")
            if max_depth == 0 or max_depth is None:
                max_depth = None
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=42)
            model.fit(X, y)
        elif model_type == "random_forest":
            n_estimators = hyperparams.get("nEstimators", 100)
            max_depth = hyperparams.get("maxDepth", 5)
            min_samples_split = hyperparams.get("minSamplesSplit", 2)
            if max_depth == 0 or max_depth is None:
                max_depth = None
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            model.fit(X, y)
        
        # 构建预测样本
        sample = pd.DataFrame([passenger])
        
        # 编码 Sex
        if "Sex" in sample.columns:
            sample["Sex"] = sample["Sex"].map({"male": 0, "female": 1})
        
        # 编码 Embarked (one-hot)
        if "Embarked" in sample.columns:
            embarked_dummies = pd.get_dummies(sample["Embarked"], prefix="Embarked")
            sample = pd.concat([sample, embarked_dummies], axis=1)
            sample = sample.drop(columns=["Embarked"])
        
        # 编码 CabinDeck (one-hot)
        if "CabinDeck" in sample.columns:
            deck_val = sample["CabinDeck"].iloc[0] if "CabinDeck" in sample.columns else None
            if pd.isna(deck_val) or deck_val == "" or deck_val is None:
                # 没有提供 CabinDeck 信息，不添加任何 deck dummy
                sample = sample.drop(columns=["CabinDeck"])
            else:
                deck_dummies = pd.get_dummies(sample["CabinDeck"], prefix="Deck")
                sample = pd.concat([sample, deck_dummies], axis=1)
                sample = sample.drop(columns=["CabinDeck"])
        
        # 编码 Title (one-hot)
        if "Title" in sample.columns:
            title_val = sample["Title"].iloc[0] if "Title" in sample.columns else None
            if pd.isna(title_val) or title_val == "" or title_val is None:
                sample = sample.drop(columns=["Title"])
            else:
                title_dummies = pd.get_dummies(sample["Title"], prefix="Title")
                sample = pd.concat([sample, title_dummies], axis=1)
                sample = sample.drop(columns=["Title"])
        
        # 确保所有特征列都存在
        for feat in available_features:
            if feat not in sample.columns:
                sample[feat] = 0
        
        sample = sample[available_features]
        
        # 预测
        if scaler:
            sample_scaled = scaler.transform(sample)
            prob = model.predict_proba(sample_scaled)[0]
        else:
            prob = model.predict_proba(sample)[0]
        
        survival_prob = round(float(prob[1]) * 100, 1)
        
        # 计算各特征的影响
        feature_impacts = []
        if model_type == "logistic_regression":
            importances = abs(model.coef_[0])
        else:
            importances = model.feature_importances_
        
        for i, feat in enumerate(available_features):
            val = float(sample.iloc[0, i]) if feat in sample.columns else 0
            impact = round(float(importances[i] * val) * 100, 1)
            feature_impacts.append({"feature": feat, "impact": impact})
        
        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_impacts = feature_impacts[:5]
        
        return {
            "survivalProbability": survival_prob,
            "topImpacts": top_impacts,
            "prediction": "survived" if survival_prob > 50 else "died",
        }
        
    except Exception as e:
        return {"error": str(e)}


def preview_data(config):
    """Return a preview of the raw dataset for display."""
    rows = config.get('rows', 10)
    page = config.get('page', 0)
    df = load_data()
    total = len(df)
    start = page * rows
    end = min(start + rows, total)
    # Convert to list of dicts with string values for JSON safety
    data = df.iloc[start:end].fillna('').to_dict(orient='records')
    # Convert numpy types
    for row in data:
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = float(v)
    return {
        'total': total,
        'page': page,
        'rows': len(data),
        'columns': list(df.columns),
        'data': data
    }

def clean_data_only(config):
    """执行数据清洗并返回清洗后的统计信息（不训练模型）"""
    df = load_data()
    df = clean_data(df, config)
    features = config.get("selectedFeatures", ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])
    available_features = [f for f in features if f in df.columns]
    has_nan = any(df[f].isna().any() for f in available_features if f in df.columns)
    
    return {
        "rows": len(df),
        "features": available_features,
        "missingAfterClean": {col: int(df[col].isna().sum()) for col in available_features if col in df.columns},
        "hasUnfilledMissing": has_nan,
        "sampleData": df[available_features].head(5).to_dict(orient="records") if len(available_features) > 0 else []
    }


# ── 统一入口（CLI 与 Vercel Python Functions 共用）──

def run_train_collect(config):
    """执行训练，按序收集全部事件（log/lc_progress/learning_curve/model_structure/result）"""
    events = []
    run_train(config, events.append)
    return events


def run_command(command, config):
    """执行非 train 命令，返回结果 dict；未知命令返回 {error: ...}"""
    if command == "info":
        return get_data_info()
    if command == "explore":
        return get_explore_data()
    if command == "clean":
        return clean_data_only(config)
    if command == "preview":
        return preview_data(config)
    if command == "predict":
        return predict(config)
    return {"error": f"Unknown command: {command}"}


# ── 兜底入口 ──
# Vercel 可能把 api/ 下任意 .py 注册为函数入口；本文件是共享模块而非 API，
# 定义 404 handler 仅用于避免 "no handler" 构建错误，正常不会被请求到。
from http.server import BaseHTTPRequestHandler  # noqa: E402


class _FallbackHandler(BaseHTTPRequestHandler):
    def _not_found(self):
        body = b'{"error": "not found"}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._not_found()

    def do_POST(self):
        self._not_found()

    def log_message(self, format, *args):  # noqa: A002
        pass


handler = _FallbackHandler
