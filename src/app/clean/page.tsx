'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import Link from 'next/link';
import { useML } from '@/lib/MLContext';

type AgeStrategy = 'mean' | 'median' | 'mode' | 'none';
type EmbarkedStrategy = 'mode' | 'drop' | 'none';
type CabinStrategy = 'drop' | 'extract_deck';
type NameStrategy = 'drop' | 'extract_title';

interface MissingValueStrategy {
  Age: AgeStrategy;
  Embarked: EmbarkedStrategy;
  Cabin: CabinStrategy;
}

interface FeatureDef {
  key: string;
  label: string;
  desc: string;
  recommended: boolean;
  disabled: boolean; // always disabled regardless of config
  conditional?: boolean; // only available under certain config
  conditionMet?: boolean; // computed: whether the condition is met
  importance: string;
}

const BASE_FEATURES: FeatureDef[] = [
  { key: 'PassengerId', label: 'PassengerId', desc: '乘客编号', recommended: false, disabled: true, importance: '干扰 - 无预测价值' },
  { key: 'Pclass', label: 'Pclass', desc: '船舱等级', recommended: true, disabled: false, importance: '高 - 社会阶层显著影响生存' },
  { key: 'Sex', label: 'Sex', desc: '性别', recommended: true, disabled: false, importance: '极高 - 性别是最强预测因子' },
  { key: 'Age', label: 'Age', desc: '年龄', recommended: true, disabled: false, importance: '中 - 儿童优先原则' },
  { key: 'SibSp', label: 'SibSp', desc: '兄弟姐妹/配偶数', recommended: false, disabled: false, importance: '低 - 家庭规模有微弱影响' },
  { key: 'Parch', label: 'Parch', desc: '父母/子女数', recommended: false, disabled: false, importance: '低 - 家庭规模有微弱影响' },
  { key: 'Fare', label: 'Fare', desc: '票价', recommended: true, disabled: false, importance: '中 - 与船舱等级相关' },
  { key: 'Embarked', label: 'Embarked', desc: '登船港口', recommended: false, disabled: false, importance: '低 - 登船港口有微弱影响' },
  { key: 'Ticket', label: 'Ticket', desc: '船票号码', recommended: false, disabled: true, importance: '干扰 - 字符串无法编码' },
  { key: 'Name', label: 'Name', desc: '姓名（含头衔）', recommended: false, disabled: true, importance: '见Title - 头衔反映社会地位' },
  { key: 'Cabin', label: 'Cabin', desc: '船舱位置', recommended: false, disabled: true, importance: '见CabinDeck - 提取甲板层信息' },
  // Derived features (conditional)
  { key: 'Title', label: 'Title', desc: '头衔（从Name提取）', recommended: false, disabled: false, conditional: true, importance: '中 - Mr/Mrs/Miss/Master/Rare反映社会地位与性别' },
  { key: 'CabinDeck', label: 'CabinDeck', desc: '甲板层（从Cabin提取）', recommended: false, disabled: false, conditional: true, importance: '低-中 - 甲板位置与生存相关，但缺失多' },
];

export default function CleanPage() {
  const { cleanConfig, updateCleanConfig, updateStepCompleted } = useML();
  const [missingStrategy, setMissingStrategy] = useState<MissingValueStrategy>({
    Age: 'median',
    Embarked: 'mode',
    Cabin: 'drop',
  });
  const [nameStrategy, setNameStrategy] = useState<NameStrategy>('drop');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(
    BASE_FEATURES.filter(f => f.recommended).map(f => f.key)
  );
  const [applied, setApplied] = useState(false);
  const [applying, setApplying] = useState(false);
  const [cleanResult, setCleanResult] = useState<{ rows: number; features: string[]; missingAfterClean: Record<string, number>; hasUnfilledMissing: boolean } | null>(null);

  // Derived: which conditional features are available
  const titleAvailable = nameStrategy === 'extract_title';
  const cabinDeckAvailable = missingStrategy.Cabin === 'extract_deck';

  // Computed features list based on current config
  const features = useMemo(() => {
    return BASE_FEATURES.map(f => ({
      ...f,
      conditionMet: f.key === 'Title' ? titleAvailable : f.key === 'CabinDeck' ? cabinDeckAvailable : true,
    }));
  }, [titleAvailable, cabinDeckAvailable]);

  // Auto-adjust selected features when config changes
  useEffect(() => {
    setSelectedFeatures(prev => {
      const validKeys = features.filter(f => !f.disabled && f.conditionMet).map(f => f.key);
      // Remove features that are no longer available
      const filtered = prev.filter(k => validKeys.includes(k));
      return filtered;
    });
  }, [features]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync with global state
  useEffect(() => {
    if (cleanConfig) {
      setMissingStrategy(cleanConfig.missingValueStrategy as MissingValueStrategy);
      setSelectedFeatures(cleanConfig.selectedFeatures);
      if (cleanConfig.nameStrategy) setNameStrategy(cleanConfig.nameStrategy as NameStrategy);
    }
  }, [cleanConfig]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleToggleFeature = (key: string) => {
    setSelectedFeatures(prev =>
      prev.includes(key) ? prev.filter(f => f !== key) : [...prev, key]
    );
    setApplied(false);
  };

  // Check if any selected feature has unfilled missing values
  const hasUnfilledMissing = useMemo(() => {
    if (!cleanResult?.hasUnfilledMissing) return false;
    // Check if any selected feature still has missing values
    const missingFeatures = cleanResult.missingAfterClean ? Object.keys(cleanResult.missingAfterClean).filter(k => (cleanResult.missingAfterClean as Record<string, number>)[k] > 0) : [];
    return selectedFeatures.some(f => missingFeatures.includes(f));
  }, [cleanResult, selectedFeatures]);

  const handleApply = useCallback(async () => {
    setApplying(true);
    const config = {
      missingValueStrategy: missingStrategy,
      selectedFeatures,
      nameStrategy,
    };

    try {
      const res = await fetch('/api/data/clean', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      const data = await res.json();
      if (data.rows) {
        setCleanResult(data);
        setApplied(true);
        updateCleanConfig({ ...config, hasUnfilledMissing: data.hasUnfilledMissing || false });
        updateStepCompleted('clean');
      }
    } catch (err) {
      console.error('Clean error:', err);
    } finally {
      setApplying(false);
    }
  }, [missingStrategy, selectedFeatures, nameStrategy, updateCleanConfig, updateStepCompleted]);

  const canProceed = applied;
  const noFeatureSelected = selectedFeatures.length === 0;

  // Feature groups for rendering
  const keptFeatures = features.filter(f => f.recommended && !f.disabled && f.conditionMet);
  const optionalFeatures = features.filter(f => !f.recommended && !f.disabled && f.conditionMet && !f.conditional);
  const conditionalFeatures = features.filter(f => f.conditional && f.conditionMet);
  const disabledFeatures = features.filter(f => f.disabled);

  return (
    <div className="space-y-8 p-5">
      <div>
        <h1 className="text-3xl font-bold text-foreground">数据清洗</h1>
        <p className="text-muted-foreground mt-2">处理缺失值，提取特征，选择用于建模的列</p>
      </div>

      {/* Missing Value Section */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">数值处理</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-4">
          {/* Age */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground">Age (年龄)</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-accent/20 text-accent">缺失 19.9%</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">年龄有约20%的缺失，需要选择处理策略</p>
            <select
              value={missingStrategy.Age}
              onChange={e => { setMissingStrategy(s => ({ ...s, Age: e.target.value as AgeStrategy })); setApplied(false); }}
              className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
            >
              <option value="mean">均值填充 (29.7)</option>
              <option value="median">中位数填充 (28.0)</option>
              <option value="mode">众数填充 (24.0)</option>
              <option value="none">保留缺失值（树模型可用）</option>
            </select>
            <p className="text-xs text-muted-foreground mt-2">
              {missingStrategy.Age === 'mean' && '均值受极端值影响较大，但保留了整体平均水平'}
              {missingStrategy.Age === 'median' && '中位数更稳健，不受极端值影响，推荐使用'}
              {missingStrategy.Age === 'mode' && '众数是最常见的年龄，可能偏低'}
              {missingStrategy.Age === 'none' && '保留缺失值，仅决策树和随机森林支持，逻辑回归不可用'}
            </p>
          </div>

          {/* Embarked */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground">Embarked (登船港)</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-success/20 text-success">缺失 0.2%</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">仅2条记录缺失，影响极小</p>
            <select
              value={missingStrategy.Embarked}
              onChange={e => { setMissingStrategy(s => ({ ...s, Embarked: e.target.value as EmbarkedStrategy })); setApplied(false); }}
              className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
            >
              <option value="mode">众数填充 (S-南安普顿)</option>
              <option value="drop">删除该行</option>
              <option value="none">保留缺失值（树模型可用）</option>
            </select>
          </div>

          {/* Cabin */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground">Cabin (舱位)</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-destructive/20 text-destructive">缺失 77.1%</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">缺失过多，但可提取甲板层信息</p>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="cabin" value="drop" checked={missingStrategy.Cabin === 'drop'}
                  onChange={() => { setMissingStrategy(s => ({ ...s, Cabin: 'drop' })); setApplied(false); }}
                  className="accent-primary" />
                <span className="text-sm text-foreground">丢弃该列</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="cabin" value="extract_deck" checked={missingStrategy.Cabin === 'extract_deck'}
                  onChange={() => { setMissingStrategy(s => ({ ...s, Cabin: 'extract_deck' })); setApplied(false); }}
                  className="accent-primary" />
                <span className="text-sm text-foreground">提取甲板层 (CabinDeck)</span>
              </label>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {missingStrategy.Cabin === 'extract_deck'
                ? '提取舱位首字母(A-G)作为甲板层，缺失值标记为U(未知)，可用作特征'
                : '完全丢弃Cabin列，无法获取甲板层信息'}
            </p>
          </div>

          {/* Name / Title */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground">Name (姓名)</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-primary/20 text-primary">完整</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">姓名含头衔(Mr/Mrs/Miss等)，反映社会地位</p>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="name" value="drop" checked={nameStrategy === 'drop'}
                  onChange={() => { setNameStrategy('drop'); setApplied(false); }}
                  className="accent-primary" />
                <span className="text-sm text-foreground">丢弃姓名</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="name" value="extract_title" checked={nameStrategy === 'extract_title'}
                  onChange={() => { setNameStrategy('extract_title'); setApplied(false); }}
                  className="accent-primary" />
                <span className="text-sm text-foreground">提取头衔 (Title)</span>
              </label>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {nameStrategy === 'extract_title'
                ? '提取为5类: Mr, Mrs, Miss, Master, Rare。头衔与性别和社会地位强相关'
                : '姓名对预测无直接帮助，丢弃'}
            </p>
          </div>
        </div>
      </section>

      {/* Feature Selection */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">
          特征选择 <span className="text-sm font-normal text-muted-foreground">已选 {selectedFeatures.length} 个</span>
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Recommended */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-medium text-success mb-3 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-success" /> 推荐保留
            </h3>
            <div className="space-y-2">
              {keptFeatures.map(f => (
                <label key={f.key} className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/30 cursor-pointer">
                  <input type="checkbox" checked={selectedFeatures.includes(f.key)}
                    onChange={() => handleToggleFeature(f.key)} className="accent-primary w-4 h-4" />
                  <span className="text-sm font-medium text-foreground">{f.key}</span>
                  <span className="text-xs text-muted-foreground">{f.desc}</span>
                  <span className="text-xs text-primary/60 ml-auto">{f.importance.split(' - ')[0]}</span>
                </label>
              ))}
            </div>
          </div>
          {/* Optional */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-medium text-warning mb-3 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-warning" /> 可选特征
            </h3>
            <div className="space-y-2">
              {optionalFeatures.map(f => (
                <label key={f.key} className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/30 cursor-pointer">
                  <input type="checkbox" checked={selectedFeatures.includes(f.key)}
                    onChange={() => handleToggleFeature(f.key)} className="accent-primary w-4 h-4" />
                  <span className="text-sm font-medium text-foreground">{f.key}</span>
                  <span className="text-xs text-muted-foreground">{f.desc}</span>
                  <span className="text-xs text-primary/60 ml-auto">{f.importance.split(' - ')[0]}</span>
                </label>
              ))}
              {conditionalFeatures.length > 0 && conditionalFeatures.map(f => (
                <label key={f.key} className="flex items-center gap-3 p-2 rounded-lg hover:bg-primary/10 cursor-pointer border border-primary/20">
                  <input type="checkbox" checked={selectedFeatures.includes(f.key)}
                    onChange={() => handleToggleFeature(f.key)} className="accent-primary w-4 h-4" />
                  <span className="text-sm font-medium text-primary">{f.key}</span>
                  <span className="text-xs text-muted-foreground">{f.desc}</span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-primary/20 text-primary ml-1">派生</span>
                </label>
              ))}
              {disabledFeatures.length > 0 && disabledFeatures.map(f => (
                <div key={f.key} className="flex items-center gap-3 p-2 rounded-lg opacity-50">
                  <input type="checkbox" checked={false} disabled className="accent-primary w-4 h-4" />
                  <span className="text-sm text-muted-foreground">{f.key}</span>
                  <span className="text-xs text-muted-foreground">{f.desc}</span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-destructive/20 text-destructive/70 ml-1">禁用</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Missing Value Warning */}
      {hasUnfilledMissing && (
        <div className="glass-card p-5 border border-warning/30">
          <h3 className="text-warning font-medium mb-2">存在未填充的缺失值</h3>
          <p className="text-sm text-muted-foreground">
            以下特征仍有缺失值：{Object.entries(cleanResult?.missingAfterClean || {})
              .filter(([k, v]) => (v as number) > 0 && selectedFeatures.includes(k))
              .map(([k]) => k).join(', ')}
            。逻辑回归不支持缺失值，仅可选择决策树或随机森林进行训练。
          </p>
        </div>
      )}

      {/* Clean Result */}
      {cleanResult && (
        <div className="glass-card p-5 border border-success/30">
          <h3 className="text-success font-medium mb-2">清洗完成</h3>
          <p className="text-sm text-muted-foreground">
            剩余 <span className="text-foreground font-medium">{cleanResult.rows}</span> 条数据，
            <span className="text-foreground font-medium">{cleanResult.features.length}</span> 个特征
            ({cleanResult.features.join(', ')})
            {cleanResult.hasUnfilledMissing && (
              <span className="text-warning ml-2">(含未填充缺失值，逻辑回归不可用)</span>
            )}
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-between items-center">
        <Link href="/explore" className="px-6 py-3 rounded-xl text-sm font-medium border border-border text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all">
          ← 返回数据探索
        </Link>
        <div className="flex gap-3">
          <button onClick={handleApply} disabled={applying || noFeatureSelected}
            className="btn-gradient px-6 py-3 rounded-xl text-sm font-medium disabled:opacity-50">
            {applying ? '应用中...' : applied ? '已应用 ✓' : '应用清洗配置'}
          </button>
          <Link href="/train"
            className={`px-6 py-3 rounded-xl text-sm font-medium transition-all ${
              canProceed
                ? 'btn-gradient'
                : 'bg-muted text-muted-foreground cursor-not-allowed pointer-events-none'
            }`}>
            下一步：模型训练 →
          </Link>
        </div>
      </div>
    </div>
  );
}
