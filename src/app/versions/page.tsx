'use client';

import { useState, useRef } from 'react';
import Link from 'next/link';
import { useML, type ModelVersion } from '@/lib/MLContext';
import { Pencil, Download, Trash2, Upload, Inbox, Eye, X } from 'lucide-react';

function getModelTypeLabel(type: string) {
  switch (type) {
    case 'decision_tree': return '决策树';
    case 'random_forest': return '随机森林';
    case 'logistic_regression': return '逻辑回归';
    default: return type;
  }
}

function formatDate(iso: string) {
  const d = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

interface DeleteModalState {
  open: boolean;
  version: ModelVersion | null;
}

interface DetailModalState {
  open: boolean;
  version: ModelVersion | null;
}

function getMissingStrategyLabel(strategy: string) {
  switch (strategy) {
    case 'mean': return '均值填充';
    case 'median': return '中位数填充';
    case 'mode': return '众数填充';
    case 'drop': return '删除';
    case 'deck': return '提取甲板层';
    case 'extract_deck': return '提取甲板层';
    case 'keep': return '保留';
    case 'none': return '不填充';
    default: return strategy;
  }
}

function getNameStrategyLabel(strategy: string) {
  switch (strategy) {
    case 'extract_title': return '提取头衔';
    case 'drop': return '删除';
    default: return strategy;
  }
}

function getHyperparamLabel(key: string) {
  const labels: Record<string, string> = {
    C: '正则化强度 C',
    maxIter: '最大迭代次数',
    maxDepth: '最大深度',
    minSamplesSplit: '最小分裂样本数',
    minSamplesLeaf: '叶节点最少样本数',
    nEstimators: '树的数量',
    solver: '优化算法',
    penalty: '正则化类型',
    criterion: '分裂标准',
  };
  return labels[key] || key;
}

function formatHyperparamValue(key: string, value: number | string | null | undefined) {
  if (value === null || value === undefined) return '不限 (None)';
  if (key === 'C') return typeof value === 'number' ? value.toFixed(2) : String(value);
  return String(value);
}

export default function VersionsPage() {
  const { versions, deleteVersion, renameVersion, importVersion } = useML();
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState('');
  const [deleteModal, setDeleteModal] = useState<DeleteModalState>({ open: false, version: null });
  const [detailModal, setDetailModal] = useState<DetailModalState>({ open: false, version: null });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleStartRename = (v: ModelVersion) => {
    setEditingId(v.versionId);
    setEditName(v.name);
  };

  const handleSaveRename = () => {
    if (editingId !== null) {
      const trimmed = editName.trim();
      if (trimmed) {
        renameVersion(editingId, trimmed);
      }
      setEditingId(null);
    }
  };

  const handleCancelRename = () => {
    setEditingId(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSaveRename();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      handleCancelRename();
    }
  };

  const handleExport = (v: ModelVersion) => {
    const exportData = {
      format: 'titanic-ml-model',
      version: 1,
      data: v,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${v.name.replace(/\s+/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const parsed = JSON.parse(ev.target?.result as string);
        // Support both wrapped format and raw ModelVersion
        const data = parsed.data || parsed;
        if (data.trainConfig && data.trainResult) {
          importVersion(data as ModelVersion);
        } else {
          alert('导入失败：文件格式不正确');
        }
      } catch {
        alert('导入失败：无法解析文件');
      }
    };
    reader.readAsText(file);
    // Reset input so same file can be re-imported
    e.target.value = '';
  };

  const handleDeleteConfirm = () => {
    if (deleteModal.version) {
      deleteVersion(deleteModal.version.versionId);
    }
    setDeleteModal({ open: false, version: null });
  };

  return (
    <div className="space-y-6 p-5 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-foreground">版本管理</h1>
        <div className="flex items-center gap-3">
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            onClick={handleImport}
            className="btn-gradient px-4 py-2 rounded-lg text-sm font-medium hover:opacity-90 active:scale-[0.98] transition-all inline-flex items-center gap-2"
          >
            <Upload className="w-3.5 h-3.5" />
            导入
          </button>
        </div>
      </div>

      {/* Version List */}
      {versions.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-5">
            <Inbox className="w-8 h-8 text-muted-foreground" />
          </div>
          <p className="text-muted-foreground text-sm mb-2">还没有训练过模型</p>
        </div>
      ) : (
        <div className="space-y-4">
          {versions.map(v => (
            <div key={v.versionId} className="glass-card p-5">
              {/* Card Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  {/* Version name - display or edit mode */}
                  {editingId === v.versionId ? (
                    <input
                      type="text"
                      value={editName}
                      onChange={e => setEditName(e.target.value)}
                      onBlur={handleSaveRename}
                      onKeyDown={handleKeyDown}
                      autoFocus
                      className="bg-muted border-none rounded-md px-3 py-1 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 transition-colors w-36"
                    />
                  ) : (
                    <div className="flex items-center gap-2">
                      <span className="text-base font-semibold text-foreground">{v.name}</span>
                      <button
                        onClick={() => handleStartRename(v)}
                        className="text-muted-foreground hover:text-primary transition-colors p-0.5"
                        title="重命名"
                      >
                        <Pencil className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                  {/* Model type badge */}
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/20 text-primary">
                    {getModelTypeLabel(v.trainConfig.modelType)}
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  {/* Created time */}
                  <span className="text-xs text-muted-foreground">{formatDate(v.createdAt)}</span>
                  {/* Action buttons */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setDetailModal({ open: true, version: v })}
                      className="text-muted-foreground hover:text-primary transition-colors p-1"
                      title="查看详情"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleExport(v)}
                      className="text-muted-foreground hover:text-accent transition-colors p-1"
                      title="导出"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setDeleteModal({ open: true, version: v })}
                      className="text-muted-foreground hover:text-destructive transition-colors p-1"
                      title="删除"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Performance preview - 4 metrics */}
              <div className="grid grid-cols-4 gap-3">
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-xs text-muted-foreground mb-1">准确率</div>
                  <div className="text-lg font-bold text-primary">{(v.trainResult.metrics.accuracy * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-xs text-muted-foreground mb-1">精确率</div>
                  <div className="text-lg font-bold text-accent">{(v.trainResult.metrics.precision * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-xs text-muted-foreground mb-1">召回率</div>
                  <div className="text-lg font-bold text-success">{(v.trainResult.metrics.recall * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-xs text-muted-foreground mb-1">F1分数</div>
                  <div className="text-lg font-bold text-foreground">{(v.trainResult.metrics.f1 * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteModal.open && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center p-4 z-50">
          <div className="bg-card rounded-xl shadow-[0_25px_50px_rgba(0,0,0,0.5)] max-w-sm w-full p-6">
            <h3 className="text-base font-semibold text-foreground mb-2">确认删除</h3>
            <p className="text-sm text-muted-foreground mb-1">
              确定要删除版本「<span className="text-foreground font-medium">{deleteModal.version?.name}</span>」吗？
            </p>
            <p className="text-sm text-destructive/80 mb-6">删除后无法恢复</p>
            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setDeleteModal({ open: false, version: null })}
                className="bg-muted text-foreground border-none px-4 py-2 rounded-lg text-sm font-medium hover:bg-muted/80 active:scale-[0.98] transition-all"
              >
                取消
              </button>
              <button
                onClick={handleDeleteConfirm}
                className="bg-destructive text-white px-4 py-2 rounded-lg text-sm font-medium hover:opacity-90 active:scale-[0.98] transition-all"
              >
                确认删除
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Version Detail Modal */}
      {detailModal.open && detailModal.version && (() => {
        const v = detailModal.version;
        const cleanCfg = v.trainConfig.cleanConfig;
        const hp = v.trainConfig.hyperparams;
        const res = v.trainResult;
        const cm = res.confusionMatrix;
        return (
          <div className="fixed inset-0 bg-black/60 flex items-center justify-center p-4 z-50" onClick={() => setDetailModal({ open: false, version: null })}>
            <div className="bg-card rounded-xl shadow-[0_25px_50px_rgba(0,0,0,0.5)] max-w-2xl w-full max-h-[85vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              {/* Header */}
              <div className="flex items-center justify-between p-5 border-b border-border/50 sticky top-0 bg-card rounded-t-xl z-10">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-semibold text-foreground">{v.name}</h3>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/20 text-primary">
                    {getModelTypeLabel(v.trainConfig.modelType)}
                  </span>
                </div>
                <button onClick={() => setDetailModal({ open: false, version: null })} className="text-muted-foreground hover:text-foreground transition-colors p-1">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="p-5 space-y-5">
                {/* Section 1: Data Cleaning Config */}
                <div>
                  <h4 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                    <span className="w-1 h-4 bg-primary rounded-full" />
                    数据清洗配置
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">Age 缺失值策略</div>
                      <div className="text-sm text-foreground font-medium">{getMissingStrategyLabel(cleanCfg.missingValueStrategy.Age)}</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">Embarked 缺失值策略</div>
                      <div className="text-sm text-foreground font-medium">{getMissingStrategyLabel(cleanCfg.missingValueStrategy.Embarked)}</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">Cabin 缺失值策略</div>
                      <div className="text-sm text-foreground font-medium">{getMissingStrategyLabel(cleanCfg.missingValueStrategy.Cabin)}</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">姓名处理策略</div>
                      <div className="text-sm text-foreground font-medium">{getNameStrategyLabel(cleanCfg.nameStrategy)}</div>
                    </div>
                  </div>
                  <div className="mt-3">
                    <div className="text-xs text-muted-foreground mb-2">选中特征 ({cleanCfg.selectedFeatures.length})</div>
                    <div className="flex flex-wrap gap-1.5">
                      {cleanCfg.selectedFeatures.map(f => (
                        <span key={f} className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-primary/10 text-primary/80 border border-primary/20">{f}</span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Section 2: Hyperparameters */}
                <div>
                  <h4 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                    <span className="w-1 h-4 bg-accent rounded-full" />
                    训练超参数
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {Object.entries(hp).map(([key, val]) => (
                      <div key={key} className="bg-muted/50 rounded-lg p-3">
                        <div className="text-xs text-muted-foreground mb-1">{getHyperparamLabel(key)}</div>
                        <div className="text-sm text-foreground font-medium font-mono">{formatHyperparamValue(key, val as number | string | null | undefined)}</div>
                      </div>
                    ))}
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">测试集比例</div>
                      <div className="text-sm text-foreground font-medium font-mono">{(v.trainConfig.testSize * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                </div>

                {/* Section 3: Training Results Summary */}
                <div>
                  <h4 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                    <span className="w-1 h-4 bg-success rounded-full" />
                    训练结果摘要
                  </h4>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                    <div className="bg-muted/50 rounded-lg p-3 text-center">
                      <div className="text-xs text-muted-foreground mb-1">准确率</div>
                      <div className="text-lg font-bold text-primary">{(res.metrics.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3 text-center">
                      <div className="text-xs text-muted-foreground mb-1">精确率</div>
                      <div className="text-lg font-bold text-accent">{(res.metrics.precision * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3 text-center">
                      <div className="text-xs text-muted-foreground mb-1">召回率</div>
                      <div className="text-lg font-bold text-success">{(res.metrics.recall * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3 text-center">
                      <div className="text-xs text-muted-foreground mb-1">F1分数</div>
                      <div className="text-lg font-bold text-foreground">{(res.metrics.f1 * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                  {/* Confusion Matrix */}
                  <div className="bg-muted/50 rounded-lg p-3">
                    <div className="text-xs text-muted-foreground mb-2">混淆矩阵</div>
                    <div className="grid grid-cols-3 gap-1 text-center text-xs max-w-[240px]">
                      <div />
                      <div className="text-muted-foreground font-medium py-1">预测生还</div>
                      <div className="text-muted-foreground font-medium py-1">预测遇难</div>
                      <div className="text-muted-foreground font-medium flex items-center justify-end pr-2">实际生还</div>
                      <div className="bg-success/20 text-success rounded p-1.5 font-mono font-medium">{cm.tp}</div>
                      <div className="bg-destructive/15 text-destructive/80 rounded p-1.5 font-mono font-medium">{cm.fn}</div>
                      <div className="text-muted-foreground font-medium flex items-center justify-end pr-2">实际遇难</div>
                      <div className="bg-destructive/15 text-destructive/80 rounded p-1.5 font-mono font-medium">{cm.fp}</div>
                      <div className="bg-success/20 text-success rounded p-1.5 font-mono font-medium">{cm.tn}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
