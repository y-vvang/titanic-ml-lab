'use client';

import { useState, useRef } from 'react';
import Link from 'next/link';
import { useI18n } from '@/lib/i18n';
import { useML, type ModelVersion } from '@/lib/MLContext';
import { Pencil, Download, Trash2, Upload, Inbox, Eye, X } from 'lucide-react';

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

export default function VersionsPage() {
  const { t } = useI18n();
  const { versions, deleteVersion, renameVersion, importVersion } = useML();

  const getModelTypeLabel = (type: string) =>
    lookupLabel(t, `versions.modelType.${type}`, type);
  const getMissingStrategyLabel = (strategy: string) =>
    lookupLabel(t, `versions.missingStrategy.${strategy}`, strategy);
  const getNameStrategyLabel = (strategy: string) =>
    lookupLabel(t, `versions.nameStrategy.${strategy}`, strategy);
  const getHyperparamLabel = (key: string) =>
    lookupLabel(t, `versions.hyperparam.${key}`, key);

  function formatHyperparamValue(key: string, value: number | string | null | undefined) {
    if (value === null || value === undefined) return t('versions.unlimited');
    if (key === 'C') return typeof value === 'number' ? value.toFixed(2) : String(value);
    return String(value);
  }

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
          alert(t('versions.importBadFormat'));
        }
      } catch {
        alert(t('versions.importUnparsable'));
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
        <h1 className="text-3xl font-bold text-foreground">{t('versions.title')}</h1>
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
            {t('versions.import')}
          </button>
        </div>
      </div>

      {/* Version List */}
      {versions.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-5">
            <Inbox className="w-8 h-8 text-muted-foreground" />
          </div>
          <p className="text-muted-foreground text-sm mb-2">{t('versions.empty')}</p>
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
                        title={t('versions.rename')}
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
                      title={t('versions.viewDetail')}
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleExport(v)}
                      className="text-muted-foreground hover:text-accent transition-colors p-1"
                      title={t('versions.export')}
                    >
                      <Download className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setDeleteModal({ open: true, version: v })}
                      className="text-muted-foreground hover:text-destructive transition-colors p-1"
                      title={t('versions.del')}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Performance preview - 4 metrics */}
              <div className="grid grid-cols-4 gap-3">
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-xs text-muted-foreground mb-1">{t('evaluate.metricAccuracy')}</div>
                  <div className="text-lg font-bold text-primary">{(v.trainResult.metrics.accuracy * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-xs text-muted-foreground mb-1">{t('evaluate.metricPrecision')}</div>
                  <div className="text-lg font-bold text-accent">{(v.trainResult.metrics.precision * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-xs text-muted-foreground mb-1">{t('evaluate.metricRecall')}</div>
                  <div className="text-lg font-bold text-success">{(v.trainResult.metrics.recall * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <div className="text-xs text-muted-foreground mb-1">{t('evaluate.metricF1')}</div>
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
            <h3 className="text-base font-semibold text-foreground mb-2">{t('versions.deleteTitle')}</h3>
            <p className="text-sm text-muted-foreground mb-1">
              {t('versions.deleteConfirmStart')}<span className="text-foreground font-medium">{deleteModal.version?.name}</span>{t('versions.deleteConfirmEnd')}
            </p>
            <p className="text-sm text-destructive/80 mb-6">{t('versions.deleteIrreversible')}</p>
            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setDeleteModal({ open: false, version: null })}
                className="bg-muted text-foreground border-none px-4 py-2 rounded-lg text-sm font-medium hover:bg-muted/80 active:scale-[0.98] transition-all"
              >
                {t('versions.cancel')}
              </button>
              <button
                onClick={handleDeleteConfirm}
                className="bg-destructive text-white px-4 py-2 rounded-lg text-sm font-medium hover:opacity-90 active:scale-[0.98] transition-all"
              >
                {t('versions.confirmDelete')}
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
                    {t('versions.cleanConfigSection')}
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">{t('versions.ageMissingStrategy')}</div>
                      <div className="text-sm text-foreground font-medium">{getMissingStrategyLabel(cleanCfg.missingValueStrategy.Age)}</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">{t('versions.embarkedMissingStrategy')}</div>
                      <div className="text-sm text-foreground font-medium">{getMissingStrategyLabel(cleanCfg.missingValueStrategy.Embarked)}</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">{t('versions.cabinMissingStrategy')}</div>
                      <div className="text-sm text-foreground font-medium">{getMissingStrategyLabel(cleanCfg.missingValueStrategy.Cabin)}</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">{t('versions.nameHandlingStrategy')}</div>
                      <div className="text-sm text-foreground font-medium">{getNameStrategyLabel(cleanCfg.nameStrategy)}</div>
                    </div>
                  </div>
                  <div className="mt-3">
                    <div className="text-xs text-muted-foreground mb-2">{t('versions.selectedFeatures', { n: cleanCfg.selectedFeatures.length })}</div>
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
                    {t('versions.hyperparamsSection')}
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {Object.entries(hp).map(([key, val]) => (
                      <div key={key} className="bg-muted/50 rounded-lg p-3">
                        <div className="text-xs text-muted-foreground mb-1">{getHyperparamLabel(key)}</div>
                        <div className="text-sm text-foreground font-medium font-mono">{formatHyperparamValue(key, val as number | string | null | undefined)}</div>
                      </div>
                    ))}
                    <div className="bg-muted/50 rounded-lg p-3">
                      <div className="text-xs text-muted-foreground mb-1">{t('versions.testSize')}</div>
                      <div className="text-sm text-foreground font-medium font-mono">{(v.trainConfig.testSize * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                </div>

                {/* Section 3: Training Results Summary */}
                <div>
                  <h4 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                    <span className="w-1 h-4 bg-success rounded-full" />
                    {t('versions.resultSummarySection')}
                  </h4>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                    <div className="bg-muted/50 rounded-lg p-3 text-center">
                      <div className="text-xs text-muted-foreground mb-1">{t('evaluate.metricAccuracy')}</div>
                      <div className="text-lg font-bold text-primary">{(res.metrics.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3 text-center">
                      <div className="text-xs text-muted-foreground mb-1">{t('evaluate.metricPrecision')}</div>
                      <div className="text-lg font-bold text-accent">{(res.metrics.precision * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3 text-center">
                      <div className="text-xs text-muted-foreground mb-1">{t('evaluate.metricRecall')}</div>
                      <div className="text-lg font-bold text-success">{(res.metrics.recall * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-3 text-center">
                      <div className="text-xs text-muted-foreground mb-1">{t('evaluate.metricF1')}</div>
                      <div className="text-lg font-bold text-foreground">{(res.metrics.f1 * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                  {/* Confusion Matrix */}
                  <div className="bg-muted/50 rounded-lg p-3">
                    <div className="text-xs text-muted-foreground mb-2">{t('evaluate.cmTitle')}</div>
                    <div className="grid grid-cols-3 gap-1 text-center text-xs max-w-[240px]">
                      <div />
                      <div className="text-muted-foreground font-medium py-1">{t('versions.cmPredictSurvived')}</div>
                      <div className="text-muted-foreground font-medium py-1">{t('versions.cmPredictDied')}</div>
                      <div className="text-muted-foreground font-medium flex items-center justify-end pr-2">{t('versions.cmActualSurvived')}</div>
                      <div className="bg-success/20 text-success rounded p-1.5 font-mono font-medium">{cm.tp}</div>
                      <div className="bg-destructive/15 text-destructive/80 rounded p-1.5 font-mono font-medium">{cm.fn}</div>
                      <div className="text-muted-foreground font-medium flex items-center justify-end pr-2">{t('versions.cmActualDied')}</div>
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

/**
 * t() falls back to the key itself when a lookup misses, which is wrong for
 * dynamic keys (e.g. `versions.modelType.${type}` from stored version data) —
 * we want the raw value instead, matching the old switch-defaults.
 */
function lookupLabel(t: (key: string) => string, key: string, fallback: string): string {
  const s = t(key);
  return s === key ? fallback : s;
}
