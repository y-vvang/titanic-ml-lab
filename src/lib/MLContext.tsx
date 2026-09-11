'use client';

import React, { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';
import { useI18n } from '@/lib/i18n';

export interface CleanConfig {
  missingValueStrategy: {
    Age: string;
    Embarked: string;
    Cabin: string;
  };
  selectedFeatures: string[];
  nameStrategy: string;
  hasUnfilledMissing?: boolean;
}

export interface TrainConfig {
  cleanConfig: CleanConfig;
  modelType: 'logistic_regression' | 'decision_tree' | 'random_forest';
  hyperparams: Record<string, number | string | null>;
  testSize: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function isTrainConfig(obj: any): obj is TrainConfig {
  return obj && typeof obj.modelType === 'string' && typeof obj.testSize === 'number';
}

export interface TrainResult {
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  };
  confusionMatrix: {
    tn: number;
    fp: number;
    fn: number;
    tp: number;
  };
  featureImportance: { feature: string; importance: number }[];
  groupAccuracy: {
    bySex?: Record<string, number>;
    byPclass?: Record<string, number>;
    byAgeGroup?: Record<string, number>;
    byEmbarked?: Record<string, number>;
    byCabinDeck?: Record<string, number>;
    byTitle?: Record<string, number>;
    bySibSp?: Record<string, number>;
    byParch?: Record<string, number>;
    byFare?: Record<string, number>;
  };
  testSize: number;
  hasUnfilledMissing?: boolean;
  modelInfo: {
    modelType: string;
    availableFeatures: string[];
    scalerUsed: boolean;
  };
}

export interface ModelVersion {
  versionId: number;
  name: string;
  createdAt: string;
  trainConfig: TrainConfig;
  trainResult: TrainResult;
  /** @deprecated Use trainConfig.modelType === 'logistic_regression' instead. Kept for backward compat with imported versions. */
  isLogisticRegression?: boolean;
}

export type StepStatus = 'pending' | 'active' | 'done';

type StepName = 'explore' | 'clean' | 'train' | 'evaluate' | 'predict';

const VERSIONS_STORAGE_KEY = 'titanic-ml-versions';
const ACTIVE_VERSION_KEY = 'titanic-ml-active-version';

interface MLContextType {
  // Step tracking
  stepCompleted: Record<StepName, boolean>;
  updateStepCompleted: (step: StepName) => void;

  // Clean config
  cleanConfig: CleanConfig | null;
  updateCleanConfig: (config: CleanConfig) => void;

  // Version management
  versions: ModelVersion[];
  activeVersionId: number | null;
  addVersion: (config: TrainConfig, result: TrainResult) => void;
  deleteVersion: (versionId: number) => void;
  renameVersion: (versionId: number, name: string) => void;
  setActiveVersion: (versionId: number) => void;
  importVersion: (data: ModelVersion) => void;
  getActiveVersion: () => ModelVersion | null;

  // Convenience accessors (derived from active version)
  trainConfig: TrainConfig | null;
  trainResult: TrainResult | null;
}

const MLContext = createContext<MLContextType | null>(null);

function loadVersions(): ModelVersion[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(VERSIONS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveVersions(versions: ModelVersion[]) {
  if (typeof window === 'undefined') return;
  localStorage.setItem(VERSIONS_STORAGE_KEY, JSON.stringify(versions));
}

function loadActiveVersionId(): number | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(ACTIVE_VERSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return typeof parsed === 'number' ? parsed : null;
  } catch {
    return null;
  }
}

function saveActiveVersionId(id: number | null) {
  if (typeof window === 'undefined') return;
  if (id === null) {
    localStorage.removeItem(ACTIVE_VERSION_KEY);
  } else {
    localStorage.setItem(ACTIVE_VERSION_KEY, String(id));
  }
}

export function MLProvider({ children }: { children: ReactNode }) {
  const { t } = useI18n();
  const [stepCompleted, setStepCompleted] = useState<Record<StepName, boolean>>({
    explore: false,
    clean: false,
    train: false,
    evaluate: false,
    predict: false,
  });
  const [cleanConfig, setCleanConfig] = useState<CleanConfig | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [activeVersionId, setActiveVersionIdState] = useState<number | null>(null);

  // Load from localStorage on mount
  useEffect(() => {
    const savedVersions = loadVersions();
    const savedActiveId = loadActiveVersionId();
    setVersions(savedVersions);
    setActiveVersionIdState(savedActiveId);

    // If there are versions but no active id, set the latest as active
    if (savedVersions.length > 0 && savedActiveId === null) {
      const latestId = savedVersions[savedVersions.length - 1].versionId;
      setActiveVersionIdState(latestId);
      saveActiveVersionId(latestId);
    }
  }, []);

  const updateStepCompleted = useCallback((step: StepName) => {
    setStepCompleted(prev => ({ ...prev, [step]: true }));
  }, []);

  const updateCleanConfig = useCallback((config: CleanConfig) => {
    setCleanConfig(config);
    setStepCompleted(prev => ({ ...prev, clean: true }));
  }, []);

  const addVersion = useCallback((config: TrainConfig, result: TrainResult) => {
    setVersions(prev => {
      const maxId = prev.length > 0 ? Math.max(...prev.map(v => v.versionId)) : 0;
      // Strip redundant fields before storing
      const { trainSize: _ts, learningCurve: _lc, ...slimResult } = result as TrainResult & { trainSize?: number; learningCurve?: unknown };
      void _ts; void _lc;
      const newVersion: ModelVersion = {
        versionId: maxId + 1,
        name: t('common.defaultVersionName', { n: maxId + 1 }),
        createdAt: new Date().toISOString(),
        trainConfig: config,
        trainResult: slimResult as TrainResult,
      };
      const updated = [...prev, newVersion];
      saveVersions(updated);
      setActiveVersionIdState(newVersion.versionId);
      saveActiveVersionId(newVersion.versionId);
      return updated;
    });
    setStepCompleted(prev => ({ ...prev, train: true, evaluate: true }));
  }, [t]);

  const deleteVersion = useCallback((versionId: number) => {
    setVersions(prev => {
      const updated = prev.filter(v => v.versionId !== versionId);
      saveVersions(updated);
      return updated;
    });
    setActiveVersionIdState(prev => {
      if (prev === versionId) {
        // Switch to the latest remaining version, or null
        const remaining = versions.filter(v => v.versionId !== versionId);
        const newActive = remaining.length > 0 ? remaining[remaining.length - 1].versionId : null;
        saveActiveVersionId(newActive);
        return newActive;
      }
      return prev;
    });
  }, [versions]);

  const renameVersion = useCallback((versionId: number, name: string) => {
    setVersions(prev => {
      const updated = prev.map(v => v.versionId === versionId ? { ...v, name } : v);
      saveVersions(updated);
      return updated;
    });
  }, []);

  const setActiveVersion = useCallback((versionId: number) => {
    setActiveVersionIdState(versionId);
    saveActiveVersionId(versionId);
    setStepCompleted(prev => ({ ...prev, train: true, evaluate: true }));
  }, []);

  const importVersion = useCallback((data: ModelVersion) => {
    setVersions(prev => {
      const maxId = prev.length > 0 ? Math.max(...prev.map(v => v.versionId)) : 0;
      const newVersion: ModelVersion = {
        ...data,
        versionId: maxId + 1,
        name: data.name || t('common.defaultVersionName', { n: maxId + 1 }),
      };
      const updated = [...prev, newVersion];
      saveVersions(updated);
      return updated;
    });
  }, [t]);

  const getActiveVersion = useCallback((): ModelVersion | null => {
    if (activeVersionId === null) return null;
    return versions.find(v => v.versionId === activeVersionId) ?? null;
  }, [versions, activeVersionId]);

  // Derived convenience accessors
  const activeVersion = getActiveVersion();
  const trainConfig = activeVersion?.trainConfig ?? null;
  const trainResult = activeVersion?.trainResult ?? null;

  return (
    <MLContext.Provider
      value={{
        stepCompleted,
        updateStepCompleted,
        cleanConfig,
        updateCleanConfig,
        versions,
        activeVersionId,
        addVersion,
        deleteVersion,
        renameVersion,
        setActiveVersion,
        importVersion,
        getActiveVersion,
        trainConfig,
        trainResult,
      }}
    >
      {children}
    </MLContext.Provider>
  );
}

export function useML() {
  const context = useContext(MLContext);
  if (!context) {
    throw new Error('useML must be used within an MLProvider');
  }
  return context;
}
