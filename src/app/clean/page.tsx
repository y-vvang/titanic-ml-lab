'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import Link from 'next/link';
import { useI18n } from '@/lib/i18n';
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

// Base feature keys with i18n-agnostic metadata; labels/descriptions come from t()
const BASE_FEATURE_KEYS: Omit<FeatureDef, 'desc' | 'importance' | 'label'>[] = [
  { key: 'PassengerId', recommended: false, disabled: true },
  { key: 'Pclass', recommended: true, disabled: false },
  { key: 'Sex', recommended: true, disabled: false },
  { key: 'Age', recommended: true, disabled: false },
  { key: 'SibSp', recommended: false, disabled: false },
  { key: 'Parch', recommended: false, disabled: false },
  { key: 'Fare', recommended: true, disabled: false },
  { key: 'Embarked', recommended: false, disabled: false },
  { key: 'Ticket', recommended: false, disabled: true },
  { key: 'Name', recommended: false, disabled: true },
  { key: 'Cabin', recommended: false, disabled: true },
  // Derived features (conditional)
  { key: 'Title', recommended: false, disabled: false, conditional: true },
  { key: 'CabinDeck', recommended: false, disabled: false, conditional: true },
];

export default function CleanPage() {
  const { t, lang } = useI18n();
  const { cleanConfig, updateCleanConfig, updateStepCompleted } = useML();

  const BASE_FEATURES: FeatureDef[] = useMemo(() => {
    const desc: Record<string, string> = {
      PassengerId: t('clean.featPassengerIdDesc'),
      Embarked: t('explore.featEmbarked'),
      Ticket: t('clean.featTicketDesc'),
      Name: t('clean.featNameDesc'),
      Cabin: t('clean.featCabinDesc'),
      Title: t('clean.featTitleDesc'),
      CabinDeck: t('clean.featCabinDeckDesc'),
      Pclass: t('explore.featPclass'),
      Sex: t('explore.featSex'),
      Age: t('explore.featAge'),
      SibSp: t('explore.featSibSp'),
      Parch: t('explore.featParch'),
      Fare: t('explore.featFare'),
    };
    const imp: Record<string, string> = {
      PassengerId: t('clean.featPassengerIdImp'),
      Pclass: t('clean.featPclassImp'),
      Sex: t('clean.featSexImp'),
      Age: t('clean.featAgeImp'),
      SibSp: t('clean.featSibSpImp'),
      Parch: t('clean.featParchImp'),
      Fare: t('clean.featFareImp'),
      Embarked: t('clean.featEmbarkedImp'),
      Ticket: t('clean.featTicketImp'),
      Name: t('clean.featNameImp'),
      Cabin: t('clean.featCabinImp'),
      Title: t('clean.featTitleImp'),
      CabinDeck: t('clean.featCabinDeckImp'),
    };
    return BASE_FEATURE_KEYS.map(f => ({ ...f, label: f.key, desc: desc[f.key], importance: imp[f.key] }));
    // `t` 的身份随 lang 变化，因此这里只需依赖 t
  }, [t]);

  const [missingStrategy, setMissingStrategy] = useState<MissingValueStrategy>({
    Age: 'median',
    Embarked: 'mode',
    Cabin: 'drop',
  });
  const [nameStrategy, setNameStrategy] = useState<NameStrategy>('drop');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(
    BASE_FEATURE_KEYS.filter(f => f.recommended).map(f => f.key)
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
  }, [BASE_FEATURES, titleAvailable, cabinDeckAvailable]);

  // Auto-adjust selected features when config changes
  useEffect(() => {
    setSelectedFeatures(prev => {
      const validKeys = features.filter(f => !f.disabled && f.conditionMet).map(f => f.key);
      // Remove features that are no longer available
      const filtered = prev.filter(k => validKeys.includes(k));
      return filtered;
    });
  }, [features]);

  // Sync with global state
  useEffect(() => {
    if (cleanConfig) {
      setMissingStrategy(cleanConfig.missingValueStrategy as MissingValueStrategy);
      setSelectedFeatures(cleanConfig.selectedFeatures);
      if (cleanConfig.nameStrategy) setNameStrategy(cleanConfig.nameStrategy as NameStrategy);
    }
  }, [cleanConfig]);

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
      lang,
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
        const { lang: _lang, ...cleanConfigPayload } = config;
        void _lang;
        updateCleanConfig({ ...cleanConfigPayload, hasUnfilledMissing: data.hasUnfilledMissing || false });
        updateStepCompleted('clean');
      }
    } catch (err) {
      console.error('Clean error:', err);
    } finally {
      setApplying(false);
    }
  }, [missingStrategy, selectedFeatures, nameStrategy, lang, updateCleanConfig, updateStepCompleted]);

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
        <h1 className="text-3xl font-bold text-foreground">{t('clean.title')}</h1>
        <p className="text-muted-foreground mt-2">{t('clean.subtitle')}</p>
      </div>

      {/* Missing Value Section */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">{t('clean.numericSection')}</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-4">
          {/* Age */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground">{t('clean.ageHeader')}</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-accent/20 text-accent">{t('clean.missingPct', { pct: '19.9' })}</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">{t('clean.ageCardDesc')}</p>
            <select
              value={missingStrategy.Age}
              onChange={e => { setMissingStrategy(s => ({ ...s, Age: e.target.value as AgeStrategy })); setApplied(false); }}
              className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
            >
              <option value="mean">{t('clean.ageMean')}</option>
              <option value="median">{t('clean.ageMedian')}</option>
              <option value="mode">{t('clean.ageMode')}</option>
              <option value="none">{t('clean.keepMissingTrees')}</option>
            </select>
            <p className="text-xs text-muted-foreground mt-2">
              {missingStrategy.Age === 'mean' && t('clean.ageMeanHint')}
              {missingStrategy.Age === 'median' && t('clean.ageMedianHint')}
              {missingStrategy.Age === 'mode' && t('clean.ageModeHint')}
              {missingStrategy.Age === 'none' && t('clean.ageNoneHint')}
            </p>
          </div>

          {/* Embarked */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground">{t('clean.embarkedHeader')}</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-success/20 text-success">{t('clean.missingPct', { pct: '0.2' })}</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">{t('clean.embarkedCardDesc')}</p>
            <select
              value={missingStrategy.Embarked}
              onChange={e => { setMissingStrategy(s => ({ ...s, Embarked: e.target.value as EmbarkedStrategy })); setApplied(false); }}
              className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
            >
              <option value="mode">{t('clean.embarkedMode')}</option>
              <option value="drop">{t('clean.dropRow')}</option>
              <option value="none">{t('clean.keepMissingTrees')}</option>
            </select>
          </div>

          {/* Cabin */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground">{t('clean.cabinHeader')}</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-destructive/20 text-destructive">{t('clean.missingPct', { pct: '77.1' })}</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">{t('clean.cabinCardDesc')}</p>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="cabin" value="drop" checked={missingStrategy.Cabin === 'drop'}
                  onChange={() => { setMissingStrategy(s => ({ ...s, Cabin: 'drop' })); setApplied(false); }}
                  className="accent-primary" />
                <span className="text-sm text-foreground">{t('clean.dropColumn')}</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="cabin" value="extract_deck" checked={missingStrategy.Cabin === 'extract_deck'}
                  onChange={() => { setMissingStrategy(s => ({ ...s, Cabin: 'extract_deck' })); setApplied(false); }}
                  className="accent-primary" />
                <span className="text-sm text-foreground">{t('clean.extractDeck')}</span>
              </label>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {missingStrategy.Cabin === 'extract_deck'
                ? t('clean.cabinExtractHint')
                : t('clean.cabinDropHint')}
            </p>
          </div>

          {/* Name / Title */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground">{t('clean.nameHeader')}</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-primary/20 text-primary">{t('clean.complete')}</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">{t('clean.nameCardDesc')}</p>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="name" value="drop" checked={nameStrategy === 'drop'}
                  onChange={() => { setNameStrategy('drop'); setApplied(false); }}
                  className="accent-primary" />
                <span className="text-sm text-foreground">{t('clean.dropName')}</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="name" value="extract_title" checked={nameStrategy === 'extract_title'}
                  onChange={() => { setNameStrategy('extract_title'); setApplied(false); }}
                  className="accent-primary" />
                <span className="text-sm text-foreground">{t('clean.extractTitle')}</span>
              </label>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {nameStrategy === 'extract_title'
                ? t('clean.nameExtractHint')
                : t('clean.nameDropHint')}
            </p>
          </div>
        </div>
      </section>

      {/* Feature Selection */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">
          {t('clean.featureSection')} <span className="text-sm font-normal text-muted-foreground">{t('clean.selectedCount', { n: selectedFeatures.length })}</span>
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Recommended */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-medium text-success mb-3 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-success" /> {t('clean.recommendedKeep')}
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
              <span className="w-2 h-2 rounded-full bg-warning" /> {t('clean.optionalFeatures')}
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
                  <span className="text-xs px-1.5 py-0.5 rounded bg-primary/20 text-primary ml-1">{t('clean.derived')}</span>
                </label>
              ))}
              {disabledFeatures.length > 0 && disabledFeatures.map(f => (
                <div key={f.key} className="flex items-center gap-3 p-2 rounded-lg opacity-50">
                  <input type="checkbox" checked={false} disabled className="accent-primary w-4 h-4" />
                  <span className="text-sm text-muted-foreground">{f.key}</span>
                  <span className="text-xs text-muted-foreground">{f.desc}</span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-destructive/20 text-destructive/70 ml-1">{t('clean.disabled')}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Missing Value Warning */}
      {hasUnfilledMissing && (
        <div className="glass-card p-5 border border-warning/30">
          <h3 className="text-warning font-medium mb-2">{t('clean.unfilledWarningTitle')}</h3>
          <p className="text-sm text-muted-foreground">
            {t('clean.unfilledWarningBody', {
              features: Object.entries(cleanResult?.missingAfterClean || {})
                .filter(([k, v]) => (v as number) > 0 && selectedFeatures.includes(k))
                .map(([k]) => k).join(', '),
            })}
          </p>
        </div>
      )}

      {/* Clean Result */}
      {cleanResult && (
        <div className="glass-card p-5 border border-success/30">
          <h3 className="text-success font-medium mb-2">{t('clean.resultTitle')}</h3>
          <p className="text-sm text-muted-foreground">
            {t('clean.resultStart')}
            <span className="text-foreground font-medium">{cleanResult.rows}</span>
            {t('clean.resultMid')}
            <span className="text-foreground font-medium">{cleanResult.features.length}</span>
            {t('clean.resultEnd')}
            {t('clean.resultFeatList', { features: cleanResult.features.join(', ') })}
            {cleanResult.hasUnfilledMissing && (
              <span className="text-warning ml-2">{t('clean.resultUnfilled')}</span>
            )}
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-between items-center">
        <Link href="/explore" className="px-6 py-3 rounded-xl text-sm font-medium border border-border text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all">
          {t('clean.backButton')}
        </Link>
        <div className="flex gap-3">
          <button onClick={handleApply} disabled={applying || noFeatureSelected}
            className="btn-gradient px-6 py-3 rounded-xl text-sm font-medium disabled:opacity-50">
            {applying ? t('clean.applying') : applied ? t('clean.applied') : t('clean.applyButton')}
          </button>
          <Link href="/train"
            className={`px-6 py-3 rounded-xl text-sm font-medium transition-all ${
              canProceed
                ? 'btn-gradient'
                : 'bg-muted text-muted-foreground cursor-not-allowed pointer-events-none'
            }`}>
            {t('common.nextStep', { step: t('nav.stepTrain') })}
          </Link>
        </div>
      </div>
    </div>
  );
}
