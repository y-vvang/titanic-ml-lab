'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useI18n } from '@/lib/i18n';

interface FeatureDistribution {
  type: 'categorical' | 'numerical';
  values?: string[];
  labels?: Record<string, string>;
  counts: Record<string, number> | number[];
  survivedCounts: Record<string, number> | number[];
  bins?: string[];
}

interface ExploreData {
  total: number;
  survived: number;
  died: number;
  survivalRate: number;
  sexSurvival: Record<string, { survived: number; died: number; rate: number }>;
  pclassSurvival: Record<string, { survived: number; died: number; rate: number }>;
  ageHistogram: { bins: number[]; survived: number[]; died: number[] };
  fareStats: {
    survived: { mean: number; median: number; q25: number; q75: number };
    died: { mean: number; median: number; q25: number; q75: number };
  };
  correlation: { columns: string[]; matrix: number[][] };
  featureDistributions: Record<string, FeatureDistribution>;
}

export default function ExplorePage() {
  const { t, lang } = useI18n();
  const [data, setData] = useState<ExploreData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Explicit lang keeps the engine output in sync with navigator.language detection
    fetch(`/api/data/explore?lang=${lang}`)
      .then(res => res.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [lang]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">{t('common.loading')}</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="glass-card p-8 text-center">
          <p className="text-destructive">{t('common.loadFailed')}: {error}</p>
        </div>
      </div>
    );
  }

  const { total, survived, died, survivalRate, sexSurvival, pclassSurvival, fareStats } = data;

  // Age histogram: create labels from bins
  const ageBins = data.ageHistogram.bins;
  const ageLabels = ageBins.slice(0, -1).map((b, i) => `${b}-${ageBins[i + 1]}`);

  // Correlation data
  const corrColumns = data.correlation.columns;
  const corrMatrix = data.correlation.matrix;

  // Feature label mapping for distributions
  const featLabels: Record<string, string> = {
    Pclass: t('explore.featPclass'), Sex: t('explore.featSex'), Age: t('explore.featAge'), SibSp: t('explore.featSibSp'),
    Parch: t('explore.featParch'), Fare: t('explore.featFare'), Embarked: t('explore.featEmbarked'),
    CabinDeck: t('explore.featCabinDeck'), Title: t('explore.featTitle'),
  };

  return (
    <div className="space-y-8 p-5">
      {/* Page Title */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">{t('explore.title')}</h1>
        <p className="text-muted-foreground mt-2">{t('explore.subtitle')}</p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-5">
          <p className="text-sm text-muted-foreground mb-1">{t('explore.totalPassengers')}</p>
          <p className="text-3xl font-bold text-foreground">{total}</p>
          <p className="text-xs text-muted-foreground mt-1">{t('explore.passengerRecords')}</p>
        </div>
        <div className="glass-card p-5 border-l-2 border-l-primary">
          <p className="text-sm text-muted-foreground mb-1">{t('explore.survivalRate')}</p>
          <p className="text-3xl font-bold text-primary">{survivalRate.toFixed(1)}%</p>
          <p className="text-xs text-muted-foreground mt-1">{t('explore.survivedOfDied', { survived, died })}</p>
        </div>
        <div className="glass-card p-5">
          <p className="text-sm text-muted-foreground mb-1">{t('explore.averageAge')}</p>
          <p className="text-3xl font-bold text-foreground">
            {data.ageHistogram.survived.reduce((a, b) => a + b, 0) > 0 ? '29.7' : 'N/A'}
          </p>
          <p className="text-xs text-muted-foreground mt-1">{t('explore.yearsOld')}</p>
        </div>
        <div className="glass-card p-5">
          <p className="text-sm text-muted-foreground mb-1">{t('explore.missingValues')}</p>
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Age</span>
              <span className="text-accent font-medium">19.9%</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Cabin</span>
              <span className="text-accent font-medium">77.1%</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Embarked</span>
              <span className="text-accent font-medium">0.2%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {/* Survival Rate Donut */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-foreground mb-1">{t('explore.donutTitle')}</h3>
          <p className="text-xs text-muted-foreground mb-4">{t('explore.donutDesc')}</p>
          <div className="flex items-center justify-center">
            <svg viewBox="0 0 200 200" className="w-48 h-48">
              <circle cx="100" cy="100" r="80" fill="none" stroke="currentColor" strokeWidth="20"
                className="text-muted/30" />
              <circle cx="100" cy="100" r="80" fill="none" stroke="currentColor" strokeWidth="20"
                className="text-primary"
                strokeDasharray={`${(survivalRate / 100) * 2 * Math.PI * 80} ${2 * Math.PI * 80}`}
                strokeDashoffset={0}
                transform="rotate(-90 100 100)" />
              <text x="100" y="95" textAnchor="middle" className="fill-foreground"
                style={{ fontSize: '28px', fontWeight: 'bold' }}>{survivalRate.toFixed(1)}%</text>
              <text x="100" y="118" textAnchor="middle" className="fill-muted-foreground"
                style={{ fontSize: '12px' }}>{t('explore.survivalRate')}</text>
            </svg>
          </div>
          <div className="flex justify-center gap-6 mt-2 text-sm">
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-primary" />{t('explore.survived')}</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-muted/30" />{t('explore.died')}</span>
          </div>
          <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
            {t('explore.donutNote')}
          </p>
        </div>

        {/* Sex vs Survival */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-foreground mb-1">{t('explore.sexTitle')}</h3>
          <p className="text-xs text-muted-foreground mb-4">{t('explore.sexDesc')}</p>
          <div className="space-y-4">
            {Object.entries(sexSurvival).map(([key, info]) => (
              <div key={key}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-foreground">{key === 'female' ? t('explore.female') : t('explore.male')}</span>
                  <span className="text-primary font-medium">{info.rate.toFixed(0)}%</span>
                </div>
                <div className="h-8 bg-muted/30 rounded-lg overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-primary to-accent rounded-lg transition-all duration-500"
                    style={{ width: `${info.rate}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
            {t('explore.sexNote')}
          </p>
        </div>

        {/* Pclass vs Survival */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-foreground mb-1">{t('explore.pclassTitle')}</h3>
          <p className="text-xs text-muted-foreground mb-4">{t('explore.pclassDesc')}</p>
          <div className="space-y-4">
            {Object.entries(pclassSurvival).sort().map(([key, info]) => (
              <div key={key}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-foreground">{t('explore.pclassLabel', { n: key })}</span>
                  <span className="text-success font-medium">{info.rate.toFixed(0)}%</span>
                </div>
                <div className="h-8 bg-muted/30 rounded-lg overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-success/80 to-accent rounded-lg transition-all duration-500"
                    style={{ width: `${info.rate}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
            {t('explore.pclassNote')}
          </p>
        </div>

        {/* Age Distribution */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-foreground mb-1">{t('explore.ageTitle')}</h3>
          <p className="text-xs text-muted-foreground mb-4">{t('explore.ageDesc')}</p>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {ageLabels.slice(0, 10).map((label, i) => {
              const maxVal = Math.max(...data.ageHistogram.survived.slice(0, 10), ...data.ageHistogram.died.slice(0, 10), 1);
              return (
                <div key={label} className="flex items-center gap-2 text-xs">
                  <span className="w-14 text-muted-foreground text-right shrink-0">{label}</span>
                  <div className="flex-1 flex gap-0.5">
                    <div className="flex-1 bg-muted/20 rounded-sm overflow-hidden h-5">
                      <div className="h-full bg-primary/60 rounded-sm"
                        style={{ width: `${(data.ageHistogram.survived[i] / maxVal) * 100}%` }} />
                    </div>
                    <div className="flex-1 bg-muted/20 rounded-sm overflow-hidden h-5">
                      <div className="h-full bg-destructive/50 rounded-sm"
                        style={{ width: `${(data.ageHistogram.died[i] / maxVal) * 100}%` }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="flex gap-4 mt-2 text-xs">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-primary/60" />{t('explore.survived')}</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-destructive/50" />{t('explore.died')}</span>
          </div>
          <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
            {t('explore.ageNote')}
          </p>
        </div>

        {/* Fare Distribution */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-foreground mb-1">{t('explore.fareTitle')}</h3>
          <p className="text-xs text-muted-foreground mb-4">{t('explore.fareDesc')}</p>
          <div className="space-y-6 mt-6">
            {[
              { label: t('explore.survivors'), stats: fareStats.survived, color: 'bg-primary' },
              { label: t('explore.nonSurvivors'), stats: fareStats.died, color: 'bg-destructive/60' },
            ].map(({ label, stats, color }) => {
              const maxFare = Math.max(fareStats.survived.q75, fareStats.died.q75, 1);
              return (
                <div key={label}>
                  <p className="text-sm text-foreground mb-2">{t('explore.fareOf', { label })}</p>
                  <div className="relative h-6 bg-muted/20 rounded">
                    <div className={`absolute h-full ${color} rounded opacity-60`}
                      style={{ left: `${(stats.q25 / maxFare) * 100}%`, width: `${((stats.q75 - stats.q25) / maxFare) * 100}%` }} />
                    <div className="absolute h-full w-0.5 bg-foreground top-0"
                      style={{ left: `${(stats.median / maxFare) * 100}%` }} />
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>Q1: £{stats.q25.toFixed(0)}</span>
                    <span>{t('explore.median', { v: stats.median.toFixed(0) })}</span>
                    <span>Q3: £{stats.q75.toFixed(0)}</span>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-muted-foreground mt-3 leading-relaxed">
            {t('explore.fareNote')}
          </p>
        </div>

        {/* Correlation Heatmap */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-foreground mb-1">{t('explore.corrTitle')}</h3>
          <p className="text-xs text-muted-foreground mb-4">{t('explore.corrDesc')}</p>
          <div className="overflow-x-auto">
            <div className="inline-block min-w-[280px]">
              <div className="flex">
                <div className="w-20" />
                {corrColumns.map(f => (
                  <div key={f} className="w-11 text-center text-[10px] text-muted-foreground truncate">{f.replace('Code','')}</div>
                ))}
              </div>
              {corrMatrix.map((row, i) => (
                <div key={corrColumns[i]} className="flex items-center">
                  <div className="w-20 text-[10px] text-muted-foreground text-right pr-2 truncate">{corrColumns[i].replace('Code','')}</div>
                  {row.map((val: number, j: number) => {
                    const absVal = Math.abs(val);
                    const isNeg = val < 0;
                    return (
                      <div key={j} className="w-11 h-10 flex items-center justify-center text-[10px] font-medium"
                        style={{
                          backgroundColor: isNeg
                            ? `rgba(239, 68, 68, ${absVal * 0.6})`
                            : `rgba(124, 92, 255, ${absVal * 0.6})`,
                          color: absVal > 0.4 ? '#fff' : 'var(--color-foreground)',
                        }}>
                        {val.toFixed(1)}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-primary/40" />{t('explore.positiveCorr')}</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-destructive/40" />{t('explore.negativeCorr')}</span>
          </div>
        </div>
      </div>

      {/* Feature Frequency Distributions */}
      {data.featureDistributions && (
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-2">{t('explore.freqTitle')}</h2>
          <p className="text-muted-foreground mb-6">{t('explore.freqDesc')}</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(data.featureDistributions).map(([feat, dist]) => {
              const isNumerical = dist.type === 'numerical';
              const entries = isNumerical
                ? (dist.bins || []).map((bin, i) => ({
                    label: bin,
                    count: (dist.counts as number[])[i] || 0,
                    survived: (dist.survivedCounts as number[])[i] || 0,
                  }))
                : (dist.values || []).map(v => ({
                    label: dist.labels?.[v] || v,
                    count: (dist.counts as Record<string, number>)[v] || 0,
                    survived: (dist.survivedCounts as Record<string, number>)[v] || 0,
                  }));
              const maxCount = Math.max(...entries.map(e => e.count), 1);
              return (
                <div key={feat} className="glass-card p-4">
                  <h4 className="text-sm font-semibold text-foreground mb-1">{featLabels[feat] || feat}</h4>
                  <p className="text-[10px] text-muted-foreground mb-3">
                    {isNumerical ? t('explore.numericalBins') : t('explore.categoricalFreq')}
                  </p>
                  <div className="space-y-1.5">
                    {entries.map((entry, i) => {
                      const died = entry.count - entry.survived;
                      const survivalRate = entry.count > 0 ? (entry.survived / entry.count) * 100 : 0;
                      return (
                        <div key={i}>
                          <div className="flex justify-between text-[10px] mb-0.5">
                            <span className="text-muted-foreground truncate max-w-[60%]">{entry.label}</span>
                            <span className="text-foreground font-medium shrink-0">
                              {entry.count}
                              <span className="text-muted-foreground ml-1">({t('explore.pctSurvived', { pct: survivalRate.toFixed(0) })})</span>
                            </span>
                          </div>
                          <div className="flex h-4 bg-muted/20 rounded-sm overflow-hidden">
                            <div
                              className="h-full bg-primary/70 rounded-l-sm transition-all duration-500"
                              style={{ width: `${(entry.survived / maxCount) * 100}%` }}
                            />
                            <div
                              className="h-full bg-destructive/40 transition-all duration-500"
                              style={{ width: `${(died / maxCount) * 100}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="flex gap-3 mt-2 text-[10px]">
                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-primary/70" />{t('explore.survived')}</span>
                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-destructive/40" />{t('explore.died')}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Next Step */}
      <div className="flex justify-between items-center">
        <Link href="/" className="px-6 py-3 rounded-xl text-sm font-medium border border-border text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all">
          {t('common.backHome')}
        </Link>
        <Link href="/clean" className="btn-gradient px-6 py-3 rounded-xl text-sm font-medium">
          {t('common.nextStep', { step: t('nav.stepClean') })}
        </Link>
      </div>
    </div>
  );
}
