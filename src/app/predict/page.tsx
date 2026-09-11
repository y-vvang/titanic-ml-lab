'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useI18n } from '@/lib/i18n';
import { useML } from '@/lib/MLContext';

interface PredictInput {
  Pclass: number;
  Sex: string;
  Age: number;
  SibSp: number;
  Parch: number;
  Fare: number;
  Embarked: string;
  CabinDeck?: string;
  Title?: string;
}

const DEFAULT_INPUT: PredictInput = {
  Pclass: 3,
  Sex: 'male',
  Age: 25,
  SibSp: 0,
  Parch: 0,
  Fare: 15,
  Embarked: 'S',
  CabinDeck: 'A',
  Title: 'Mr',
};

export default function PredictPage() {
  const { t, lang } = useI18n();
  const { versions, activeVersionId, setActiveVersion, trainResult, trainConfig, getActiveVersion } = useML();
  const activeVersion = getActiveVersion();
  const cleanConfig = activeVersion?.trainConfig.cleanConfig ?? null;
  const selectedFeatures = cleanConfig?.selectedFeatures || [];
  const [input, setInput] = useState<PredictInput>({ ...DEFAULT_INPUT });
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<{
    survivalProbability: number;
    prediction: string;
    topImpacts: Array<{ feature: string; impact: number }>;
  } | null>(null);

  const hasModel = !!trainResult;

  // Check if logistic regression with unfilled missing values
  const isLogisticWithMissing = activeVersion?.trainConfig.modelType === 'logistic_regression' && activeVersion.trainResult.hasUnfilledMissing;

  const handlePredict = async () => {
    if (!hasModel) {
      alert(t('evaluate.trainFirst'));
      return;
    }
    setPredicting(true);
    try {
      const res = await fetch('/api/model/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          passenger: input,
          modelInfo: trainResult?.modelInfo,
          cleanConfig: cleanConfig,
          hyperparams: trainConfig?.hyperparams,
          lang,
        }),
      });
      const data = await res.json();
      if (data.survivalProbability !== undefined) {
        setResult(data);
      }
    } catch (err) {
      console.error('Predict error:', err);
    } finally {
      setPredicting(false);
    }
  };

  const handleReset = () => {
    setInput({ ...DEFAULT_INPUT });
    setResult(null);
  };

  // Gauge SVG calculation
  const prob = result ? result.survivalProbability / 100 : 0;
  const gaugeAngle = prob * 180; // 0 to 180 degrees
  const gaugeR = 80;
  const gaugeCx = 100;
  const gaugeCy = 100;

  // Calculate arc path for gauge
  const describeArc = (cx: number, cy: number, r: number, startAngle: number, endAngle: number) => {
    const start = polarToCartesian(cx, cy, r, endAngle);
    const end = polarToCartesian(cx, cy, r, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1';
    return `M ${end.x} ${end.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${start.x} ${start.y}`;
  };

  const polarToCartesian = (cx: number, cy: number, r: number, angle: number) => {
    const rad = ((angle - 180) * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  };

  const getProbColor = (p: number) => {
    if (p >= 0.7) return 'var(--color-success)';
    if (p >= 0.4) return 'var(--color-accent)';
    return 'var(--color-destructive)';
  };

  const getProbLabel = (p: number) => {
    if (p >= 0.7) return t('predict.probHigh');
    if (p >= 0.4) return t('predict.probMedium');
    return t('predict.probLow');
  };

  return (
    <div className="space-y-8 p-5">
      <div>
        <h1 className="text-3xl font-bold text-foreground">{t('predict.title')}</h1>
        <p className="text-muted-foreground mt-2">{t('predict.subtitle')}</p>
      </div>

      {!hasModel ? (
        <div className="flex items-center justify-center min-h-[40vh]">
          <div className="glass-card p-8 text-center">
            <p className="text-muted-foreground mb-6">{t('evaluate.trainFirst')}</p>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Version Selector */}
          {versions.length > 0 && (
            <div className="glass-card p-4 flex items-center gap-4">
              <label className="text-sm text-muted-foreground whitespace-nowrap">{t('evaluate.selectVersion')}</label>
              <select
                value={activeVersionId ?? ''}
                onChange={(e) => setActiveVersion(Number(e.target.value))}
                className="flex-1 bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
              >
                {versions.map(v => (
                  <option key={v.versionId} value={v.versionId}>
                    {v.name} — {t(`versions.modelType.${v.trainConfig.modelType}`)} ({(v.trainResult.metrics.accuracy * 100).toFixed(1)}%)
                  </option>
                ))}
              </select>
            </div>
          )}

          {isLogisticWithMissing && (
            <div className="glass-card p-4 border border-destructive/30">
              <p className="text-sm text-destructive">{t('predict.logisticMissingWarning')}</p>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Form */}
          <div className="glass-card p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4">{t('predict.passengerInfo')}</h2>
            <div className="space-y-5">
              {/* Pclass */}
              <div>
                <label className="text-sm font-medium text-foreground block mb-2">{t('predict.fieldPclass')}</label>
                <select value={input.Pclass}
                  onChange={e => setInput(p => ({ ...p, Pclass: parseInt(e.target.value) }))}
                  className="w-full bg-muted border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30">
                  <option value={1}>{t('predict.pclass1')}</option>
                  <option value={2}>{t('predict.pclass2')}</option>
                  <option value={3}>{t('predict.pclass3')}</option>
                </select>
              </div>

              {/* Sex */}
              <div>
                <label className="text-sm font-medium text-foreground block mb-2">{t('predict.fieldSex')}</label>
                <div className="flex gap-3">
                  {['male', 'female'].map(sex => (
                    <button key={sex}
                      onClick={() => setInput(p => ({ ...p, Sex: sex }))}
                      className={`flex-1 py-2.5 rounded-lg text-sm font-medium transition-all ${
                        input.Sex === sex
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted text-muted-foreground hover:text-foreground'
                      }`}>
                      {sex === 'male' ? t('explore.male') : t('explore.female')}
                    </button>
                  ))}
                </div>
              </div>

              {/* Age */}
              <div>
                <label className="text-sm font-medium text-foreground block mb-2">
                  {t('predict.fieldAge')}: <span className="text-accent font-mono">{input.Age}</span> {t('predict.unitYears')}
                </label>
                <input type="number" value={input.Age} min={0} max={100}
                  onChange={e => setInput(p => ({ ...p, Age: parseInt(e.target.value) || 0 }))}
                  className="w-full bg-muted border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
              </div>

              {/* SibSp & Parch */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">{t('explore.featSibSp')}</label>
                  <input type="number" value={input.SibSp} min={0} max={8}
                    onChange={e => setInput(p => ({ ...p, SibSp: parseInt(e.target.value) || 0 }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">{t('explore.featParch')}</label>
                  <input type="number" value={input.Parch} min={0} max={6}
                    onChange={e => setInput(p => ({ ...p, Parch: parseInt(e.target.value) || 0 }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
                </div>
              </div>

              {/* Fare */}
              <div>
                <label className="text-sm font-medium text-foreground block mb-2">
                  {t('predict.fieldFare')}: <span className="text-accent font-mono">{input.Fare}</span> {t('predict.unitPounds')}
                </label>
                <input type="number" value={input.Fare} min={0} max={600} step={1}
                  onChange={e => setInput(p => ({ ...p, Fare: parseFloat(e.target.value) || 0 }))}
                  className="w-full bg-muted border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
              </div>

              {/* Embarked */}
              <div>
                <label className="text-sm font-medium text-foreground block mb-2">{t('predict.fieldEmbarked')}</label>
                <select value={input.Embarked}
                  onChange={e => setInput(p => ({ ...p, Embarked: e.target.value }))}
                  className="w-full bg-muted border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30">
                  <option value="S">{t('predict.portS')}</option>
                  <option value="C">{t('predict.portC')}</option>
                  <option value="Q">{t('predict.portQ')}</option>
                </select>
              </div>

              {/* CabinDeck - only show if was used in training */}
              {selectedFeatures.includes('CabinDeck') && (
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">{t('predict.fieldCabinDeck')}</label>
                  <select value={input.CabinDeck || ''}
                    onChange={e => setInput(p => ({ ...p, CabinDeck: e.target.value }))}
                    disabled={isLogisticWithMissing}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:opacity-50 disabled:cursor-not-allowed">
                    {['A', 'B', 'C', 'D', 'E', 'F', 'G'].map(d => (
                      <option key={d} value={d}>{t('predict.deckOption', { d })}</option>
                    ))}
                  </select>
                </div>
              )}

              {/* Title - only show if was used in training */}
              {selectedFeatures.includes('Title') && (
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">{t('predict.fieldTitle')}</label>
                  <select value={input.Title || ''}
                    onChange={e => setInput(p => ({ ...p, Title: e.target.value }))}
                    disabled={isLogisticWithMissing}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:opacity-50 disabled:cursor-not-allowed">
                    <option value="Mr">{t('predict.titleMr')}</option>
                    <option value="Mrs">{t('predict.titleMrs')}</option>
                    <option value="Miss">{t('predict.titleMiss')}</option>
                    <option value="Master">{t('predict.titleMaster')}</option>
                    <option value="Rare">{t('predict.titleRare')}</option>
                  </select>
                </div>
              )}

              {/* Buttons */}
              <div className="flex gap-3 pt-2">
                <button onClick={handlePredict} disabled={predicting}
                  className="flex-1 btn-gradient py-3 rounded-xl text-sm font-medium disabled:opacity-50">
                  {predicting ? t('predict.predicting') : t('predict.predictButton')}
                </button>
                <button onClick={handleReset}
                  className="px-5 py-3 rounded-xl text-sm font-medium border border-border text-muted-foreground hover:text-foreground transition-all">
                  {t('predict.reset')}
                </button>
              </div>
            </div>
          </div>

          {/* Result Area */}
          <div className="space-y-6">
            {/* Gauge */}
            <div className="glass-card p-6">
              <h2 className="text-lg font-semibold text-foreground mb-4">{t('predict.gaugeTitle')}</h2>
              <div className="flex flex-col items-center">
                <svg viewBox="0 0 200 120" className="w-64 h-40">
                  {/* Background arc */}
                  <path d={describeArc(gaugeCx, gaugeCy, gaugeR, 0, 180)}
                    fill="none" stroke="currentColor" strokeWidth="16"
                    className="text-muted/20" strokeLinecap="round" />
                  {/* Value arc */}
                  {result && (
                    <path d={describeArc(gaugeCx, gaugeCy, gaugeR, 0, gaugeAngle)}
                      fill="none" stroke={getProbColor(prob)} strokeWidth="16"
                      strokeLinecap="round" style={{ filter: `drop-shadow(0 0 6px ${getProbColor(prob)})` }} />
                  )}
                  {/* Needle */}
                  {result && (
                    <line
                      x1={gaugeCx} y1={gaugeCy}
                      x2={polarToCartesian(gaugeCx, gaugeCy, gaugeR - 20, gaugeAngle).x}
                      y2={polarToCartesian(gaugeCx, gaugeCy, gaugeR - 20, gaugeAngle).y}
                      stroke="var(--color-accent)" strokeWidth="2" strokeLinecap="round"
                      style={{ filter: 'drop-shadow(0 0 4px var(--color-accent))' }}
                    />
                  )}
                  {/* Center dot */}
                  <circle cx={gaugeCx} cy={gaugeCy} r="4" fill="var(--color-accent)" />
                  {/* Labels */}
                  <text x="20" y="110" className="fill-muted-foreground" style={{ fontSize: '10px' }}>0%</text>
                  <text x="90" y="20" className="fill-muted-foreground" style={{ fontSize: '10px' }}>50%</text>
                  <text x="165" y="110" className="fill-muted-foreground" style={{ fontSize: '10px' }}>100%</text>
                </svg>

                {/* Probability number */}
                {result ? (
                  <div className="text-center mt-2">
                    <p className="text-4xl font-bold" style={{ color: getProbColor(prob) }}>
                      {(prob * 100).toFixed(1)}%
                    </p>
                    <p className="text-sm mt-1 font-medium" style={{ color: getProbColor(prob) }}>
                      {t('predict.probWithLabel', { label: getProbLabel(prob) })}
                    </p>
                  </div>
                ) : (
                  <p className="text-muted-foreground text-sm mt-4">{t('predict.fillThenPredict')}</p>
                )}
              </div>
            </div>

            {/* Interpretation */}
            {result && (
              <div className="glass-card p-6">
                <h2 className="text-lg font-semibold text-foreground mb-3">{t('predict.interpretation')}</h2>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {prob >= 0.7
                    ? t('predict.interpHigh')
                    : prob >= 0.4
                    ? t('predict.interpMedium')
                    : t('predict.interpLow')}
                </p>
                {/* Top Factors */}
                {result.topImpacts && result.topImpacts.length > 0 && (
                  <div className="mt-4">
                    <p className="text-xs text-muted-foreground mb-2">{t('predict.topFactors')}</p>
                    <div className="flex flex-wrap gap-2">
                      {result.topImpacts.map((f, i) => (
                        <span key={i} className={`text-xs px-3 py-1 rounded-full ${
                          f.impact > 0
                            ? 'bg-success/20 text-success'
                            : 'bg-destructive/20 text-destructive'
                        }`}>
                          {f.feature} {f.impact > 0 ? '+' : ''}{f.impact}%
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          </div>
        </div>
      )}

      {/* Bottom Actions */}
      <div className="flex justify-between items-center">
        <Link href="/evaluate" className="px-6 py-3 rounded-xl text-sm font-medium border border-border text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all">
          {t('predict.backButton')}
        </Link>
        <p className="text-xs text-muted-foreground">{t('predict.footer')}</p>
      </div>
    </div>
  );
}
