'use client';

import Link from 'next/link';
import { useML } from '@/lib/MLContext';

export default function EvaluatePage() {
  const { versions, activeVersionId, setActiveVersion, trainResult: result } = useML();

  if (!result || versions.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="glass-card p-8 text-center">
          <p className="text-muted-foreground mb-6">请先完成模型训练</p>
          <div className="flex gap-3 justify-center">
            <Link href="/train" className="px-6 py-3 rounded-xl text-sm font-medium border border-border text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all">
              ← 返回模型训练
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const { metrics, confusionMatrix: cm, featureImportance, groupAccuracy } = result;
  const { accuracy, precision, recall, f1 } = metrics;

  // Confusion matrix values
  const tp = cm?.tp ?? 0;
  const fp = cm?.fp ?? 0;
  const fn = cm?.fn ?? 0;
  const tn = cm?.tn ?? 0;
  const matrixMax = Math.max(tp, fp, fn, tn, 1);

  // Feature importance sorted (array of {feature, importance})
  const sortedFeatures: Array<{ feature: string; importance: number }> = featureImportance
    ? [...featureImportance].sort((a, b) => b.importance - a.importance)
    : [];

  const featureMax = sortedFeatures.length > 0 ? sortedFeatures[0].importance : 1;

  return (
    <div className="space-y-8 p-5">
      <div>
        <h1 className="text-3xl font-bold text-foreground">模型评估</h1>
        <p className="text-muted-foreground mt-2">深入了解模型的表现，理解各项指标的含义</p>
      </div>

      {/* Version Selector */}
      {versions.length > 0 && (
        <div className="glass-card p-4 flex items-center gap-4">
          <label className="text-sm text-muted-foreground whitespace-nowrap">选择模型版本</label>
          <select
            value={activeVersionId ?? ''}
            onChange={(e) => setActiveVersion(Number(e.target.value))}
            className="flex-1 bg-muted/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
          >
            {versions.map(v => (
              <option key={v.versionId} value={v.versionId}>
                {v.name} — {v.trainConfig.modelType === 'decision_tree' ? '决策树' : v.trainConfig.modelType === 'random_forest' ? '随机森林' : '逻辑回归'} ({(v.trainResult.metrics.accuracy * 100).toFixed(1)}%)
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Core Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: '准确率', sublabel: 'Accuracy', value: accuracy, icon: '🎯', desc: '正确预测的比例', color: 'primary' },
          { label: '精确率', sublabel: 'Precision', value: precision, icon: '🔫', desc: '预测为生还中实际生还的比例', color: 'accent' },
          { label: '召回率', sublabel: 'Recall', value: recall, icon: '📡', desc: '实际生还中被正确识别的比例', color: 'success' },
          { label: 'F1分数', sublabel: 'F1-Score', value: f1, icon: '⚖️', desc: '精确率和召回率的调和平均', color: 'foreground' },
        ].map(m => (
          <div key={m.label} className="glass-card p-5">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">{m.icon}</span>
              <div>
                <p className="text-sm font-medium text-foreground">{m.label}</p>
                <p className="text-[10px] text-muted-foreground">{m.sublabel}</p>
              </div>
            </div>
            <p className={`text-3xl font-bold text-${m.color === 'foreground' ? 'foreground' : m.color}`}>
              {(m.value * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground mt-1">{m.desc}</p>
          </div>
        ))}
      </div>

      {/* Confusion Matrix */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">混淆矩阵</h2>
        <div className="glass-card p-5">
          <p className="text-xs text-muted-foreground mb-4">混淆矩阵展示了模型预测与实际结果的对比。对角线(左上到右下)上的数字越大越好。</p>
          <div className="flex items-start gap-6">
            <div className="overflow-x-auto">
              <table className="border-collapse">
                <thead>
                  <tr>
                    <th className="p-2" />
                    <th className="p-2 text-xs text-muted-foreground" />
                    <th colSpan={2} className="p-2 text-center text-sm font-medium text-foreground">实际值</th>
                  </tr>
                  <tr>
                    <th className="p-2" />
                    <th className="p-2" />
                    <th className="p-3 text-xs text-center text-muted-foreground">遇难 (0)</th>
                    <th className="p-3 text-xs text-center text-muted-foreground">生还 (1)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th rowSpan={2} className="p-2 text-sm font-medium text-foreground writing-mode-vertical"
                      style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}>预测值</th>
                    <th className="p-3 text-xs text-muted-foreground text-right">遇难 (0)</th>
                    <td className="p-3">
                      <div className="w-20 h-20 rounded-lg flex flex-col items-center justify-center text-lg font-bold"
                        style={{ backgroundColor: `rgba(124,92,255,${tn / matrixMax * 0.6})` }}>
                        <span className="text-foreground">{tn}</span>
                        <span className="text-[10px] text-muted-foreground">TN</span>
                      </div>
                    </td>
                    <td className="p-3">
                      <div className="w-20 h-20 rounded-lg flex flex-col items-center justify-center text-lg font-bold"
                        style={{ backgroundColor: `rgba(239,68,68,${fp / matrixMax * 0.5})` }}>
                        <span className="text-foreground">{fp}</span>
                        <span className="text-[10px] text-muted-foreground">FP</span>
                      </div>
                    </td>
                  </tr>
                  <tr>
                    <th className="p-3 text-xs text-muted-foreground text-right">生还 (1)</th>
                    <td className="p-3">
                      <div className="w-20 h-20 rounded-lg flex flex-col items-center justify-center text-lg font-bold"
                        style={{ backgroundColor: `rgba(239,68,68,${fn / matrixMax * 0.5})` }}>
                        <span className="text-foreground">{fn}</span>
                        <span className="text-[10px] text-muted-foreground">FN</span>
                      </div>
                    </td>
                    <td className="p-3">
                      <div className="w-20 h-20 rounded-lg flex flex-col items-center justify-center text-lg font-bold"
                        style={{ backgroundColor: `rgba(124,92,255,${tp / matrixMax * 0.6})` }}>
                        <span className="text-foreground">{tp}</span>
                        <span className="text-[10px] text-muted-foreground">TP</span>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="text-xs text-muted-foreground space-y-2 flex-1">
              <p><span className="text-primary font-medium">TN ({tn})</span>: 正确预测为遇难</p>
              <p><span className="text-primary font-medium">TP ({tp})</span>: 正确预测为生还</p>
              <p><span className="text-destructive font-medium">FP ({fp})</span>: 误判为生还（假阳性）</p>
              <p><span className="text-destructive font-medium">FN ({fn})</span>: 漏判生还者（假阴性）</p>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Importance */}
      {sortedFeatures.length > 0 && (
        <section>
          <h2 className="text-xl font-semibold text-foreground mb-4">特征重要性</h2>
          <div className="glass-card p-5">
            <p className="text-xs text-muted-foreground mb-4">每个特征对模型预测的贡献程度，数值越大影响越强</p>
            <div className="space-y-3">
              {sortedFeatures.map((f) => (
                <div key={f.feature} className="flex items-center gap-3">
                  <span className="w-24 text-sm text-foreground text-right shrink-0">{f.feature}</span>
                  <div className="flex-1 h-7 bg-muted/20 rounded-lg overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-primary to-accent rounded-lg transition-all duration-700"
                      style={{ width: `${(f.importance / featureMax) * 100}%` }} />
                  </div>
                  <span className="w-16 text-sm text-accent font-mono text-right">{f.importance.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Group Accuracy */}
      {groupAccuracy && (
        <section>
          <h2 className="text-xl font-semibold text-foreground mb-4">分组准确率</h2>
          <p className="text-xs text-muted-foreground mb-4">按各训练特征分组查看模型预测准确率，了解模型在不同群体上的表现差异</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {([
              { key: 'bySex' as const, label: '按性别', itemLabels: { male: '男性', female: '女性' } },
              { key: 'byPclass' as const, label: '按舱等', itemLabels: { '1': '1等舱', '2': '2等舱', '3': '3等舱' } },
              { key: 'byAgeGroup' as const, label: '按年龄段', itemLabels: { child: '儿童', adult: '成人', elder: '老人' } },
              { key: 'byEmbarked' as const, label: '按登船港口', itemLabels: { S: '南安普顿', C: '瑟堡', Q: '皇后镇' } },
              { key: 'byCabinDeck' as const, label: '按甲板层', itemLabels: {} as Record<string, string> },
              { key: 'byTitle' as const, label: '按头衔', itemLabels: {} as Record<string, string> },
              { key: 'bySibSp' as const, label: '按SibSp', itemLabels: {} as Record<string, string> },
              { key: 'byParch' as const, label: '按Parch', itemLabels: {} as Record<string, string> },
              { key: 'byFare' as const, label: '按票价段', itemLabels: { low: '低价(<15)', medium: '中价(15-50)', high: '高价(≥50)' } },
            ] as const).filter(g => groupAccuracy[g.key] && Object.keys(groupAccuracy[g.key]!).length > 0).map(({ key, label, itemLabels }) => {
              const items = groupAccuracy[key]!;
              return (
                <div key={key} className="glass-card p-5">
                  <h3 className="text-sm font-medium text-foreground mb-3">{label}</h3>
                  <div className="space-y-3">
                    {Object.entries(items).map(([itemLabel, acc]: [string, number]) => (
                      <div key={itemLabel}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-muted-foreground">{(itemLabels as Record<string, string>)[itemLabel] || itemLabel}</span>
                          <span className="text-primary font-medium">{(acc * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-6 bg-muted/20 rounded-lg overflow-hidden">
                          <div className="h-full bg-gradient-to-r from-primary/70 to-accent/70 rounded-lg"
                            style={{ width: `${acc * 100}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Actions */}
      <div className="flex justify-between items-center">
        <Link href="/train" className="px-6 py-3 rounded-xl text-sm font-medium border border-border text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all">
          ← 调整参数重新训练
        </Link>
        <Link href="/predict" className="btn-gradient px-6 py-3 rounded-xl text-sm font-medium">
          试试自定义预测 →
        </Link>
      </div>
    </div>
  );
}
