'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Link from 'next/link';
import { useML, type TrainConfig } from '@/lib/MLContext';

type ModelType = 'logistic_regression' | 'decision_tree' | 'random_forest';

interface Hyperparams {
  C?: number;
  maxIter?: number;
  solver?: string;
  penalty?: string;
  maxDepth?: number | null;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  criterion?: string;
  nEstimators?: number;
}

interface LogEntry {
  message: string;
  type: 'info' | 'success' | 'warning' | 'detail';
  timestamp: string;
}

interface LearningCurveData {
  trainSizes: number[];
  trainScores: number[];
  valScores: number[];
  trainScoresStd?: number[];
  valScoresStd?: number[];
}

interface TreeNode {
  type: 'split' | 'leaf' | 'truncated';
  feature?: string;
  threshold?: number;
  samples?: number;
  prediction?: number;
  value?: number[];
  depth?: number;
  left?: TreeNode;
  right?: TreeNode;
}

interface ModelStructure {
  type: 'logistic_regression' | 'decision_tree' | 'random_forest';
  coefficients?: { feature: string; coefficient: number; absCoefficient: number }[];
  intercept?: number;
  treeData?: TreeNode;
  maxDepth?: number;
  nNodes?: number;
  nLeaves?: number;
  oobCurve?: { nEstimators: number[]; oobErrors: number[] };
  treeDepths?: number[];
  avgDepth?: number;
  maxTreeDepth?: number;
}

const MODEL_INFO: Record<ModelType, { label: string; tag: string; pros: string[]; cons: string[] }> = {
  logistic_regression: {
    label: '逻辑回归',
    tag: '简单可解释',
    pros: ['结果可解释性强', '训练速度快', '不易过拟合'],
    cons: ['只能处理线性关系', '对异常值敏感'],
  },
  decision_tree: {
    label: '决策树',
    tag: '直观易懂',
    pros: ['可视化决策过程', '无需特征缩放', '可处理非线性'],
    cons: ['容易过拟合', '对数据变化敏感'],
  },
  random_forest: {
    label: '随机森林',
    tag: '强大集成',
    pros: ['准确率高', '抗过拟合', '可评估特征重要性'],
    cons: ['训练时间较长', '模型不易解释'],
  },
};

function LearningCurve({ data, lcProgress }: { data: LearningCurveData; lcProgress?: { current: number; total: number } }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const width = 640;
  const height = 320;
  const margin = { top: 30, right: 20, bottom: 40, left: 55 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const n = data.trainSizes.length;
  if (n === 0) return null;

  const allScores = [...data.trainScores, ...data.valScores];
  const yMin = Math.floor(Math.min(...allScores) * 10) / 10;
  const yMax = Math.ceil(Math.max(...allScores) * 10) / 10;
  const xMin = data.trainSizes[0];
  const xMax = data.trainSizes[n - 1];
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 0.1;

  const scaleX = (v: number) => ((v - xMin) / xRange) * plotW;
  const scaleY = (v: number) => plotH - ((v - yMin) / yRange) * plotH;
  const px = (i: number) => margin.left + scaleX(data.trainSizes[i]);
  const pyTrain = (i: number) => margin.top + scaleY(data.trainScores[i]);
  const pyVal = (i: number) => margin.top + scaleY(data.valScores[i]);

  // Catmull-Rom smooth path
  const smoothPath = (pts: [number, number][]): string => {
    if (pts.length < 2) return '';
    let d = `M ${pts[0][0]} ${pts[0][1]}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(0, i - 1)];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[Math.min(pts.length - 1, i + 2)];
      const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
      const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
      const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
      const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2[0]} ${p2[1]}`;
    }
    return d;
  };

  const trainPts: [number, number][] = data.trainSizes.map((_, i) => [px(i), pyTrain(i)]);
  const valPts: [number, number][] = data.trainSizes.map((_, i) => [px(i), pyVal(i)]);

  const trainPath = smoothPath(trainPts);
  const valPath = smoothPath(valPts);

  // Area paths (from line down to bottom)
  const trainAreaPath = trainPath + ` L ${px(n - 1)} ${margin.top + plotH} L ${px(0)} ${margin.top + plotH} Z`;
  const valAreaPath = valPath + ` L ${px(n - 1)} ${margin.top + plotH} L ${px(0)} ${margin.top + plotH} Z`;

  // Std deviation bands
  const trainStd = data.trainScoresStd || [];
  const valStd = data.valScoresStd || [];

  const bandPath = (scores: number[], stds: number[], yFn: (i: number) => number): string => {
    if (stds.length < n) return '';
    const upper: [number, number][] = [];
    const lower: [number, number][] = [];
    for (let i = 0; i < n; i++) {
      const x = px(i);
      upper.push([x, margin.top + scaleY(scores[i] + stds[i])]);
      lower.push([x, margin.top + scaleY(scores[i] - stds[i])]);
    }
    return smoothPath(upper) + ' ' + smoothPath([...lower].reverse())
      .replace('M', 'L') + ' Z';
  };

  const yTicks = 5;
  const yStep = (yMax - yMin) / yTicks;

  // X-axis: show ~6 ticks
  const xTickStep = Math.max(1, Math.floor(n / 6));
  const xTickIndices = Array.from({ length: n }, (_, i) => i).filter(i => i % xTickStep === 0 || i === n - 1);

  // Tooltip position
  const tooltipX = hoverIdx !== null ? px(hoverIdx) : 0;
  const tooltipY = hoverIdx !== null ? Math.min(pyTrain(hoverIdx), pyVal(hoverIdx)) : 0;

  return (
    <div ref={containerRef}>
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-foreground">学习曲线</h4>
        {lcProgress && lcProgress.current < lcProgress.total && (
          <span className="text-xs text-muted-foreground">
            计算中 [{Array.from({ length: 20 }, (_, i) =>
              i < Math.round(lcProgress.current / lcProgress.total * 20) ? '█' : '░'
            ).join('')}] {lcProgress.current}/{lcProgress.total}
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground mb-3">展示模型随训练数据量增加的学习效果。训练分数与验证分数越接近，模型泛化越好；差距大说明过拟合。</p>
      <div className="relative" onMouseLeave={() => setHoverIdx(null)}>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ cursor: 'crosshair' }}>
          <defs>
            <linearGradient id="trainGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#7C5CFF" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#7C5CFF" stopOpacity="0.02" />
            </linearGradient>
            <linearGradient id="valGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#69E7FF" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#69E7FF" stopOpacity="0.02" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {Array.from({ length: yTicks + 1 }, (_, i) => {
            const y = margin.top + scaleY(yMin + i * yStep);
            return <line key={`hg${i}`} x1={margin.left} y1={y} x2={margin.left + plotW} y2={y} stroke="rgba(255,255,255,0.06)" />;
          })}
          {xTickIndices.map(i => (
            <line key={`vg${i}`} x1={px(i)} y1={margin.top} x2={px(i)} y2={margin.top + plotH} stroke="rgba(255,255,255,0.04)" />
          ))}

          {/* Axes */}
          <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + plotH} stroke="rgba(255,255,255,0.15)" />
          <line x1={margin.left} y1={margin.top + plotH} x2={margin.left + plotW} y2={margin.top + plotH} stroke="rgba(255,255,255,0.15)" />

          {/* Y-axis labels */}
          {Array.from({ length: yTicks + 1 }, (_, i) => {
            const val = yMin + i * yStep;
            const y = margin.top + scaleY(val);
            return <text key={`yl${i}`} x={margin.left - 8} y={y + 4} textAnchor="end" fill="rgba(255,255,255,0.45)" fontSize="10">{(val * 100).toFixed(0)}%</text>;
          })}
          {/* X-axis labels */}
          {xTickIndices.map(i => (
            <text key={`xl${i}`} x={px(i)} y={height - 10} textAnchor="middle" fill="rgba(255,255,255,0.45)" fontSize="10">{data.trainSizes[i]}%</text>
          ))}

          {/* Axis titles */}
          <text x={margin.left + plotW / 2} y={height - 1} textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="9">训练样本比例</text>
          <text x={10} y={margin.top + plotH / 2} textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="9" transform={`rotate(-90, 10, ${margin.top + plotH / 2})`}>准确率</text>

          {/* Std deviation bands */}
          {trainStd.length >= n && <path d={bandPath(data.trainScores, trainStd, pyTrain)} fill="#7C5CFF" fillOpacity="0.08" stroke="none" />}
          {valStd.length >= n && <path d={bandPath(data.valScores, valStd, pyVal)} fill="#69E7FF" fillOpacity="0.06" stroke="none" />}

          {/* Area fills */}
          <path d={trainAreaPath} fill="url(#trainGrad)" />
          <path d={valAreaPath} fill="url(#valGrad)" />

          {/* Lines */}
          <path d={trainPath} fill="none" stroke="#7C5CFF" strokeWidth="2.5" strokeLinejoin="round" />
          <path d={valPath} fill="none" stroke="#69E7FF" strokeWidth="2" strokeLinejoin="round" strokeDasharray="8,4" />

          {/* Data points */}
          {data.trainSizes.map((_, i) => (
            <circle key={`td${i}`} cx={px(i)} cy={pyTrain(i)} r={hoverIdx === i ? 5 : 3} fill="#7C5CFF" stroke={hoverIdx === i ? '#fff' : 'none'} strokeWidth={1.5} style={{ transition: 'r 0.15s' }} />
          ))}
          {data.trainSizes.map((_, i) => (
            <circle key={`vd${i}`} cx={px(i)} cy={pyVal(i)} r={hoverIdx === i ? 5 : 3} fill="#69E7FF" stroke={hoverIdx === i ? '#fff' : 'none'} strokeWidth={1.5} style={{ transition: 'r 0.15s' }} />
          ))}

          {/* Legend */}
          <circle cx={margin.left + 12} cy={margin.top + 12} r={4} fill="#7C5CFF" />
          <line x1={margin.left + 20} y1={margin.top + 12} x2={margin.left + 40} y2={margin.top + 12} stroke="#7C5CFF" strokeWidth="2.5" />
          <text x={margin.left + 44} y={margin.top + 16} fill="#7C5CFF" fontSize="10">训练分数</text>
          <circle cx={margin.left + 120} cy={margin.top + 12} r={4} fill="#69E7FF" />
          <line x1={margin.left + 128} y1={margin.top + 12} x2={margin.left + 148} y2={margin.top + 12} stroke="#69E7FF" strokeWidth="2" strokeDasharray="6,3" />
          <text x={margin.left + 152} y={margin.top + 16} fill="#69E7FF" fontSize="10">验证分数</text>

          {/* Crosshair + tooltip on hover */}
          {hoverIdx !== null && (
            <>
              <line x1={tooltipX} y1={margin.top} x2={tooltipX} y2={margin.top + plotH} stroke="rgba(255,255,255,0.25)" strokeWidth="1" strokeDasharray="4,3" />
              <line x1={margin.left} y1={pyTrain(hoverIdx)} x2={margin.left + plotW} y2={pyTrain(hoverIdx)} stroke="#7C5CFF" strokeWidth="0.5" strokeDasharray="3,3" opacity="0.4" />
              <line x1={margin.left} y1={pyVal(hoverIdx)} x2={margin.left + plotW} y2={pyVal(hoverIdx)} stroke="#69E7FF" strokeWidth="0.5" strokeDasharray="3,3" opacity="0.4" />
            </>
          )}

          {/* Invisible hit areas for hover */}
          {data.trainSizes.map((_, i) => (
            <rect key={`hit${i}`} x={px(i) - (plotW / n) / 2} y={margin.top} width={plotW / n} height={plotH}
              fill="transparent" onMouseEnter={() => setHoverIdx(i)} />
          ))}
        </svg>

        {/* Tooltip card */}
        {hoverIdx !== null && (
          <div className="absolute pointer-events-none z-10 bg-[#1a1d2e]/95 border border-border/50 rounded-lg px-3 py-2 shadow-lg backdrop-blur-sm text-xs"
            style={{ left: `min(${(tooltipX / width) * 100}%, calc(100% - 180px))`, top: `${Math.max(0, (tooltipY / height) * 100 - 12)}%`, transform: 'translateX(8px)' }}>
            <p className="text-muted-foreground mb-1">样本比例: <span className="text-foreground font-medium">{data.trainSizes[hoverIdx]}%</span></p>
            <p className="text-[#7C5CFF]">训练: {(data.trainScores[hoverIdx] * 100).toFixed(1)}%{trainStd[hoverIdx] != null ? ` ± ${(trainStd[hoverIdx] * 100).toFixed(1)}%` : ''}</p>
            <p className="text-[#69E7FF]">验证: {(data.valScores[hoverIdx] * 100).toFixed(1)}%{valStd[hoverIdx] != null ? ` ± ${(valStd[hoverIdx] * 100).toFixed(1)}%` : ''}</p>
          </div>
        )}
      </div>
    </div>
  );
}

function LogisticRegressionViz({ data }: { data: ModelStructure }) {
  const coeffs = data.coefficients || [];
  if (coeffs.length === 0) return null;
  const maxAbs = Math.max(...coeffs.map(c => c.absCoefficient));

  return (
    <div>
      <h4 className="text-sm font-medium text-foreground mb-2">特征权重分析</h4>
      <p className="text-xs text-muted-foreground mb-3">逻辑回归中，正权重提高生还概率，负权重降低生还概率。绝对值越大，影响越强。</p>
      <div className="space-y-2">
        {coeffs.map((c, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-20 text-right truncate">{c.feature}</span>
            <div className="flex-1 h-5 relative">
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white/10" />
              {c.coefficient >= 0 ? (
                <div className="absolute left-1/2 top-0 h-5 rounded-r bg-primary/60 transition-all"
                  style={{ width: `${(c.absCoefficient / maxAbs) * 48}%` }}>
                  <span className="text-[9px] text-primary-foreground pl-1 leading-5">{c.coefficient.toFixed(3)}</span>
                </div>
              ) : (
                <div className="absolute right-1/2 top-0 h-5 rounded-l bg-accent/60 transition-all"
                  style={{ width: `${(c.absCoefficient / maxAbs) * 48}%` }}>
                  <span className="text-[9px] text-accent pr-1 leading-5 float-right">{c.coefficient.toFixed(3)}</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      {data.intercept !== undefined && (
        <p className="text-xs text-muted-foreground mt-3">截距 (intercept): <span className="text-accent font-mono">{data.intercept.toFixed(4)}</span></p>
      )}
    </div>
  );
}

function TreeNodeViz({ node, maxDepth = 3 }: { node: TreeNode; maxDepth?: number }) {
  if (node.type === 'truncated') {
    return (
      <div className="text-xs text-muted-foreground/50 italic text-center py-1">
        ... (深度 {node.depth}，已截断)
      </div>
    );
  }

  if (node.type === 'leaf') {
    return (
      <div className="flex flex-col items-center">
        <div className="px-3 py-1.5 rounded-lg bg-success/20 border border-success/30 text-center min-w-[60px]">
          <span className="text-xs font-medium text-success">
            生还={node.prediction === 1 ? '是' : '否'}
          </span>
          <span className="text-[10px] text-muted-foreground block">
            {node.samples}人
          </span>
        </div>
      </div>
    );
  }

  const currentDepth = node.depth || 0;
  const showChildren = currentDepth < maxDepth;

  return (
    <div className="flex flex-col items-center">
      <div className="px-3 py-1.5 rounded-lg bg-primary/20 border border-primary/40 text-center min-w-[80px]">
        <span className="text-xs font-medium text-foreground">{node.feature}</span>
        <span className="text-[10px] text-muted-foreground block">
          ≤ {node.threshold}{node.samples !== undefined ? ` (${node.samples}人)` : ''}
        </span>
      </div>
      {showChildren && (node.left || node.right) && (
        <div className="flex items-start gap-0 mt-1">
          {/* Left branch (<= threshold) */}
          <div className="flex flex-col items-center">
            <div className="w-px h-3 bg-primary/30" />
            <span className="text-[9px] text-primary mb-0.5">是</span>
            {node.left && <TreeNodeViz node={node.left} maxDepth={maxDepth} />}
          </div>
          {/* Right branch (> threshold) */}
          <div className="flex flex-col items-center ml-4">
            <div className="w-px h-3 bg-accent/30" />
            <span className="text-[9px] text-accent mb-0.5">否</span>
            {node.right && <TreeNodeViz node={node.right} maxDepth={maxDepth} />}
          </div>
        </div>
      )}
      {!showChildren && (
        <div className="text-[10px] text-muted-foreground/50 mt-1">...</div>
      )}
    </div>
  );
}

function DecisionTreeViz({ data }: { data: ModelStructure }) {
  return (
    <div>
      <h4 className="text-sm font-medium text-foreground mb-2">决策树结构</h4>
      <p className="text-xs text-muted-foreground mb-3">
        从根节点开始，每个节点按特征值分裂。紫色节点=分裂条件，绿色节点=预测结果。
        {data.nNodes && <span> 共 {data.nNodes} 个节点、{data.nLeaves} 个叶子。</span>}
      </p>
      <div className="overflow-x-auto py-2">
        {data.treeData && <TreeNodeViz node={data.treeData} maxDepth={3} />}
      </div>
    </div>
  );
}

function RandomForestViz({ data }: { data: ModelStructure }) {
  const oobCurve = data.oobCurve;
  const treeDepths = data.treeDepths || [];

  return (
    <div>
      <h4 className="text-sm font-medium text-foreground mb-2">随机森林分析</h4>
      <p className="text-xs text-muted-foreground mb-3">
        随机森林由多棵决策树投票决定结果。OOB 错误率越低，模型越好。
        {data.avgDepth && <span> 平均树深度 {data.avgDepth}，最大深度 {data.maxTreeDepth}。</span>}
      </p>

      {/* OOB Curve */}
      {oobCurve && oobCurve.oobErrors.length > 0 && (
        <div className="mb-4">
          <h5 className="text-xs font-medium text-muted-foreground mb-2">OOB 错误率曲线</h5>
          <p className="text-[10px] text-muted-foreground/60 mb-2">袋外(OOB)错误率随树数量增加的变化趋势，趋于平稳说明树数量足够</p>
          <svg viewBox="0 0 400 180" className="w-full max-w-md">
            {(() => {
              const margin = { top: 15, right: 15, bottom: 30, left: 45 };
              const w = 400 - margin.left - margin.right;
              const h = 180 - margin.top - margin.bottom;
              const maxErr = Math.max(...oobCurve.oobErrors);
              const minErr = Math.min(...oobCurve.oobErrors.filter(e => e > 0));
              const errRange = maxErr - Math.max(0, minErr - 0.05);
              const xStep = oobCurve.nEstimators.length > 1 ? w / (oobCurve.nEstimators.length - 1) : 0;

              const points = oobCurve.oobErrors.map((err, i) =>
                `${margin.left + i * xStep},${margin.top + h - ((err - Math.max(0, minErr - 0.05)) / errRange) * h}`
              ).join(' ');

              return (
                <>
                  <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + h} stroke="rgba(255,255,255,0.2)" />
                  <line x1={margin.left} y1={margin.top + h} x2={margin.left + w} y2={margin.top + h} stroke="rgba(255,255,255,0.2)" />
                  <polyline points={points} fill="none" stroke="#62FAD3" strokeWidth="2" strokeLinejoin="round" />
                  {oobCurve.oobErrors.map((err, i) => (
                    <circle key={i} cx={margin.left + i * xStep} cy={margin.top + h - ((err - Math.max(0, minErr - 0.05)) / errRange) * h} r="2.5" fill="#62FAD3" />
                  ))}
                  <text x={200} y={175} textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9">树数量</text>
                  <text x={5} y={95} textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9" transform="rotate(-90, 5, 95)">错误率</text>
                </>
              );
            })()}
          </svg>
        </div>
      )}

      {/* Tree Depth Distribution */}
      {treeDepths.length > 0 && (
        <div>
          <h5 className="text-xs font-medium text-muted-foreground mb-2">树深度分布</h5>
          <div className="flex items-end gap-1 h-16">
            {(() => {
              const depthCounts: Record<number, number> = {};
              treeDepths.forEach(d => { depthCounts[d] = (depthCounts[d] || 0) + 1; });
              const maxCount = Math.max(...Object.values(depthCounts));
              return Object.entries(depthCounts).sort(([a], [b]) => Number(a) - Number(b)).map(([depth, count]) => (
                <div key={depth} className="flex flex-col items-center flex-1" title={`深度${depth}: ${count}棵树`}>
                  <div className="w-full bg-accent/50 rounded-t transition-all"
                    style={{ height: `${(count / maxCount) * 56}px` }} />
                  <span className="text-[9px] text-muted-foreground">{depth}</span>
                </div>
              ));
            })()}
          </div>
          <p className="text-[10px] text-muted-foreground/60 mt-1">各深度的树数量</p>
        </div>
      )}
    </div>
  );
}

function LogisticRegressionDiagram() {
  return (
    <div className="bg-muted/30 rounded-xl p-4 border border-border/30">
      <h4 className="text-sm font-medium text-foreground mb-2">模型结构简化示意图</h4>
      <p className="text-xs text-muted-foreground mb-3">逻辑回归将输入特征加权求和，通过 Sigmoid 函数映射为 0~1 之间的概率</p>
      <svg viewBox="0 0 520 140" className="w-full max-w-xl">
        {/* Input features */}
        <text x="10" y="30" fill="#7C5CFF" fontSize="10" fontWeight="600">输入特征</text>
        {['Pclass', 'Sex', 'Age', 'Fare', '...'].map((f, i) => (
          <g key={i}>
            <rect x="10" y={38 + i * 20} width="70" height="16" rx="4" fill="#7C5CFF" fillOpacity="0.15" stroke="#7C5CFF" strokeWidth="0.8" />
            <text x="45" y={49 + i * 20} textAnchor="middle" fill="#7C5CFF" fontSize="8">{f}</text>
          </g>
        ))}

        {/* Weight arrows */}
        {['w₁', 'w₂', 'w₃', 'w₄', '...'].map((w, i) => (
          <g key={i}>
            <line x1="82" y1={46 + i * 20} x2="155" y2="70" stroke="rgba(255,255,255,0.25)" strokeWidth="0.8" />
            <text x="118" y={54 + i * 14} textAnchor="middle" fill="#69E7FF" fontSize="8">{w}</text>
          </g>
        ))}

        {/* Summation node */}
        <circle cx="175" cy="70" r="20" fill="#7C5CFF" fillOpacity="0.2" stroke="#7C5CFF" strokeWidth="1.2" />
        <text x="175" y="74" textAnchor="middle" fill="#7C5CFF" fontSize="14" fontWeight="700">Σ</text>
        <text x="175" y="105" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="8">加权求和</text>
        <text x="175" y="115" textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="7">z = w₁x₁ + w₂x₂ + ... + b</text>

        {/* Arrow to sigmoid */}
        <line x1="196" y1="70" x2="255" y2="70" stroke="rgba(255,255,255,0.3)" strokeWidth="1" markerEnd="url(#arrowLR)" />
        <defs><marker id="arrowLR" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="rgba(255,255,255,0.4)" /></marker></defs>

        {/* Sigmoid function box */}
        <rect x="258" y="45" width="80" height="50" rx="8" fill="#69E7FF" fillOpacity="0.1" stroke="#69E7FF" strokeWidth="1" />
        <text x="298" y="66" textAnchor="middle" fill="#69E7FF" fontSize="10" fontWeight="600">Sigmoid</text>
        <text x="298" y="80" textAnchor="middle" fill="#69E7FF" fontSize="7" opacity="0.7">σ(z) = 1/(1+e⁻ᶻ)</text>

        {/* Arrow to output */}
        <line x1="340" y1="70" x2="390" y2="70" stroke="rgba(255,255,255,0.3)" strokeWidth="1" markerEnd="url(#arrowLR)" />

        {/* Output probability */}
        <rect x="393" y="45" width="110" height="50" rx="8" fill="#62FAD3" fillOpacity="0.1" stroke="#62FAD3" strokeWidth="1" />
        <text x="448" y="66" textAnchor="middle" fill="#62FAD3" fontSize="10" fontWeight="600">生还概率</text>
        <text x="448" y="82" textAnchor="middle" fill="#62FAD3" fontSize="9">P = 0 ~ 100%</text>

        {/* Threshold line */}
        <line x1="448" y1="97" x2="448" y2="120" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" strokeDasharray="3,2" />
        <text x="425" y="132" textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="7">P ≥ 50%</text>
        <text x="475" y="132" textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="7">P &lt; 50%</text>
        <text x="410" y="132" textAnchor="end" fill="#62FAD3" fontSize="7">生还</text>
        <text x="490" y="132" textAnchor="start" fill="#F87171" fontSize="7">遇难</text>
      </svg>
    </div>
  );
}

function DecisionTreeDiagram() {
  return (
    <div className="bg-muted/30 rounded-xl p-4 border border-border/30">
      <h4 className="text-sm font-medium text-foreground mb-2">模型结构简化示意图</h4>
      <p className="text-xs text-muted-foreground mb-3">决策树从根节点开始，按特征条件逐层分裂，叶子节点给出预测结果</p>
      <svg viewBox="0 0 520 180" className="w-full max-w-xl">
        {/* Root node */}
        <rect x="185" y="8" width="150" height="36" rx="8" fill="#7C5CFF" fillOpacity="0.2" stroke="#7C5CFF" strokeWidth="1" />
        <text x="260" y="24" textAnchor="middle" fill="#7C5CFF" fontSize="9" fontWeight="600">年龄 ≤ 12?</text>
        <text x="260" y="36" textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="7">根节点 (Root)</text>

        {/* Left branch - Yes */}
        <line x1="220" y1="44" x2="120" y2="72" stroke="rgba(255,255,255,0.25)" strokeWidth="1" />
        <text x="160" y="58" fill="#62FAD3" fontSize="8" fontWeight="600">是</text>

        {/* Right branch - No */}
        <line x1="300" y1="44" x2="400" y2="72" stroke="rgba(255,255,255,0.25)" strokeWidth="1" />
        <text x="355" y="58" fill="#69E7FF" fontSize="8" fontWeight="600">否</text>

        {/* Level 1 Left - Leaf */}
        <rect x="45" y="74" width="150" height="36" rx="8" fill="#7C5CFF" fillOpacity="0.15" stroke="#7C5CFF" strokeWidth="0.8" />
        <text x="120" y="90" textAnchor="middle" fill="#7C5CFF" fontSize="9" fontWeight="600">性别 = 女性?</text>
        <text x="120" y="102" textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="7">内部节点</text>

        {/* Level 1 Right - Internal */}
        <rect x="325" y="74" width="150" height="36" rx="8" fill="#7C5CFF" fillOpacity="0.15" stroke="#7C5CFF" strokeWidth="0.8" />
        <text x="400" y="90" textAnchor="middle" fill="#7C5CFF" fontSize="9" fontWeight="600">舱等 ≤ 2?</text>
        <text x="400" y="102" textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="7">内部节点</text>

        {/* Level 2 leaves from left */}
        <line x1="90" y1="110" x2="50" y2="138" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
        <text x="65" y="126" fill="#62FAD3" fontSize="7">是</text>
        <line x1="150" y1="110" x2="185" y2="138" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
        <text x="172" y="126" fill="#69E7FF" fontSize="7">否</text>

        {/* Level 2 leaves from right */}
        <line x1="370" y1="110" x2="335" y2="138" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
        <text x="348" y="126" fill="#62FAD3" fontSize="7">是</text>
        <line x1="430" y1="110" x2="465" y2="138" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
        <text x="452" y="126" fill="#69E7FF" fontSize="7">否</text>

        {/* Leaf nodes */}
        <rect x="15" y="140" width="70" height="28" rx="6" fill="#62FAD3" fillOpacity="0.15" stroke="#62FAD3" strokeWidth="0.8" />
        <text x="50" y="157" textAnchor="middle" fill="#62FAD3" fontSize="9" fontWeight="600">生还</text>

        <rect x="150" y="140" width="70" height="28" rx="6" fill="#F87171" fillOpacity="0.15" stroke="#F87171" strokeWidth="0.8" />
        <text x="185" y="157" textAnchor="middle" fill="#F87171" fontSize="9" fontWeight="600">遇难</text>

        <rect x="300" y="140" width="70" height="28" rx="6" fill="#62FAD3" fillOpacity="0.15" stroke="#62FAD3" strokeWidth="0.8" />
        <text x="335" y="157" textAnchor="middle" fill="#62FAD3" fontSize="9" fontWeight="600">生还</text>

        <rect x="430" y="140" width="70" height="28" rx="6" fill="#F87171" fillOpacity="0.15" stroke="#F87171" strokeWidth="0.8" />
        <text x="465" y="157" textAnchor="middle" fill="#F87171" fontSize="9" fontWeight="600">遇难</text>

        {/* Legend */}
        <text x="260" y="175" textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="7">紫色 = 分裂条件 | 绿色 = 预测生还 | 红色 = 预测遇难</text>
      </svg>
    </div>
  );
}

export default function TrainPage() {
  const { cleanConfig, addVersion, updateStepCompleted } = useML();
  const [modelType, setModelType] = useState<ModelType>('logistic_regression');
  const [hyperparams, setHyperparams] = useState<Hyperparams>({ C: 1.0, maxIter: 1000, solver: 'lbfgs', penalty: 'l2' });
  const [testSize, setTestSize] = useState(0.3);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<Record<string, unknown> | null>(null);
  const [completed, setCompleted] = useState(false);
  const [learningCurve, setLearningCurve] = useState<LearningCurveData | null>(null);
  const [lcProgress, setLcProgress] = useState<{ current: number; total: number } | null>(null);
  const [modelStructure, setModelStructure] = useState<ModelStructure | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  };

  const handleTrain = useCallback(async () => {
    if (training) return;
    if (!cleanConfig) {
      alert('请先完成数据清洗');
      return;
    }

    setTraining(true);
    setLogs([]);
    setTrainResult(null);
    setCompleted(false);
    setLearningCurve(null);
    setLcProgress(null);
    setModelStructure(null);

    const config: TrainConfig = {
      cleanConfig: cleanConfig!,
      modelType,
      hyperparams: hyperparams as Record<string, number | string | null>,
      testSize,
    };

    try {
      const response = await fetch('/api/model/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(line.slice(6));
              if (parsed.type === 'log') {
                const logLevel = parsed.level || parsed.logType || 'info';
                const ts = parsed.timestamp || new Date().toLocaleTimeString('en-GB', { hour12: false });
                setLogs(prev => [...prev, { message: parsed.message, type: logLevel as 'info' | 'success' | 'warning' | 'detail', timestamp: ts }]);
                scrollToBottom();
              } else if (parsed.type === 'progress') {
                const ts = parsed.timestamp || new Date().toLocaleTimeString('en-GB', { hour12: false });
                setLogs(prev => [...prev, { message: parsed.message, type: parsed.level || 'info' as const, timestamp: ts }]);
                scrollToBottom();
              } else if (parsed.type === 'lc_progress') {
                setLcProgress({ current: parsed.current, total: parsed.total });
              } else if (parsed.type === 'learning_curve') {
                setLearningCurve(parsed.data);
                setLcProgress(null);
              } else if (parsed.type === 'result') {
                setTrainResult(parsed.data);
                setLogs(prev => [...prev, { message: '训练流程全部完成!', type: 'success' as const, timestamp: new Date().toLocaleTimeString('en-GB', { hour12: false }) }]);
                setCompleted(true);
                addVersion(config, parsed.data);
                updateStepCompleted('train');
                scrollToBottom();
              } else if (parsed.type === 'error') {
                setLogs(prev => [...prev, { message: `错误: ${parsed.message}`, type: 'warning' as const, timestamp: new Date().toLocaleTimeString('en-GB', { hour12: false }) }]);
                scrollToBottom();
              } else if (parsed.type === 'model_structure') {
                setModelStructure(parsed.data);
              }
            } catch {
              // skip invalid JSON
            }
          }
        }
      }
    } catch (err) {
      setLogs(prev => [...prev, { message: `连接失败: ${err}`, type: 'warning' as const, timestamp: new Date().toLocaleTimeString('en-GB', { hour12: false }) }]);
    } finally {
      setTraining(false);
    }
  }, [training, cleanConfig, modelType, hyperparams, testSize, addVersion, updateStepCompleted]);

  // Auto-switch away from logistic_regression if disabled
  const [prevHasUnfilled, setPrevHasUnfilled] = useState(false);
  useEffect(() => {
    const currentUnfilled = !!cleanConfig?.hasUnfilledMissing;
    if (currentUnfilled && !prevHasUnfilled && modelType === 'logistic_regression') {
      setModelType('decision_tree');
      setTrainResult(null);
      setCompleted(false);
      setLearningCurve(null);
      setModelStructure(null);
    }
    setPrevHasUnfilled(currentUnfilled);
  }, [cleanConfig?.hasUnfilledMissing, modelType, prevHasUnfilled]);

  const handleModelChange = (type: ModelType) => {
    if (type === 'logistic_regression' && cleanConfig?.hasUnfilledMissing) return;
    setModelType(type);
    setTrainResult(null);
    setCompleted(false);
    setLearningCurve(null);
    setModelStructure(null);
    switch (type) {
      case 'logistic_regression':
        setHyperparams({ C: 1.0, maxIter: 1000, solver: 'lbfgs', penalty: 'l2' });
        break;
      case 'decision_tree':
        setHyperparams({ maxDepth: 5, minSamplesSplit: 2, minSamplesLeaf: 1, criterion: 'gini' });
        break;
      case 'random_forest':
        setHyperparams({ nEstimators: 100, maxDepth: null, minSamplesSplit: 2 });
        break;
    }
  };

  const hasUnfilledMissing = cleanConfig?.hasUnfilledMissing === true;
  const logisticRegressionDisabled = hasUnfilledMissing;

  const testSamples = Math.round(891 * testSize);
  const trainSamples = 891 - testSamples;

  return (
    <div className="space-y-8 p-5">
      <div>
        <h1 className="text-3xl font-bold text-foreground">模型训练</h1>
        <p className="text-muted-foreground mt-2">选择模型，配置参数，观察训练过程</p>
        {hasUnfilledMissing && (
          <div className="mt-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 text-sm">
            当前训练数据中存在未填充的缺失值，逻辑回归不支持缺失值，已自动禁用。决策树和随机森林可以处理缺失值。
          </div>
        )}
      </div>

      {/* Model Selection */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">选择模型</h2>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {(Object.entries(MODEL_INFO) as [ModelType, typeof MODEL_INFO[ModelType]][]).map(([type, info]) => {
            const isDisabled = type === 'logistic_regression' && logisticRegressionDisabled;
            return (
            <button
              key={type}
              onClick={() => !isDisabled && handleModelChange(type)}
              disabled={isDisabled}
              className={`glass-card p-5 text-left transition-all ${
                isDisabled
                  ? 'opacity-40 cursor-not-allowed border-border/50'
                  : modelType === type
                    ? 'border-primary ring-2 ring-primary/30 shadow-[0_0_15px_rgba(124,92,255,0.2)] cursor-pointer'
                    : 'hover:border-primary/30 cursor-pointer'
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold text-foreground">{info.label}</h3>
                <span className="text-xs px-2 py-0.5 rounded-full bg-primary/20 text-primary">{info.tag}</span>
              </div>
              <div className="space-y-2 text-xs">
                <div>
                  <span className="text-success">优点：</span>
                  <span className="text-muted-foreground">{info.pros.join('、')}</span>
                </div>
                <div>
                  <span className="text-destructive">缺点：</span>
                  <span className="text-muted-foreground">{info.cons.join('、')}</span>
                </div>
              </div>
              {modelType === type && !isDisabled && (
                <div className="mt-3 text-xs text-primary font-medium">✓ 已选择</div>
              )}
              {isDisabled && (
                <div className="mt-3 text-xs text-amber-400 font-medium">不支持缺失值</div>
              )}
            </button>
            );
          })}
        </div>
      </section>

      {/* Model Structure Concept Diagram */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">模型原理示意</h2>
        {modelType === 'logistic_regression' && <LogisticRegressionDiagram />}
        {modelType === 'decision_tree' && <DecisionTreeDiagram />}
        {modelType === 'random_forest' && (
          <div className="bg-muted/30 rounded-xl p-4 border border-border/30">
            <h4 className="text-sm font-medium text-foreground mb-2">模型结构简化示意图</h4>
            <p className="text-xs text-muted-foreground">随机森林由多棵决策树组成，每棵树独立预测后投票决定最终结果。单棵树的结构与决策树相同，因此可参考决策树示意图理解基本单元。</p>
          </div>
        )}
      </section>

      {/* Hyperparameters */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">超参数配置</h2>
        <div className="glass-card p-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {modelType === 'logistic_regression' && (
              <>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">
                    正则化强度 C: <span className="text-accent font-mono">{hyperparams.C?.toFixed(2)}</span>
                  </label>
                  <input type="range" min="-2" max="2" step="0.1"
                    value={Math.log10(hyperparams.C || 1)}
                    onChange={e => setHyperparams(p => ({ ...p, C: Math.pow(10, parseFloat(e.target.value)) }))}
                    className="w-full accent-primary" />
                  <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                    <span>0.01 (强正则化)</span>
                    <span>100 (弱正则化)</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">控制正则化强度。C 值越大，正则化越弱，模型越容易拟合训练数据（可能过拟合）；C 值越小，正则化越强，模型越平滑</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">最大迭代次数</label>
                  <input type="number" value={hyperparams.maxIter || 1000} min={100} max={10000} step={100}
                    onChange={e => setHyperparams(p => ({ ...p, maxIter: parseInt(e.target.value) || 1000 }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
                  <p className="text-xs text-muted-foreground mt-1">优化算法的最大迭代次数。若训练出现收敛警告，可增大此值</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">优化算法 (solver)</label>
                  <select value={hyperparams.solver || 'lbfgs'}
                    onChange={e => {
                      const newSolver = e.target.value;
                      setHyperparams(p => {
                        // Auto-adjust penalty for solver compatibility
                        let newPenalty = p.penalty || 'l2';
                        if (newSolver === 'liblinear') {
                          // liblinear supports l1 and l2
                        } else if (newSolver === 'saga') {
                          // saga supports l1, l2, and none
                        } else {
                          // lbfgs only supports l2 and none
                          if (newPenalty === 'l1') newPenalty = 'l2';
                        }
                        return { ...p, solver: newSolver, penalty: newPenalty };
                      });
                    }}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                  >
                    <option value="lbfgs">L-BFGS (默认，适合小数据集)</option>
                    <option value="liblinear">Liblinear (适合小数据集，支持L1)</option>
                    <option value="saga">SAGA (适合大数据集，支持所有惩罚)</option>
                  </select>
                  <p className="text-xs text-muted-foreground mt-1">求解最优权重的优化算法。L-BFGS 是默认选择；Liblinear 支持 L1 正则化；SAGA 适用于大数据集</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">正则化类型 (penalty)</label>
                  <select value={hyperparams.penalty || 'l2'}
                    onChange={e => setHyperparams(p => ({ ...p, penalty: e.target.value }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                  >
                    <option value="l2">L2 正则化 (Ridge，默认)</option>
                    {(hyperparams.solver === 'liblinear' || hyperparams.solver === 'saga') && (
                      <option value="l1">L1 正则化 (Lasso，特征选择)</option>
                    )}
                    {(hyperparams.solver === 'lbfgs' || hyperparams.solver === 'saga') && (
                      <option value="none">无正则化</option>
                    )}
                  </select>
                  <p className="text-xs text-muted-foreground mt-1">L2 正则化使权重均匀分散；L1 正则化使不重要特征权重归零（自动特征选择）；无正则化可能过拟合</p>
                </div>
              </>
            )}
            {modelType === 'decision_tree' && (
              <>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">最大深度 (max_depth)</label>
                  <select value={hyperparams.maxDepth === null ? 'none' : (hyperparams.maxDepth ?? 5)}
                    onChange={e => setHyperparams(p => ({ ...p, maxDepth: e.target.value === 'none' ? null : parseInt(e.target.value) }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                  >
                    <option value="none">不限 (None)</option>
                    {Array.from({ length: 20 }, (_, i) => (
                      <option key={i + 1} value={i + 1}>{i + 1}</option>
                    ))}
                  </select>
                  <p className="text-xs text-muted-foreground mt-1">树的最大深度。深度越大越容易过拟合，设为&ldquo;不限&rdquo;则树会完全生长直到所有叶子纯净</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">最小分裂样本数 (min_samples_split)</label>
                  <input type="number" value={hyperparams.minSamplesSplit || 2} min={2} max={50}
                    onChange={e => setHyperparams(p => ({ ...p, minSamplesSplit: parseInt(e.target.value) || 2 }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
                  <p className="text-xs text-muted-foreground mt-1">一个节点至少需要这么多样本才允许继续分裂。增大此值可防止过拟合</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">叶节点最少样本数 (min_samples_leaf)</label>
                  <input type="number" value={hyperparams.minSamplesLeaf || 1} min={1} max={50}
                    onChange={e => setHyperparams(p => ({ ...p, minSamplesLeaf: parseInt(e.target.value) || 1 }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
                  <p className="text-xs text-muted-foreground mt-1">叶子节点至少包含的样本数。增大此值使模型更保守，避免学习噪声</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">分裂标准 (criterion)</label>
                  <select value={hyperparams.criterion || 'gini'}
                    onChange={e => setHyperparams(p => ({ ...p, criterion: e.target.value }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                  >
                    <option value="gini">基尼系数 (Gini，默认)</option>
                    <option value="entropy">信息熵 (Entropy)</option>
                  </select>
                  <p className="text-xs text-muted-foreground mt-1">衡量节点纯度的标准。Gini 计算更快，Entropy 更倾向于产生平衡的树</p>
                </div>
              </>
            )}
            {modelType === 'random_forest' && (
              <>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">
                    树的数量 (n_estimators): <span className="text-accent font-mono">{hyperparams.nEstimators}</span>
                  </label>
                  <input type="range" min={10} max={200} step={10}
                    value={hyperparams.nEstimators || 100}
                    onChange={e => setHyperparams(p => ({ ...p, nEstimators: parseInt(e.target.value) }))}
                    className="w-full accent-primary" />
                  <p className="text-xs text-muted-foreground mt-1">森林中树的数量。越多通常越准确，但训练时间越长</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">最大深度 (max_depth)</label>
                  <select value={hyperparams.maxDepth ?? 'none'}
                    onChange={e => setHyperparams(p => ({ ...p, maxDepth: e.target.value === 'none' ? null : parseInt(e.target.value) }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                  >
                    <option value="none">不限 (None)</option>
                    {Array.from({ length: 20 }, (_, i) => (
                      <option key={i + 1} value={i + 1}>{i + 1}</option>
                    ))}
                  </select>
                  <p className="text-xs text-muted-foreground mt-1">每棵树的最大深度。随机森林中通常不限制深度，因为集成本身能抗过拟合</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">最小分裂样本数 (min_samples_split)</label>
                  <input type="number" value={hyperparams.minSamplesSplit || 2} min={2} max={50}
                    onChange={e => setHyperparams(p => ({ ...p, minSamplesSplit: parseInt(e.target.value) || 2 }))}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30" />
                  <p className="text-xs text-muted-foreground mt-1">一个节点至少需要这么多样本才允许继续分裂</p>
                </div>
              </>
            )}
          </div>
        </div>
      </section>

      {/* Train/Test Split */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">训练集 / 测试集比例</h2>
        <div className="glass-card p-5">
          <div className="flex items-center gap-6">
            <div className="flex-1">
              <label className="text-sm font-medium text-foreground block mb-2">
                测试集比例: <span className="text-accent font-mono">{(testSize * 100).toFixed(0)}%</span>
              </label>
              <input type="range" min={0.1} max={0.5} step={0.05}
                value={testSize}
                onChange={e => { setTestSize(parseFloat(e.target.value)); setCompleted(false); }}
                className="w-full accent-primary" />
            </div>
            <div className="text-sm text-muted-foreground space-y-1 min-w-[160px]">
              <p>训练集: <span className="text-primary font-medium">{trainSamples}</span> 条</p>
              <p>测试集: <span className="text-accent font-medium">{testSamples}</span> 条</p>
            </div>
          </div>
        </div>
      </section>

      {/* Training Log */}
      <section>
        <h2 className="text-xl font-semibold text-foreground mb-4">训练日志</h2>
        <div ref={logRef}
          className="bg-[#0a0c14] border border-border/50 rounded-xl p-4 h-80 overflow-y-auto font-mono text-xs space-y-0.5 custom-scrollbar">
          {logs.length === 0 ? (
            <p className="text-muted-foreground/40 py-2">等待开始训练...</p>
          ) : (
            logs.map((log, i) => {
              const msg = log.message;
              const isDetail = msg.startsWith('  ');
              const icon = msg.includes('✓') ? '' : msg.includes('▶') ? '' : msg.includes('⚡') ? '' : msg.includes('→') ? '' : '';
              return (
                <p key={i} className={
                  `leading-relaxed ${isDetail ? 'pl-3' : ''} ` +
                  (msg.includes('✓') ? 'text-[#62FAD3]' :
                   msg.includes('⚠') || msg.includes('错误') ? 'text-[#FF6B6B]' :
                   msg.includes('⚡') ? 'text-[#FFB347]' :
                   msg.includes('▶') ? 'text-[#69E7FF]' :
                   isDetail ? 'text-muted-foreground/60' :
                   'text-muted-foreground/80')
                }>
                  <span className="text-muted-foreground/30 mr-2 select-none">[{log.timestamp || new Date().toLocaleTimeString('en-GB', {hour12: false})}]</span>
                  {msg}
                </p>
              );
            })
          )}
        </div>
      </section>

      {/* Learning Curve Visualization */}
      {learningCurve && (
        <section className="glass-card p-5">
          <LearningCurve data={learningCurve} lcProgress={lcProgress || undefined} />
        </section>
      )}

      {/* Model Structure Visualization */}
      {modelStructure && (
        <section className="glass-card p-5">
          {modelStructure.type === 'logistic_regression' && <LogisticRegressionViz data={modelStructure} />}
          {modelStructure.type === 'decision_tree' && <DecisionTreeViz data={modelStructure} />}
          {modelStructure.type === 'random_forest' && <RandomForestViz data={modelStructure} />}
        </section>
      )}

      {/* Quick Result Preview */}
      {trainResult && (
        <div className="glass-card p-5 border border-success/30">
          <h3 className="text-success font-medium mb-3">训练完成 - 快速预览</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-primary">{(((trainResult as Record<string, unknown>).metrics as Record<string, number>).accuracy * 100).toFixed(1)}%</p>
              <p className="text-xs text-muted-foreground">准确率</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-accent">{(((trainResult as Record<string, unknown>).metrics as Record<string, number>).precision * 100).toFixed(1)}%</p>
              <p className="text-xs text-muted-foreground">精确率</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-success">{(((trainResult as Record<string, unknown>).metrics as Record<string, number>).recall * 100).toFixed(1)}%</p>
              <p className="text-xs text-muted-foreground">召回率</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground">{(((trainResult as Record<string, unknown>).metrics as Record<string, number>).f1 * 100).toFixed(1)}%</p>
              <p className="text-xs text-muted-foreground">F1分数</p>
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-between items-center">
        <Link href="/clean" className="px-6 py-3 rounded-xl text-sm font-medium border border-border text-muted-foreground hover:text-foreground hover:border-primary/30 transition-all">
          ← 返回数据清洗
        </Link>
        <div className="flex gap-3">
          <button onClick={handleTrain} disabled={training || !cleanConfig}
            className="btn-gradient px-6 py-3 rounded-xl text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed">
            {training ? (
              <span className="flex items-center gap-2">
                <span className="w-4 h-4 border-2 border-foreground border-t-transparent rounded-full animate-spin" />
                训练中...
              </span>
            ) : '开始训练'}
          </button>
          {completed && (
            <Link href="/evaluate" className="btn-gradient px-6 py-3 rounded-xl text-sm font-medium">
              查看评估结果 →
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
