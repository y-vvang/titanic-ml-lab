'use client';

import { useCallback } from 'react';
import Link from 'next/link';
import { BarChart3, Database, Cpu, CheckCircle, Target, Lightbulb, ArrowRight, Download } from 'lucide-react';

export default function HomePage() {
  const handleDownload = useCallback(() => {
    const link = document.createElement('a');
    link.href = '/dataset.csv';
    link.download = 'titanic_dataset.csv';
    link.click();
  }, []);

  return (
    <main className="max-w-7xl mx-auto w-full px-6 py-12">
      {/* Hero Section */}
      <section className="text-center mb-20 relative">
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-[600px] h-[600px] rounded-full bg-[#7C5CFF]/8 blur-[120px]" />
        </div>
        <div className="relative z-10">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-[#7C5CFF] via-[#69E7FF] to-[#62FAD3] bg-clip-text text-transparent">
            Titanic ML Lab
          </h1>
          <p className="text-2xl text-[#9AA7C7] mb-4">机器学习从零开始</p>
          <p className="text-sm text-[#9AA7C7]/50 max-w-2xl mx-auto mb-4">
            一步步理解机器学习的核心思想
          </p>
          <div className="flex items-center justify-center gap-4 mb-7">
            <div className="flex items-center gap-2 text-sm text-[#9AA7C7]">
              <Database className="w-4 h-4 text-[#69E7FF]" />
              <span>891 条真实数据</span>
            </div>
            <div className="w-1 h-1 rounded-full bg-[#9AA7C7]/40" />
            <div className="flex items-center gap-2 text-sm text-[#9AA7C7]">
              <Cpu className="w-4 h-4 text-[#7C5CFF]" />
              <span>3 种模型可选</span>
            </div>
          </div>
          <div className="flex items-center justify-center gap-4">
            <Link
              href="/explore"
              className="btn-gradient inline-flex items-center gap-2 px-8 py-3 rounded-xl text-base glow-primary"
            >
              开始探索
              <ArrowRight className="w-4 h-4" />
            </Link>
            <button
              onClick={handleDownload}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-base border border-[#7C5CFF]/30 text-[#7C5CFF] hover:bg-[#7C5CFF]/10 transition-colors"
            >
              <Download className="w-4 h-4" />
              下载数据集
            </button>
          </div>
        </div>
      </section>

      {/* Case Introduction */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-20">
        <div className="glass rounded-xl p-6 hover:border-[#7C5CFF]/30 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-[#7C5CFF]/15 flex items-center justify-center mb-4">
            <Target className="w-5 h-5 text-[#7C5CFF]" />
          </div>
          <h3 className="text-lg font-semibold mb-2">什么是二分类问题</h3>
          <p className="text-sm text-[#9AA7C7] leading-relaxed">
            机器学习中最基础的任务类型之一——将数据分为两类。在泰坦尼克号案例中，
            就是预测一位乘客<strong className="text-[#62FAD3]">生还</strong>还是<strong className="text-[#FF6B6B]">遇难</strong>。
          </p>
          <div className="flex gap-2 mt-4">
            <span className="px-2 py-0.5 rounded-full bg-[#62FAD3]/15 text-[#62FAD3] text-xs font-medium">生还 = 1</span>
            <span className="px-2 py-0.5 rounded-full bg-[#FF6B6B]/15 text-[#FF6B6B] text-xs font-medium">遇难 = 0</span>
          </div>
        </div>

        <div className="glass rounded-xl p-6 hover:border-[#7C5CFF]/30 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-[#69E7FF]/15 flex items-center justify-center mb-4">
            <Lightbulb className="w-5 h-5 text-[#69E7FF]" />
          </div>
          <h3 className="text-lg font-semibold mb-2">为什么选泰坦尼克号</h3>
          <p className="text-sm text-[#9AA7C7] leading-relaxed">
            Kaggle 经典入门案例，数据量适中、特征直观——性别、年龄、舱位等级都能引发直觉判断，
            是理解机器学习思想的绝佳起点。
          </p>
          <div className="flex gap-2 mt-4">
            <span className="px-2 py-0.5 rounded-full bg-[#7C5CFF]/15 text-[#7C5CFF] text-xs font-medium">Kaggle 经典</span>
            <span className="px-2 py-0.5 rounded-full bg-[#69E7FF]/15 text-[#69E7FF] text-xs font-medium">直觉友好</span>
          </div>
        </div>

        <div className="glass rounded-xl p-6 hover:border-[#7C5CFF]/30 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-[#62FAD3]/15 flex items-center justify-center mb-4">
            <CheckCircle className="w-5 h-5 text-[#62FAD3]" />
          </div>
          <h3 className="text-lg font-semibold mb-2">你将学到什么</h3>
          <p className="text-sm text-[#9AA7C7] leading-relaxed">
            亲手体验完整的机器学习流程：从数据探索与清洗，到模型训练与评估，
            再到用自己训练的模型进行预测。这些技能可以迁移到任何数据科学项目。
          </p>
          <div className="flex gap-2 mt-4">
            <span className="px-2 py-0.5 rounded-full bg-[#62FAD3]/15 text-[#62FAD3] text-xs font-medium">完整流程</span>
            <span className="px-2 py-0.5 rounded-full bg-[#FFD166]/15 text-[#FFD166] text-xs font-medium">可迁移技能</span>
          </div>
        </div>
      </section>

      {/* Flow Preview */}
      <section>
        <h2 className="text-2xl font-bold text-center mb-10">你的机器学习之旅</h2>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {[
            { step: 1, icon: BarChart3, title: '数据探索', desc: '了解数据全貌，发现特征与生还的关联', href: '/explore', color: '#7C5CFF' },
            { step: 2, icon: Database, title: '数据清洗', desc: '处理缺失值，选择与构造特征', href: '/clean', color: '#69E7FF' },
            { step: 3, icon: Cpu, title: '模型训练', desc: '选择算法，配置参数，观察训练过程', href: '/train', color: '#62FAD3' },
            { step: 4, icon: Target, title: '模型评估', desc: '查看准确率、混淆矩阵、特征重要性', href: '/evaluate', color: '#FFD166' },
            { step: 5, icon: Lightbulb, title: '自定义预测', desc: '输入乘客信息，实时预测生还概率', href: '/predict', color: '#FF6B6B' },
          ].map((item) => (
            <div key={item.step}>
              <div className="glass rounded-xl p-5 text-center transition-all group">
                <div
                  className="w-12 h-12 rounded-xl mx-auto mb-3 flex items-center justify-center"
                  style={{ backgroundColor: `${item.color}15` }}
                >
                  <item.icon className="w-6 h-6" style={{ color: item.color }} />
                </div>
                <div className="text-xs font-bold mb-1" style={{ color: item.color }}>STEP {item.step}</div>
                <h3 className="text-sm font-semibold mb-1 transition-colors">{item.title}</h3>
                <p className="text-xs h-10 text-[#9AA7C7]">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
