'use client';

import { useCallback } from 'react';
import Link from 'next/link';
import { BarChart3, Database, Cpu, CheckCircle, Target, Lightbulb, ArrowRight, Download } from 'lucide-react';
import { useI18n } from '@/lib/i18n';

export default function HomePage() {
  const { t } = useI18n();

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
          <p className="text-2xl text-[#9AA7C7] mb-4">{t('home.heroSubtitle')}</p>
          <p className="text-sm text-[#9AA7C7]/50 max-w-2xl mx-auto mb-4">
            {t('home.heroTagline')}
          </p>
          <div className="flex items-center justify-center gap-4 mb-7">
            <div className="flex items-center gap-2 text-sm text-[#9AA7C7]">
              <Database className="w-4 h-4 text-[#69E7FF]" />
              <span>{t('home.realData')}</span>
            </div>
            <div className="w-1 h-1 rounded-full bg-[#9AA7C7]/40" />
            <div className="flex items-center gap-2 text-sm text-[#9AA7C7]">
              <Cpu className="w-4 h-4 text-[#7C5CFF]" />
              <span>{t('home.modelsAvailable')}</span>
            </div>
          </div>
          <div className="flex items-center justify-center gap-4">
            <Link
              href="/explore"
              className="btn-gradient inline-flex items-center gap-2 px-8 py-3 rounded-xl text-base glow-primary"
            >
              {t('home.startExploring')}
              <ArrowRight className="w-4 h-4" />
            </Link>
            <button
              onClick={handleDownload}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-base border border-[#7C5CFF]/30 text-[#7C5CFF] hover:bg-[#7C5CFF]/10 transition-colors"
            >
              <Download className="w-4 h-4" />
              {t('home.downloadDataset')}
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
          <h3 className="text-lg font-semibold mb-2">{t('home.whatBinaryTitle')}</h3>
          <p className="text-sm text-[#9AA7C7] leading-relaxed">
            {t('home.whatBinaryDescA')}
            <strong className="text-[#62FAD3]">{t('home.survived')}</strong>
            {t('home.whatBinaryDescB')}
            <strong className="text-[#FF6B6B]">{t('home.died')}</strong>
            {t('home.whatBinaryDescEnd')}
          </p>
          <div className="flex gap-2 mt-4">
            <span className="px-2 py-0.5 rounded-full bg-[#62FAD3]/15 text-[#62FAD3] text-xs font-medium">{t('home.survivedOne')}</span>
            <span className="px-2 py-0.5 rounded-full bg-[#FF6B6B]/15 text-[#FF6B6B] text-xs font-medium">{t('home.diedZero')}</span>
          </div>
        </div>

        <div className="glass rounded-xl p-6 hover:border-[#7C5CFF]/30 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-[#69E7FF]/15 flex items-center justify-center mb-4">
            <Lightbulb className="w-5 h-5 text-[#69E7FF]" />
          </div>
          <h3 className="text-lg font-semibold mb-2">{t('home.whyTitanicTitle')}</h3>
          <p className="text-sm text-[#9AA7C7] leading-relaxed">
            {t('home.whyTitanicDesc')}
          </p>
          <div className="flex gap-2 mt-4">
            <span className="px-2 py-0.5 rounded-full bg-[#7C5CFF]/15 text-[#7C5CFF] text-xs font-medium">{t('home.kaggleClassic')}</span>
            <span className="px-2 py-0.5 rounded-full bg-[#69E7FF]/15 text-[#69E7FF] text-xs font-medium">{t('home.intuitionFriendly')}</span>
          </div>
        </div>

        <div className="glass rounded-xl p-6 hover:border-[#7C5CFF]/30 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-[#62FAD3]/15 flex items-center justify-center mb-4">
            <CheckCircle className="w-5 h-5 text-[#62FAD3]" />
          </div>
          <h3 className="text-lg font-semibold mb-2">{t('home.learnTitle')}</h3>
          <p className="text-sm text-[#9AA7C7] leading-relaxed">
            {t('home.learnDesc')}
          </p>
          <div className="flex gap-2 mt-4">
            <span className="px-2 py-0.5 rounded-full bg-[#62FAD3]/15 text-[#62FAD3] text-xs font-medium">{t('home.fullWorkflow')}</span>
            <span className="px-2 py-0.5 rounded-full bg-[#FFD166]/15 text-[#FFD166] text-xs font-medium">{t('home.transferableSkills')}</span>
          </div>
        </div>
      </section>

      {/* Flow Preview */}
      <section>
        <h2 className="text-2xl font-bold text-center mb-10">{t('home.journeyTitle')}</h2>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {[
            { step: 1, icon: BarChart3, title: t('nav.stepExplore'), desc: t('home.flowExploreDesc'), href: '/explore', color: '#7C5CFF' },
            { step: 2, icon: Database, title: t('nav.stepClean'), desc: t('home.flowCleanDesc'), href: '/clean', color: '#69E7FF' },
            { step: 3, icon: Cpu, title: t('nav.stepTrain'), desc: t('home.flowTrainDesc'), href: '/train', color: '#62FAD3' },
            { step: 4, icon: Target, title: t('nav.stepEvaluate'), desc: t('home.flowEvaluateDesc'), href: '/evaluate', color: '#FFD166' },
            { step: 5, icon: Lightbulb, title: t('nav.stepPredict'), desc: t('home.flowPredictDesc'), href: '/predict', color: '#FF6B6B' },
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
