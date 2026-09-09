'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Brain, ChevronRight } from 'lucide-react';
import { useML } from '@/lib/MLContext';

const STEPS = [
  { label: '数据探索', href: '/explore', key: 'explore' as const },
  { label: '数据清洗', href: '/clean', key: 'clean' as const },
  { label: '模型训练', href: '/train', key: 'train' as const },
  { label: '模型评估', href: '/evaluate', key: 'evaluate' as const },
  { label: '自定义预测', href: '/predict', key: 'predict' as const },
];

export function StepNav() {
  const pathname = usePathname();
  const { stepCompleted } = useML();

  // Determine current step from pathname
  const currentStepIndex = STEPS.findIndex(s => s.href === pathname);

  return (
    <div className="glass border-t border-white/[0.06] flex items-center justify-center gap-6 px-6 py-2.5 overflow-x-hidden">
      {STEPS.map((step, i) => {
        const isCurrentStep = i === currentStepIndex;
        const isDone = stepCompleted[step.key];
        const colorClass = isDone
          ? 'text-[#62FAD3]'
          : isCurrentStep
            ? 'text-[#7C5CFF]'
            : 'text-[#9AA7C7]';
        const bgClass = isDone
          ? 'bg-[#62FAD3]/20'
          : isCurrentStep
            ? 'bg-[#7C5CFF]/20'
            : 'bg-[#181D3A]';
        return (
          <div key={step.href} className="flex items-center gap-1.5">
            {i > 0 && <ChevronRight className="w-3.5 h-3.5 text-[#9AA7C7]/50 mr-1" />}
            <Link
              href={step.href}
              className={`flex items-center gap-2 ${colorClass} hover:opacity-80 transition-opacity`}
            >
              <div className={`w-6 h-6 rounded-full ${bgClass} flex items-center justify-center text-xs font-bold`}>
                {isDone ? '✓' : i + 1}
              </div>
              <span className="text-xs font-medium whitespace-nowrap">{step.label}</span>
            </Link>
          </div>
        );
      })}
    </div>
  );
}

export function TopNav() {
  const pathname = usePathname();

  return (
    <header className="glass-strong sticky top-0 z-40 h-10 flex items-center justify-between px-4">
      <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
        <Brain className="text-[#7C5CFF] w-5 h-5" />
        <span className="font-bold text-base text-[#F7FAFF]">Titanic ML Lab</span>
      </Link>
      <nav className="flex items-center gap-1">
        <Link href="/" className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${pathname === "/" ? 'text-[#7C5CFF] bg-[#7C5CFF]/15' : 'text-[#9AA7C7] hover:text-[#F7FAFF] hover:bg-[#181D3A]'}`}>
          首页
        </Link>
        <Link href="/versions" className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${pathname === "/versions" ? 'text-[#7C5CFF] bg-[#7C5CFF]/15' : 'text-[#9AA7C7] hover:text-[#F7FAFF] hover:bg-[#181D3A]'}`}>
          版本管理
        </Link>
        <a href="https://www.kaggle.com/competitions/titanic" target="_blank" rel="noopener noreferrer" className="px-3 py-1.5 rounded-md transition-colors hover:bg-[#181D3A]"><img src="https://www.kaggle.com/static/images/site-logo.svg" className="w-10 h-5" alt="Kaggle" /></a>
      </nav>
    </header>
  );
}
