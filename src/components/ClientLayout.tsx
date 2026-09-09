'use client';

import { ReactNode } from 'react';
import { MLProvider } from '@/lib/MLContext';
import { TopNav, StepNav } from '@/components/StepNav';
import { usePathname } from 'next/navigation';

function LayoutInner({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const isHome = pathname === '/';

  return (
    <div className="min-h-screen bg-[#070A14] text-[#F7FAFF]">
      <TopNav />
      {!isHome && <StepNav />}
      {children}
    </div>
  );
}

export function ClientLayout({ children }: { children: ReactNode }) {
  return (
    <MLProvider>
      <LayoutInner>{children}</LayoutInner>
    </MLProvider>
  );
}
