'use client';

import { ReactNode, useInsertionEffect } from 'react';
import { I18nProvider } from '@/lib/i18n';
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
    useInsertionEffect(() => {
        const d = document.documentElement, zh = /^zh/i.test(navigator.language);
        d.lang = zh ? "zh-CN" : "en";
        d.dataset.lang = zh ? "zh" : "en";
    });

  return (
    <html lang="en">
      <body>
        <I18nProvider>
          <MLProvider>
            <LayoutInner>{children}</LayoutInner>
          </MLProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
