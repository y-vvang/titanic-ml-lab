'use client';

import { createContext, useCallback, useContext, useSyncExternalStore, ReactNode } from 'react';
import { en } from './en';
import { zh } from './zh';

export type Lang = 'zh' | 'en';

type Params = Record<string, string | number>;

type Dict = typeof en;

/** Walks dotted key paths ('versions.modelType.decision_tree') through nested dictionary objects. */
function lookup(dict: Dict, key: string): string | undefined {
  let cur: unknown = dict;
  for (const seg of key.split('.')) {
    if (cur && typeof cur === 'object' && seg in (cur as Record<string, unknown>)) {
      cur = (cur as Record<string, unknown>)[seg];
    } else {
      return undefined;
    }
  }
  return typeof cur === 'string' ? cur : undefined;
}

interface I18nContextValue {
  lang: Lang;
  t: (key: string, params?: Params) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

/**
 * Language is auto-detected from navigator.language (zh* → zh, anything else → en).
 * No persistence, no toggle, no routing — re-detected on every load.
 *
 * Detection runs in a blocking inline <script> at the top of <body> (see
 * app/layout.tsx), which sets <html lang> and dataset.lang before first paint.
 * Here we pick that up via useSyncExternalStore: the server render and the
 * hydration render both use 'en' (the getServerSnapshot) so hydration never
 * mismatches, and React synchronously re-renders with the client snapshot
 * immediately after hydration — earlier than any useEffect-based approach,
 * and without making routes dynamic (no cookies() in the layout).
 */
function getSnapshot(): Lang {
  return document.documentElement.dataset.lang === 'zh' ? 'zh' : 'en';
}

/** No runtime language switching exists, so there is nothing to subscribe to. */
const subscribe = () => () => {};

export function I18nProvider({ children }: { children: ReactNode }) {
  const lang = useSyncExternalStore(subscribe, getSnapshot, (): Lang => 'en');

  const t = useCallback(
    (key: string, params?: Params): string => {
      // Every fallback goes to English: missing zh key → en value → key itself.
      const raw = (lang === 'zh' ? lookup(zh, key) : undefined) ?? lookup(en, key) ?? key;
      if (!params) return raw;
      return raw.replace(/\{(\w+)\}/g, (_, name: string) =>
        params[name] !== undefined ? String(params[name]) : `{${name}}`,
      );
    },
    [lang],
  );

  return <I18nContext.Provider value={{ lang, t }}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return ctx;
}
