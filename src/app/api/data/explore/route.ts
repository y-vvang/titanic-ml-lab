import { NextResponse } from 'next/server';
import { runEngine } from '@/lib/ml-engine';

// Vercel 上该路由代理 Python 函数，冷启动（sklearn import）可能较慢
export const maxDuration = 30;

export async function GET(request: Request) {
  try {
    // Optional explicit language override from the client (?lang=zh|en);
    // runEngine falls back to the Accept-Language header when absent.
    const lang = new URL(request.url).searchParams.get('lang') ?? undefined;
    const data = await runEngine('explore', lang ? { lang } : {}, request);
    return NextResponse.json(data);
  } catch (error) {
    console.error('Explore data error:', error);
    return NextResponse.json({ error: 'Failed to load explore data' }, { status: 500 });
  }
}
