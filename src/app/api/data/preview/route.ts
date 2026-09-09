import { NextRequest, NextResponse } from 'next/server';
import { runEngine } from '@/lib/ml-engine';

// Vercel 上该路由代理 Python 函数，冷启动（sklearn import）可能较慢
export const maxDuration = 30;

export async function GET(request: NextRequest) {
  try {
    const rows = parseInt(request.nextUrl.searchParams.get('rows') || '10', 10);
    const data = await runEngine('preview', { rows }, request);
    return NextResponse.json(data);
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : 'Failed to get data preview';
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
