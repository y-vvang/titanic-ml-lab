import { NextResponse } from 'next/server';
import { runEngine } from '@/lib/ml-engine';

// Vercel 上该路由代理 Python 函数，冷启动（sklearn import）可能较慢
export const maxDuration = 30;

export async function POST(request: Request) {
  try {
    const config = await request.json();
    const data = await runEngine('clean', config, request);
    return NextResponse.json(data);
  } catch (error) {
    console.error('Clean data error:', error);
    return NextResponse.json({ error: 'Data cleaning failed' }, { status: 500 });
  }
}
