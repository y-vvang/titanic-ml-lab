import { NextResponse } from 'next/server';
import { runEngine } from '@/lib/ml-engine';

// Vercel 上该路由代理 Python 函数，冷启动（sklearn import）可能较慢
export const maxDuration = 30;

export async function GET(request: Request) {
  try {
    const data = await runEngine('explore', {}, request);
    return NextResponse.json(data);
  } catch (error) {
    console.error('Explore data error:', error);
    return NextResponse.json({ error: '获取探索数据失败' }, { status: 500 });
  }
}
