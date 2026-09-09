import { NextResponse } from 'next/server';
import { runEngine } from '@/lib/ml-engine';

// Vercel 上该路由代理 Python 函数，冷启动（sklearn import）可能较慢
export const maxDuration = 30;

export async function GET(request: Request) {
  try {
    const data = await runEngine('info', {}, request);
    return NextResponse.json(data);
  } catch (error) {
    console.error('Data info error:', error);
    return NextResponse.json({ error: '获取数据信息失败' }, { status: 500 });
  }
}
