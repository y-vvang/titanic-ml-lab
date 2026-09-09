import { NextResponse } from 'next/server';
import { runEngine } from '@/lib/ml-engine';

// Vercel 上该路由代理 Python 预测函数（重训后打分），冷启动 + 训练需要更长时间
export const maxDuration = 60;

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { passenger, modelInfo, cleanConfig, hyperparams } = body;

    const config = {
      passenger,
      modelInfo: modelInfo || {},
      cleanConfig: cleanConfig || {},
      hyperparams: hyperparams || {},
    };

    const data = await runEngine('predict', config, request);
    return NextResponse.json(data);
  } catch (error) {
    console.error('Predict error:', error);
    return NextResponse.json({ error: '预测失败' }, { status: 500 });
  }
}
