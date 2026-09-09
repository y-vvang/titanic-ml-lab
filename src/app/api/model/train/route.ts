import { spawn } from 'child_process';
import { engineScriptPath, isVercel, pythonFunctionUrl, resolvePython } from '@/lib/ml-engine';

// Vercel 上该路由代理 Python 训练函数，冷启动 + 训练 + 学习曲线计算需要更长时间
export const maxDuration = 60;

const SSE_HEADERS = {
  'Content-Type': 'text/event-stream',
  'Cache-Control': 'no-cache',
  Connection: 'keep-alive',
};

function sseChunk(event: unknown): Uint8Array {
  return new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);
}

export async function POST(request: Request) {
  const config = await request.json();

  // Vercel 模式：Python 函数一次性返回全部事件，这里转成 SSE 流（前端契约不变）
  if (isVercel()) {
    const stream = new ReadableStream({
      async start(controller) {
        try {
          const res = await fetch(pythonFunctionUrl(request, 'train'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
          });
          const body = await res.json();
          const events = Array.isArray(body?.events) ? body.events : [];
          for (const event of events) {
            controller.enqueue(sseChunk(event));
            if (event.type === 'result') break;
          }
          if (events.length === 0) {
            controller.enqueue(
              sseChunk({ type: 'error', message: '训练无输出，请检查函数日志' }),
            );
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          controller.enqueue(sseChunk({ type: 'error', message }));
        }
        controller.close();
      },
    });
    return new Response(stream, { headers: SSE_HEADERS });
  }

  // 本地模式：spawn Python CLI，逐行流式转发（原行为）
  const stream = new ReadableStream({
    async start(controller) {
      const configStr = JSON.stringify(config);
      const python = await resolvePython();
      const proc = spawn(python, [engineScriptPath(), 'train', configStr]);

      proc.stdout.on('data', (data: Buffer) => {
        const lines = data.toString('utf-8').split('\n').filter(Boolean);
        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            controller.enqueue(sseChunk(parsed));

            // If it's the final result, close the stream
            if (parsed.type === 'result') {
              proc.kill();
              controller.close();
            }
          } catch {
            // Not JSON, skip
          }
        }
      });

      proc.stderr.on('data', (data: Buffer) => {
        console.error('ML Engine stderr:', data.toString());
      });

      proc.on('close', (code) => {
        if (code !== 0 && code !== null) {
          controller.enqueue(
            sseChunk({ type: 'error', message: `进程退出码: ${code}` }),
          );
        }
        try {
          controller.close();
        } catch {
          // Already closed
        }
      });

      proc.on('error', (err) => {
        controller.enqueue(sseChunk({ type: 'error', message: err.message }));
        controller.close();
      });
    },
  });

  return new Response(stream, { headers: SSE_HEADERS });
}
