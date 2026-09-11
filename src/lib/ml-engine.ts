import { execFile } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execFileAsync = promisify(execFile);
const SCRIPT_PATH = path.resolve(process.cwd(), 'scripts/ml_engine.py');

/** Vercel 部署环境：ML 逻辑跑在 /api/py/* 的 Python Serverless Functions 里 */
export function isVercel(): boolean {
  return !!process.env.VERCEL;
}

export interface EngineEvent {
  type: string;
  [key: string]: unknown;
}

export type Lang = 'zh' | 'en';

/**
 * 解析引擎输出语言（服务端）：
 * 1. config 里显式带的 lang（客户端 fetch 传入，与 navigator.language 检测一致）优先；
 * 2. 否则回退 Accept-Language 请求头（与浏览器语言设置同源）；
 * 3. 一切未知情况 → 英文。
 */
export function detectLang(request?: Request, explicit?: unknown): Lang {
  if (explicit === 'zh' || explicit === 'en') return explicit;
  const al = request?.headers.get('accept-language') ?? '';
  return /^zh/i.test(al.trim()) ? 'zh' : 'en';
}

/** 解析 Python 函数的完整 URL（Vercel 模式下 Next.js 路由代理请求用） */
export function pythonFunctionUrl(request: Request, command: string): string {
  const pathname = `/api/py/${command}`;
  // 优先用部署 URL（生产/预览域名，由 Vercel 注入）
  if (process.env.VERCEL_URL) return `https://${process.env.VERCEL_URL}${pathname}`;
  if (process.env.URL) return `${process.env.URL}${pathname}`;
  // 回退：从请求头推导出本次请求的主机
  const proto = request.headers.get('x-forwarded-proto') ?? 'https';
  const host = request.headers.get('x-forwarded-host') ?? request.headers.get('host');
  if (host) return `${proto}://${host}${pathname}`;
  return new URL(pathname, request.url).toString();
}

// ── 本地模式：spawn Python CLI ──

/** CLI 包装脚本路径（核心逻辑在 api/py/shared/ml_engine.py） */
export function engineScriptPath(): string {
  return SCRIPT_PATH;
}

let resolvedPython: string | null = null;

/** Windows 上可能只有 python 而没有 python3，首次调用时探测并缓存 */
export async function resolvePython(): Promise<string> {
  if (resolvedPython) return resolvedPython;
  for (const candidate of ['python3', 'python']) {
    try {
      await execFileAsync(candidate, ['--version']);
      resolvedPython = candidate;
      return candidate;
    } catch {
      // 尝试下一个
    }
  }
  throw new Error('Python 3 not found (tried python3 / python). Please install it and add it to PATH');
}

/**
 * 执行非 train 命令，返回引擎结果。
 * - Vercel：POST /api/py/<command>（Python Serverless Function，依赖见 requirements.txt）
 * - 本地：spawn python3 scripts/ml_engine.py <command> <json>
 */
export async function runEngine(
  command: string,
  config: unknown,
  request?: Request,
): Promise<unknown> {
  // 统一注入 lang：显式传入 > Accept-Language > en（见 detectLang）
  const cfg = {
    ...((config ?? {}) as Record<string, unknown>),
    lang: detectLang(request, (config as Record<string, unknown> | undefined)?.lang),
  };

  if (isVercel() && request) {
    const res = await fetch(pythonFunctionUrl(request, command), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    if (!res.ok) {
      throw new Error(`ML engine ${command} call failed: HTTP ${res.status}`);
    }
    return res.json();
  }

  const python = await resolvePython();
  const { stdout } = await execFileAsync(
    python,
    [SCRIPT_PATH, command, JSON.stringify(cfg)],
    { timeout: 30_000, maxBuffer: 10 * 1024 * 1024 },
  );
  return JSON.parse(stdout.trim());
}
