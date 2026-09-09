import type { Metadata } from 'next';
import { Inspector } from 'react-dev-inspector';
import './globals.css';
import { ClientLayout } from '@/components/ClientLayout';

export const metadata: Metadata = {
  title: {
    default: 'Titanic ML Lab | 机器学习从零开始',
    template: '%s | Titanic ML Lab',
  },
  description:
    '以泰坦尼克号生还预测为案例的交互式机器学习科普平台，让小白亲身走完数据清洗、模型训练、模型评估全流程。',
  keywords: ['机器学习', '泰坦尼克号', '数据科学', 'sklearn', '科普'],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const isDev = process.env.COZE_PROJECT_ENV === 'DEV';

  return (
    <html lang="zh-CN">
      <body className="antialiased">
        {isDev && <Inspector />}
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  );
}
