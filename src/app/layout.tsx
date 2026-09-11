import type { Metadata } from 'next';
import './globals.css';
import { ClientLayout } from '@/components/ClientLayout';

export const metadata: Metadata = {
  title: {
    default: 'Titanic ML Lab | Machine Learning from Scratch',
    template: '%s | Titanic ML Lab',
  },
  description:
    'An interactive, beginner-friendly machine learning walkthrough built on the Titanic survival dataset — explore the data, clean it, train and evaluate models step by step.',
  keywords: [
    'machine learning',
    'Titanic',
    'data science',
    'sklearn',
    'interactive tutorial',
    '机器学习',
    '数据科学',
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ClientLayout>{children}</ClientLayout>
  );
}
