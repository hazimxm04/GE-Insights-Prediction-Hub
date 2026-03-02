import React, { ReactNode } from 'react';

interface DashboardLayoutProps {
  children: ReactNode;
}

const Header = () => (
  <header className="w-full p-4 border-b-2 border-gray-800 bg-gray-100 text-center">
    <h1 className="text-3xl font-bold mb-4 tracking-tighter text-gray-900">
      GE14 Predictive System
    </h1>
  </header>
);

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="flex flex-col h-screen font-sans text-gray-900 bg-gray-100 border-4 border-black m-4">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        {children}
      </div>
    </div>
  );
}