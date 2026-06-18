"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import Link from "next/link";

function ErrorContent() {
  const searchParams = useSearchParams();
  const error = searchParams.get("error");

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#0D0A07] text-[#F5E6D0] p-4 text-center">
      <h1 className="text-4xl font-bold mb-4 text-[#E6B23C]">Authentication Error</h1>
      <p className="text-xl mb-8">
        There was an error signing in. Please try again.
      </p>
      {error && (
        <div className="bg-red-500/20 text-red-200 p-4 rounded-lg mb-8 max-w-md break-words">
          Error details: {error}
        </div>
      )}
      <Link 
        href="/login" 
        className="px-6 py-3 bg-[#E6B23C] text-[#0D0A07] rounded-xl font-bold uppercase tracking-widest hover:bg-[#FFD369] transition-colors"
      >
        Back to Login
      </Link>
    </div>
  );
}

export default function AuthCodeErrorPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#0D0A07] flex items-center justify-center text-[#E6B23C]">Loading...</div>}>
      <ErrorContent />
    </Suspense>
  );
}
