import { type NextRequest, NextResponse } from 'next/server'
import { updateSession } from './lib/supabase/middleware'
import { createClient } from './lib/supabase/server'

export async function proxy(request: NextRequest) {
  const { nextUrl } = request;
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user && nextUrl.pathname !== '/login' && !nextUrl.pathname.startsWith('/auth')) {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  if (user && nextUrl.pathname === '/login') {
    return NextResponse.redirect(new URL('/', request.url));
  }

  return await updateSession(request)
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
}
