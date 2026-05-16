import { type NextRequest, NextResponse } from 'next/server'
import { updateSession } from './lib/supabase/middleware'
import { createClient } from './lib/supabase/server'

export async function middleware(request: NextRequest) {
  const { nextUrl } = request;
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  // If the user is not logged in and trying to access a protected page
  if (!user && nextUrl.pathname !== '/login' && !nextUrl.pathname.startsWith('/auth')) {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  // If the user is logged in and trying to access the login page
  if (user && nextUrl.pathname === '/login') {
    return NextResponse.redirect(new URL('/', request.url));
  }

  return await updateSession(request)
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * Feel free to modify this pattern to include more paths.
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
}
