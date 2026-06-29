import { NextResponse } from 'next/server'
import { createClient } from '../../../lib/supabase/server'

export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url)
  const code = searchParams.get('code')
  const next = searchParams.get('next') ?? '/'

  if (code) {
    const supabase = await createClient()
    const { data: { session }, error } = await supabase.auth.exchangeCodeForSession(code)
    if (!error && session) {
      // Sync actual first name to user_metadata in profiles table
      const actualFirstName = session.user.user_metadata?.full_name?.split(" ")[0] || session.user.user_metadata?.name?.split(" ")[0] || session.user.email?.split("@")[0] || "Unknown";
      await supabase.from("profiles").update({
        user_metadata: { name: actualFirstName }
      }).eq("id", session.user.id);

      const forwardedHost = request.headers.get('x-forwarded-host')
      const isLocalEnv = process.env.NODE_ENV === 'development'
      if (isLocalEnv) {
        return NextResponse.redirect(`${origin}${next}`)
      } else if (forwardedHost) {
        return NextResponse.redirect(`https://${forwardedHost}${next}`)
      } else {
        return NextResponse.redirect(`${origin}${next}`)
      }
    } else {
      console.error('Auth error in callback:', error)
      const errorMessage = error?.message || 'Failed to exchange code for session'
      return NextResponse.redirect(`${origin}/auth/auth-code-error?error=${encodeURIComponent(errorMessage)}`)
    }
  }

  const authError = searchParams.get('error_description') || searchParams.get('error') || 'no_code'
  return NextResponse.redirect(`${origin}/auth/auth-code-error?error=${encodeURIComponent(authError)}`)
}
