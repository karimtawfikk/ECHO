import { NextResponse } from 'next/server'
// The client you created in Step 2
import { createClient } from '../../../lib/supabase/server'

export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url)
  const code = searchParams.get('code')
  // if "next" is in search params, use it as the redirection URL
  const next = searchParams.get('next') ?? '/'

  if (code) {
    const supabase = await createClient()
    const { error } = await supabase.auth.exchangeCodeForSession(code)
    if (!error) {
      const forwardedHost = request.headers.get('x-forwarded-host') // original origin before load balancer
      const isLocalEnv = process.env.NODE_ENV === 'development'
      if (isLocalEnv) {
        // we can be sure that there is no proxy involved in local dev
        return NextResponse.redirect(`${origin}${next}`)
      } else if (forwardedHost) {
        return NextResponse.redirect(`https://${forwardedHost}${next}`)
      } else {
        return NextResponse.redirect(`${origin}${next}`)
      }
    } else {
      console.error('Auth error in callback:', error)
      return NextResponse.redirect(`${origin}/auth/auth-code-error?error=${encodeURIComponent(error.message)}`)
    }
  }

  // return the user to an error page with instructions
  const authError = searchParams.get('error_description') || searchParams.get('error') || 'no_code'
  return NextResponse.redirect(`${origin}/auth/auth-code-error?error=${encodeURIComponent(authError)}`)
}
