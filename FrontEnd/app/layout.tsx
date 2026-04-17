import "./globals.css";
import type { ReactNode } from "react";
import Providers from "./providers";
import { Cinzel_Decorative, EB_Garamond, Playfair_Display, Plus_Jakarta_Sans, Cormorant_Garamond } from "next/font/google";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-jakarta",
  display: "swap",
});

const playfair = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
  display: "swap",
});

const cinzelDec = Cinzel_Decorative({
  weight: ["400", "700"],
  subsets: ["latin"],
  variable: "--font-cinzel-dec",
  display: "swap",
});

const ebGaramond = EB_Garamond({
  subsets: ["latin"],
  variable: "--font-garamond",
  display: "swap",
});

const cormorant = Cormorant_Garamond({
  weight: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-cormorant",
  display: "swap",
});

export const metadata = {
  title: {
    default: "E.C.H.O — Every Capture Has Origins",
    template: "%s | E.C.H.O",
  },
  description:
    "AI-powered portal to Ancient Egypt. Upload, recognize, and converse with pharaohs across millennia.",
  keywords: [
    "Ancient Egypt",
    "Pharaoh",
    "AI",
    "Image Recognition",
    "Tourism",
    "ECHO",
  ],
  openGraph: {
    title: "E.C.H.O — Every Capture Has Origin",
    description:
      "Upload artifacts, chat with pharaohs, and explore Ancient Egypt through AI.",
    siteName: "E.C.H.O",
    type: "website",
  },
  metadataBase: new URL("https://echo-app.vercel.app"),
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: "#0D0A07",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${jakarta.variable} ${playfair.variable} ${cinzelDec.variable} ${ebGaramond.variable} ${cormorant.variable}`}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}