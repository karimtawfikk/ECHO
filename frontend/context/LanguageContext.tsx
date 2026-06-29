"use client";

import React, { createContext, useContext, useState, ReactNode, useEffect } from "react";
import { dictionaries, Language, DictionaryKey } from "../lib/i18n/dictionaries";

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: DictionaryKey, variables?: Record<string, string>) => string;
  isRTL: boolean;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>("EN");

  useEffect(() => {
    const savedLang = localStorage.getItem("echo-language") as Language;
    if (savedLang && ["EN", "AR", "FR"].includes(savedLang)) {
      setLanguage(savedLang);
      document.documentElement.dir = savedLang === "AR" ? "rtl" : "ltr";
      document.documentElement.lang = savedLang.toLowerCase();
    } else {
      document.documentElement.dir = "ltr";
      document.documentElement.lang = "en";
    }
  }, []);

  const handleSetLanguage = (lang: Language) => {
    setLanguage(lang);
    localStorage.setItem("echo-language", lang);
    document.documentElement.dir = lang === "AR" ? "rtl" : "ltr";
    document.documentElement.lang = lang.toLowerCase();
  };

  const t = (key: DictionaryKey, variables?: Record<string, string>): string => {
    let text = dictionaries[language][key] || dictionaries["EN"][key] || key;

    if (variables) {
      Object.entries(variables).forEach(([vKey, vVal]) => {
        text = text.replace(`{${vKey}}`, vVal);
      });
    }

    return text;
  };

  const isRTL = language === "AR";

  return (
    <LanguageContext.Provider value={{ language, setLanguage: handleSetLanguage, t, isRTL }}>
      <div className={isRTL ? "font-arabic" : ""}>
        {children}
      </div>
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return context;
}
