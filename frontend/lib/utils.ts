import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const getAssumedImageUrl = (name: string, isPharaoh: boolean) => {
  if (isPharaoh) {
    if (name === 'Akhenaton') return '/images/pharaohs/Akhenaton.JPG';
    if (name === 'Cleopatra VII Philopator') return '/images/pharaohs/Cleopatra%20VII%20Philopator.jpg';
    if (name === 'Hatshepsut') return '/images/pharaohs/Hatshepsut.JPG';
    if (name === 'Ramesses II') return '/images/pharaohs/Ramesses%20II.jpg';
    if (name === 'Tutankhamun') return '/images/pharaohs/Tutankhamun.jpg';
  } else {
    if (name === 'Pyramids of Giza') return '/images/landmarks/Pyramids%20of%20Giza.webp';
    if (name === 'Sphinx') return '/images/landmarks/Sphinx.jpg';
    if (name === 'Temple of Karnak') return '/images/landmarks/Temple%20of%20Karnak.jpg';
    if (name === 'Temple of Luxor') return '/images/landmarks/Temple%20of%20Luxor.jpg';
    if (name === 'The Great Temple of Ramesses II at Abu Simbel') return '/images/landmarks/The%20Great%20Temple%20of%20Ramesses%20II%20at%20Abu%20Simbel.webp';
  }
  return null;
};

export function cleanEntityName(name: string | null | undefined): string {
  if (!name) return "";
  return name.replace(/\s*\([^)]*\)\s*/g, " ").replace(/\s+/g, " ").trim();
}