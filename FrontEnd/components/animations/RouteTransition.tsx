"use client";

import { motion, AnimatePresence } from "framer-motion";
import { usePathname } from "next/navigation";
import { ReactNode } from "react";

export default function RouteTransition({ children, fullScreen = false }: { children: ReactNode, fullScreen?: boolean }) {
  const pathname = usePathname();

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={pathname}
        initial={{ opacity: 0, y: 10, filter: "blur(10px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        exit={{ opacity: 0, y: -10, filter: "blur(10px)" }}
        transition={{
          duration: 0.4,
          ease: [0.22, 1, 0.36, 1], // Custom cubic-bezier for premium feel
        }}
        className={`w-full ${fullScreen ? 'h-full flex flex-col' : ''}`}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
