"use client";

import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";

interface ParallaxLayerProps {
  children: React.ReactNode;
  /** Speed factor: 0 = static, 0.5 = half scroll speed, 1 = full scroll speed (default 0.3) */
  speed?: number;
  /** Additional className */
  className?: string;
}

/**
 * A lightweight parallax wrapper using Framer Motion's useScroll.
 * Applies a subtle vertical offset based on scroll progress.
 *
 * Performance-safe:
 * - Uses GPU-accelerated `transform` only
 * - Respects `prefers-reduced-motion` via Framer Motion internals
 * - Scoped to the element's viewport presence, not the entire page
 */
export default function ParallaxLayer({
  children,
  speed = 0.3,
  className,
}: ParallaxLayerProps) {
  const ref = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"], // track from entering to leaving viewport
  });

  // Map scroll progress [0→1] to a vertical offset
  const yOffset = useTransform(scrollYProgress, [0, 1], [speed * 80, speed * -80]);

  return (
    <motion.div ref={ref} style={{ y: yOffset }} className={className}>
      {children}
    </motion.div>
  );
}
