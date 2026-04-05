"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";

interface ScrollRevealProps {
  children: React.ReactNode;
  /** Animation direction: "up" | "down" | "left" | "right" */
  direction?: "up" | "down" | "left" | "right";
  /** Delay in seconds (default 0) */
  delay?: number;
  /** Duration in seconds (default 0.5) */
  duration?: number;
  /** Distance to travel in px (default 30) */
  distance?: number;
  /** Only animate once (default true) */
  once?: boolean;
  /** Viewport trigger margin (default "-50px") */
  margin?: string;
  /** Additional className */
  className?: string;
}

const directionMap = {
  up: { y: 1, x: 0 },
  down: { y: -1, x: 0 },
  left: { x: 1, y: 0 },
  right: { x: -1, y: 0 },
};

/**
 * A reusable scroll-activated reveal component.
 * Wraps children in a Framer Motion element that fades in
 * and slides from the specified direction when scrolled into view.
 *
 * Respects `prefers-reduced-motion` via Framer Motion's built-in support.
 */
export default function ScrollReveal({
  children,
  direction = "up",
  delay = 0,
  duration = 0.5,
  distance = 30,
  once = true,
  margin = "-50px",
  className,
}: ScrollRevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once, margin: margin as `${number}px` });

  const d = directionMap[direction];

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, x: d.x * distance, y: d.y * distance }}
      animate={isInView ? { opacity: 1, x: 0, y: 0 } : undefined}
      transition={{
        duration,
        delay,
        ease: [0.25, 0.1, 0.25, 1], // subtle ease-out
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
