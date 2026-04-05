import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        core: "var(--bg-core)",
        surface: "var(--bg-surface)",
        card: "var(--bg-card)",
        primary: "var(--text-primary)",
        secondary: "var(--text-secondary)",
        muted: "var(--text-muted)",
        bullish: "var(--signal-bullish)",
        bearish: "var(--signal-bearish)",
        neutral: "var(--signal-neutral)",
        premium: "var(--accent-premium)"
      },
    },
  },
  plugins: [],
};
export default config;
