import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "ui-sans-serif", "system-ui"],
      },
      colors: {
        ink: {
          950: "#07090d",
          900: "#0c1017",
          800: "#121821",
          700: "#1a2230",
        },
      },
    },
  },
  plugins: [],
};

export default config;
