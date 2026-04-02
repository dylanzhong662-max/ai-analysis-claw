/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg:      "#0f0f12",
        surface: "#1a1a24",
        border:  "#2a2a38",
        muted:   "#94a3b8",
      },
    },
  },
  plugins: [],
}
