/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
    theme: {
        extend: {
            colors: {
                bg: "#050816",
                card: "#0f172a",
                accent: "#22d3ee"
            }
        }
    },
    plugins: []
};
