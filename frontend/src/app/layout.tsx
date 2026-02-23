import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Navi Mumbai House Price Predictor | AI-Powered Real Estate Estimates",
  description:
    "Get accurate house price predictions in Navi Mumbai using advanced machine learning. Covers Kharghar, Panvel, Vashi, Nerul, Airoli, and more.",
  keywords: [
    "Navi Mumbai",
    "house price predictor",
    "real estate",
    "ML",
    "property valuation",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
