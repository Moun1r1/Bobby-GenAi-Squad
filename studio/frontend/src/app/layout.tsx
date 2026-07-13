import "./globals.css";
import { Providers } from "./providers";

export const metadata = { title: "Bobby Studio", description: "Visualize & manage generative agent squads" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-mono antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
