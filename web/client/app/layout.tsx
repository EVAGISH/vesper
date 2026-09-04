import type { Metadata } from "next";
import "./globals.css";
import { TopBar } from "@/components/topbar";
import { VesperProvider } from "@/components/vesper-provider";

export const metadata: Metadata = {
  title: "Vesper",
  description: "Drone-simulation runs browser",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="flex h-dvh flex-col overflow-hidden text-[13px]">
        <VesperProvider>
          <TopBar />
          {children}
        </VesperProvider>
      </body>
    </html>
  );
}
