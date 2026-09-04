"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

export function CommandBar({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="mt-2.5 flex items-center gap-2 overflow-hidden rounded-md border border-border bg-background">
      <code className="flex-1 overflow-x-auto whitespace-nowrap px-2.5 py-1.5 font-mono text-[11px] text-secondary-foreground">
        {command}
      </code>
      <Button
        variant="secondary"
        size="sm"
        className="mr-1 h-6 shrink-0 cursor-pointer px-2 font-mono text-[10px]"
        onClick={() => {
          navigator.clipboard.writeText(command);
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        }}
      >
        {copied ? "copied" : "copy"}
      </Button>
    </div>
  );
}
