"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Rich markdown renderer. `compact` = chat-bubble spacing (no top/bottom margins); `invert` = light text on a dark bubble.
export function Markdown({ children, compact, invert, className = "" }:
  { children: string; compact?: boolean; invert?: boolean; className?: string }) {
  const prose = [
    "prose prose-sm max-w-none",
    invert ? "prose-invert" : "prose-slate",
    compact ? "prose-p:my-0 prose-headings:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-pre:my-1.5" : "",
    "prose-pre:bg-slate-900 prose-pre:text-slate-100 prose-code:before:content-none prose-code:after:content-none",
    "prose-a:text-blue-600 prose-table:text-[13px]",
    className,
  ].join(" ");
  return (
    <div className={prose}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}
