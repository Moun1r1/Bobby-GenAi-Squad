"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// react-markdown with CITATIONS: inline links of the form `[1](#src-1)` render as Perplexity-style superscript
// chips that scroll to (and flash) the matching source element `#src-1`. Everything else renders as rich markdown.
function flash(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("ring-2", "ring-blue-400");
  setTimeout(() => el.classList.remove("ring-2", "ring-blue-400"), 1200);
}

export function CitedMarkdown({ children, sources = [] }: { children: string; sources?: string[] }) {
  return (
    <div className="prose prose-sm prose-slate max-w-none prose-pre:bg-slate-900 prose-pre:text-slate-100 prose-a:text-blue-600 prose-headings:font-semibold prose-p:leading-7">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children, ...props }) {
            if (href && href.startsWith("#src-")) {
              const n = parseInt(href.replace("#src-", ""), 10);
              return (
                <a
                  href={href}
                  title={sources[n - 1] ? sources[n - 1].slice(0, 220) : undefined}
                  onClick={(e) => { e.preventDefault(); flash(href.slice(1)); }}
                  className="no-underline inline-flex items-center justify-center align-super text-[10px] leading-none font-semibold text-blue-700 bg-blue-100 hover:bg-blue-200 rounded px-1 mx-0.5 cursor-pointer"
                >
                  {children}
                </a>
              );
            }
            return <a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a>;
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
