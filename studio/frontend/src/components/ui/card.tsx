import * as React from "react";
import { cn } from "@/lib/utils";

export const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} className={cn("rounded-xl border border-slate-200 bg-white", className)} {...props} />
);
Card.displayName = "Card";

export const CardHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) =>
  <div className={cn("flex flex-col gap-1 p-4", className)} {...props} />;

export const CardTitle = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) =>
  <div className={cn("text-[14px] font-semibold text-slate-900", className)} {...props} />;

export const CardDescription = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) =>
  <div className={cn("text-[12px] text-slate-500", className)} {...props} />;

export const CardContent = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) =>
  <div className={cn("p-4 pt-0", className)} {...props} />;
