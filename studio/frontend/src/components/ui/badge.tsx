import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-blue-100 text-blue-700",
        proven: "border-transparent bg-green-100 text-green-700",
        active: "border-transparent bg-blue-100 text-blue-700",
        warn: "border-transparent bg-amber-100 text-amber-700",
        dead: "border-transparent bg-slate-100 text-slate-500",
        error: "border-transparent bg-red-100 text-red-600",
        outline: "border-slate-200 text-slate-600",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}
export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
