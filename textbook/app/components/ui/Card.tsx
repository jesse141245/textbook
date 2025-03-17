import { cn } from "@/app/lib/utils";
export function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={`bg-white p-4 rounded-lg shadow-md ${className}`}>{children}</div>;
}

export function CardContent({ children }: { children: React.ReactNode }) {
  return <div className="p-4">{children}</div>;
}

