import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'text' | 'muted';
  size?: 'sm' | 'md' | 'lg';
  fullWidth?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', fullWidth = false, children, ...props }, ref) => {
    const baseStyles = "inline-flex items-center justify-center font-semibold transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2D6A4F] disabled:opacity-50 disabled:cursor-not-allowed";
    
    const variants = {
      primary: "bg-[#2D6A4F] text-white hover:bg-[#245C43] shadow-[0_4px_12px_rgba(45,106,79,0.25)] border border-transparent rounded-lg",
      secondary: "bg-transparent border-[1.5px] border-[#E2E2E2] text-[#1A1A1A] hover:border-[#2D6A4F] hover:text-[#2D6A4F] rounded-lg",
      text: "bg-transparent text-[#2D6A4F] hover:underline p-0 border-none shadow-none rounded-none",
      muted: "bg-[#F0F0EE] text-[#6B6B6B] hover:bg-[#E2E2E2] hover:text-[#1A1A1A] border border-transparent rounded-lg"
    };

    const sizes = {
      sm: "text-sm px-3 py-1.5 h-8",
      md: "text-sm px-5 py-2.5 h-10",
      lg: "text-base px-6 py-3 h-12"
    };

    // Text buttons don't really have standard padding sizes in the same way
    const textSizes = {
      sm: "text-xs",
      md: "text-sm",
      lg: "text-base"
    };

    const sizeStyles = variant === 'text' ? textSizes[size] : sizes[size];

    return (
      <button
        ref={ref}
        className={cn(
          baseStyles,
          variants[variant],
          sizeStyles,
          fullWidth && "w-full",
          className
        )}
        {...props}
      >
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  hoverable?: boolean;
  noPadding?: boolean;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, hoverable = false, noPadding = false, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "bg-white border border-[#E2E2E2] rounded-xl shadow-[0_1px_4px_rgba(0,0,0,0.06)] overflow-hidden",
          hoverable && "hover:border-[#2D6A4F] transition-colors cursor-pointer",
          !noPadding && "p-5",
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);
Card.displayName = "Card";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'primary' | 'secondary' | 'muted' | 'outline' | 'success' | 'warning' | 'error';
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = 'primary', children, ...props }, ref) => {
    const variants = {
      primary: "bg-[#D8E8E0] text-[#2D6A4F]",
      secondary: "bg-[#F0F0EE] text-[#6B6B6B]",
      muted: "bg-[#F4F4F2] text-[#6B6B6B]",
      outline: "bg-transparent border border-[#E2E2E2] text-[#6B6B6B]",
      success: "bg-[#D8E8E0] text-[#2D6A4F]",
      warning: "bg-amber-100 text-amber-800", // Avoid using this if possible per spec
      error: "bg-red-50 text-[#C0392B]",
    };

    return (
      <span
        ref={ref}
        className={cn(
          "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium font-sans",
          variants[variant],
          className
        )}
        {...props}
      >
        {children}
      </span>
    );
  }
);
Badge.displayName = "Badge";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  fullWidth?: boolean;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, fullWidth = true, ...props }, ref) => {
    return (
      <div className={cn("flex flex-col gap-1.5", fullWidth && "w-full")}>
        {label && (
          <label className="text-sm font-medium text-[#1A1A1A]">
            {label}
          </label>
        )}
        <input
          ref={ref}
          className={cn(
            "flex h-10 w-full rounded-lg border-[1.5px] border-[#E2E2E2] bg-white px-3 py-2 text-sm placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] disabled:cursor-not-allowed disabled:opacity-50 transition-colors",
            error && "border-[#C0392B] focus:border-[#C0392B]",
            className
          )}
          {...props}
        />
        {error && (
          <span className="text-xs text-[#C0392B]">{error}</span>
        )}
      </div>
    );
  }
);
Input.displayName = "Input";
