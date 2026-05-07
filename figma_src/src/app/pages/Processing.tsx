import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router';
import { motion, AnimatePresence } from 'motion/react';
import { Check } from 'lucide-react';
import { cn } from '../components/ui-kit';

const steps = [
  { label: 'Reading', message: 'Reading your content...', sub: 'Scanning through your material carefully.' },
  { label: 'Analyzing', message: 'Finding the key ideas...', sub: 'Identifying important concepts and themes.' },
  { label: 'Structuring', message: 'Organizing your notes...', sub: 'Arranging everything in a clear structure.' },
  { label: 'Finalizing', message: 'Building your flashcards...', sub: 'Almost there — polishing the final output.' },
];

export function Processing() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [done, setDone] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setStep(prev => {
        if (prev >= steps.length - 1) {
          clearInterval(interval);
          setDone(true);
          setTimeout(() => navigate('/app/notes/1'), 1200);
          return prev;
        }
        return prev + 1;
      });
    }, 2200);
    return () => clearInterval(interval);
  }, [navigate]);

  return (
    <div className="h-screen w-full bg-[#F4F4F2] flex flex-col items-center justify-center relative overflow-hidden">
      {/* Logo */}
      <div className="absolute top-8 left-1/2 -translate-x-1/2 flex items-center gap-1">
        <span className="font-serif text-[20px] font-medium text-[#1A1A1A]">Nudge</span>
        <span className="w-[7px] h-[7px] rounded-full bg-[#2D6A4F] mb-[1px]" />
      </div>

      <div className="flex flex-col items-center max-w-[440px] w-full px-8 z-10">
        {/* Animated blob */}
        <div className="relative mb-12">
          <motion.div
            className="w-[100px] h-[100px] bg-[#2D6A4F]"
            animate={{
              scale: [0.95, 1.05, 0.95],
              borderRadius: [
                '40% 60% 70% 30% / 40% 50% 60% 50%',
                '60% 40% 30% 70% / 60% 30% 70% 40%',
                '40% 60% 70% 30% / 40% 50% 60% 50%',
              ],
            }}
            transition={{ duration: 4, ease: 'easeInOut', repeat: Infinity }}
          />
          {done && (
            <motion.div
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              className="absolute inset-0 flex items-center justify-center"
            >
              <Check size={32} className="text-white" strokeWidth={3} />
            </motion.div>
          )}
        </div>

        {/* Rotating heading */}
        <div className="h-[80px] flex flex-col items-center justify-center text-center mb-12">
          <AnimatePresence mode="wait">
            <motion.h2
              key={step}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4 }}
              className="font-serif text-[28px] font-medium text-[#1A1A1A] leading-tight"
            >
              {steps[step].message}
            </motion.h2>
          </AnimatePresence>
          <AnimatePresence mode="wait">
            <motion.p
              key={`sub-${step}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              className="text-[14px] text-[#6B6B6B] mt-2"
            >
              {steps[step].sub}
            </motion.p>
          </AnimatePresence>
        </div>

        {/* Step Indicators */}
        <div className="w-full flex justify-between items-start relative mb-12">
          {/* Connecting line */}
          <div className="absolute top-[13px] left-[13px] right-[13px] h-[1px] bg-[#E2E2E2]" />

          {steps.map((s, i) => {
            const completed = i < step;
            const current = i === step;
            return (
              <div key={i} className="flex flex-col items-center gap-2 relative z-10">
                <div
                  className={cn(
                    'w-[26px] h-[26px] rounded-full flex items-center justify-center border-2 transition-all duration-500',
                    completed || current
                      ? 'bg-[#2D6A4F] border-[#2D6A4F]'
                      : 'bg-[#F4F4F2] border-[#E2E2E2]'
                  )}
                >
                  {(completed || current) && (
                    <Check size={13} strokeWidth={3} className="text-white" />
                  )}
                </div>
                <span
                  className={cn(
                    'font-mono text-[10px] uppercase tracking-[0.03em] whitespace-nowrap',
                    current ? 'text-[#1A1A1A] font-medium' : 'text-[#6B6B6B]'
                  )}
                >
                  {s.label}
                </span>
              </div>
            );
          })}
        </div>

        {/* Cancel */}
        <button
          onClick={() => navigate(-1)}
          className="text-[13px] text-[#6B6B6B] hover:text-[#2D6A4F] transition-colors"
        >
          Cancel and go back
        </button>
      </div>
    </div>
  );
}