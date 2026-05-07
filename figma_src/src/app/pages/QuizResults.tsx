import React, { useState } from 'react';
import { Link } from 'react-router';
import { Card } from '../components/ui-kit';
import { XCircle, ChevronDown, ChevronUp, RotateCcw } from 'lucide-react';
import { cn } from '../components/ui-kit';
import { motion } from 'motion/react';

const wrongAnswers = [
  {
    id: 1,
    question: 'What is the direct product of the Calvin Cycle?',
    yours: 'Glucose',
    correct: 'G3P (Glyceraldehyde-3-phosphate)',
    explanation:
      'While glucose is the ultimate goal, the direct product of the Calvin Cycle is G3P. Two molecules of G3P are needed to synthesize one glucose.',
  },
  {
    id: 2,
    question: 'How many ATP are consumed per CO₂ fixed in the Calvin Cycle?',
    yours: '2 ATP',
    correct: '3 ATP',
    explanation:
      'The Calvin Cycle uses 3 ATP and 2 NADPH per CO₂ fixed. Over 6 turns (to produce one glucose), 18 ATP total are consumed.',
  },
];

const weakTopics = [
  { label: 'Calvin Cycle', pct: '60%' },
  { label: 'Electron Transport', pct: '75%' },
];

export function QuizResults() {
  const score = 8;
  const total = 10;
  const pct = 80;
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const getScoreMessage = () => {
    if (pct >= 90) return 'Outstanding! You have an excellent grasp of this topic.';
    if (pct >= 80) return 'Great work! A quick review of the Calvin Cycle could push you to 100%.';
    if (pct >= 60) return 'Good progress. Revisiting a few key topics will make a real difference.';
    return 'Keep going — every attempt builds your understanding.';
  };

  return (
    <div className="max-w-[600px] mx-auto text-center animate-in fade-in duration-500 space-y-10">
      {/* Score Header */}
      <div>
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        >
          <h1 className="font-serif text-[56px] font-medium text-[#1A1A1A] leading-none mb-2">
            {score} / {total}
          </h1>
          <p className="font-serif text-[28px] font-medium text-[#2D6A4F] mb-4">{pct}%</p>
        </motion.div>

        {/* Score bar */}
        <div className="h-2 bg-[#E2E2E2] rounded-full overflow-hidden max-w-[300px] mx-auto mb-4">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.8, ease: 'easeOut', delay: 0.3 }}
            className="h-full bg-[#2D6A4F] rounded-full"
          />
        </div>

        <p className="text-[15px] text-[#6B6B6B] max-w-[380px] mx-auto leading-relaxed">
          {getScoreMessage()}
        </p>
      </div>

      {/* Weak Topics */}
      <div>
        <h3 className="font-mono text-[11px] text-[#6B6B6B] uppercase tracking-[0.03em] mb-3">
          Topics to review
        </h3>
        <div className="flex justify-center flex-wrap gap-2">
          {weakTopics.map(t => (
            <span
              key={t.label}
              className="bg-[#F0F0EE] text-[#6B6B6B] text-[12px] font-medium px-3 py-1.5 rounded-full"
            >
              {t.label} · {t.pct}
            </span>
          ))}
        </div>
      </div>

      {/* Wrong Answers */}
      <div className="text-left space-y-3">
        <h3 className="font-serif text-[22px] font-medium text-[#1A1A1A] mb-4">
          Review Incorrect Answers
        </h3>

        {wrongAnswers.map(wa => (
          <Card key={wa.id} className="p-5 border-l-[3px] border-l-[#C0392B]">
            <div className="flex items-start justify-between gap-4 mb-4">
              <h4 className="font-serif text-[17px] font-medium text-[#1A1A1A] leading-snug">
                {wa.question}
              </h4>
              <XCircle size={18} className="text-[#C0392B] flex-shrink-0 mt-0.5" />
            </div>

            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3 bg-red-50 rounded-lg border border-red-100">
                <p className="font-mono text-[10px] text-[#C0392B] uppercase tracking-[0.03em] mb-1">
                  Your answer
                </p>
                <p className="text-[13px] text-[#1A1A1A]">{wa.yours}</p>
              </div>
              <div className="p-3 bg-[#D8E8E0]/30 rounded-lg border border-[#D8E8E0]">
                <p className="font-mono text-[10px] text-[#2D6A4F] uppercase tracking-[0.03em] mb-1">
                  Correct answer
                </p>
                <p className="text-[13px] text-[#1A1A1A]">{wa.correct}</p>
              </div>
            </div>

            <button
              onClick={() => setExpandedId(expandedId === wa.id ? null : wa.id)}
              className="flex items-center gap-1.5 text-[13px] text-[#2D6A4F] font-medium hover:underline"
            >
              {expandedId === wa.id ? (
                <>
                  Hide explanation <ChevronUp size={14} />
                </>
              ) : (
                <>
                  Show explanation <ChevronDown size={14} />
                </>
              )}
            </button>

            {expandedId === wa.id && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 pt-3 border-t border-[#E2E2E2]"
              >
                <p className="text-[13px] text-[#6B6B6B] leading-relaxed">
                  <span className="font-semibold text-[#2D6A4F]">Explanation: </span>
                  {wa.explanation}
                </p>
              </motion.div>
            )}
          </Card>
        ))}
      </div>

      {/* Actions */}
      <div className="flex justify-center gap-4">
        <Link to="/app/notes/1">
          <button className="bg-transparent border-[1.5px] border-[#E2E2E2] text-[#1A1A1A] text-[14px] font-semibold px-6 py-2.5 rounded-lg hover:border-[#2D6A4F] hover:text-[#2D6A4F] transition-colors">
            Review Notes →
          </button>
        </Link>
        <Link to="/app/quizzes/start">
          <button className="flex items-center gap-2 bg-[#2D6A4F] text-white text-[14px] font-semibold px-6 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors">
            Try Again <RotateCcw size={15} />
          </button>
        </Link>
      </div>
    </div>
  );
}