import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router';
import { Card } from '../components/ui-kit';
import { ArrowLeft, ArrowRight, Clock, CheckCircle, XCircle, HelpCircle } from 'lucide-react';
import { cn } from '../components/ui-kit';
import { motion, AnimatePresence } from 'motion/react';

const questions = [
  {
    id: 1,
    text: 'Which molecule is the primary electron donor in the light-dependent reactions of photosynthesis?',
    options: ['Water (H₂O)', 'Carbon Dioxide (CO₂)', 'Oxygen (O₂)', 'Glucose (C₆H₁₂O₆)'],
    correct: 0,
    explanation: 'Water is split via photolysis to donate electrons to Photosystem II, releasing O₂ as a byproduct.',
  },
  {
    id: 2,
    text: 'Where does the Calvin Cycle take place within the chloroplast?',
    options: ['Thylakoid membrane', 'Stroma', 'Cytoplasm', 'Mitochondria'],
    correct: 1,
    explanation: 'The Calvin Cycle occurs in the stroma of the chloroplast, using ATP and NADPH from the light reactions.',
  },
  {
    id: 3,
    text: 'What is the direct product of the Calvin Cycle?',
    options: ['Glucose', 'ATP', 'G3P (Glyceraldehyde-3-phosphate)', 'NADPH'],
    correct: 2,
    explanation: 'G3P (glyceraldehyde-3-phosphate) is the direct 3-carbon product, which is later used to synthesize glucose.',
  },
  {
    id: 4,
    text: 'Which pigment primarily absorbs red and blue light for photosynthesis?',
    options: ['Carotenoids', 'Xanthophylls', 'Chlorophyll-a', 'Phycocyanin'],
    correct: 2,
    explanation: 'Chlorophyll-a is the primary photosynthetic pigment that absorbs red (680nm) and blue (430nm) light.',
  },
  {
    id: 5,
    text: 'How many ATP molecules are produced per turn of the Calvin Cycle?',
    options: ['1 ATP', '2 ATP', '3 ATP', '6 ATP'],
    correct: 2,
    explanation: 'Three ATP and two NADPH are consumed per CO₂ fixed in the Calvin Cycle.',
  },
];

export function Quiz() {
  const navigate = useNavigate();
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [timeLeft, setTimeLeft] = useState(600);
  const [answers, setAnswers] = useState<Record<number, number>>({});

  useEffect(() => {
    const t = setInterval(() => setTimeLeft(p => (p > 0 ? p - 1 : 0)), 1000);
    return () => clearInterval(t);
  }, []);

  const fmt = (s: number) =>
    `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

  const q = questions[current];
  const progress = ((current + 1) / questions.length) * 100;

  const handleSelect = (i: number) => {
    if (!answered) setSelected(i);
  };

  const handleSubmit = () => {
    if (selected === null) return;
    setAnswered(true);
    setAnswers(prev => ({ ...prev, [current]: selected }));
  };

  const handleNext = () => {
    if (current < questions.length - 1) {
      setCurrent(c => c + 1);
      setSelected(answers[current + 1] ?? null);
      setAnswered(current + 1 in answers);
    } else {
      navigate('/app/quizzes/results');
    }
  };

  const optionStyle = (i: number) => {
    if (!answered) {
      if (selected === i) return 'border-[#2D6A4F] border-l-[4px] bg-[#D8E8E0]/30';
      return 'border-[#E2E2E2] bg-white hover:border-[#2D6A4F] hover:bg-[#D8E8E0]/10';
    }
    if (i === q.correct) return 'border-[#52796F] border-l-[4px] bg-[#D8E8E0]/30';
    if (selected === i) return 'border-[#C0392B] border-l-[4px] bg-red-50';
    return 'border-[#E2E2E2] bg-white opacity-50';
  };

  return (
    <div className="max-w-[720px] mx-auto animate-in fade-in duration-500">
      {/* Top Bar */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="font-mono text-[11px] text-[#6B6B6B] uppercase tracking-[0.03em]">
              Biology Quiz
            </p>
            <p className="text-[15px] font-medium text-[#1A1A1A]">Photosynthesis</p>
          </div>

          <span className="font-mono text-[14px] font-medium text-[#1A1A1A]">
            Question {current + 1} of {questions.length}
          </span>

          <div
            className={cn(
              'flex items-center gap-2 px-3 py-1.5 rounded-full',
              timeLeft < 120 ? 'bg-red-50 text-[#C0392B]' : 'bg-[#D8E8E0] text-[#2D6A4F]'
            )}
          >
            <Clock size={14} />
            <span className="font-mono text-[13px] font-medium">{fmt(timeLeft)}</span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="h-[6px] bg-[#E2E2E2] rounded-full overflow-hidden">
          <div
            className="h-full bg-[#2D6A4F] rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Question */}
      <AnimatePresence mode="wait">
        <motion.div
          key={current}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.2 }}
        >
          <div className="mb-6">
            <span className="font-mono text-[17px] font-medium text-[#2D6A4F] tracking-[0.03em]">
              Q{current + 1}.
            </span>
            <h2 className="font-serif text-[22px] font-medium text-[#1A1A1A] leading-snug mt-1">
              {q.text}
            </h2>
          </div>

          {/* Options */}
          <div className="space-y-3 mb-6">
            {q.options.map((opt, i) => (
              <div
                key={i}
                onClick={() => handleSelect(i)}
                className={cn(
                  'p-4 rounded-xl border-[1.5px] transition-all duration-200 cursor-pointer flex items-center justify-between',
                  optionStyle(i)
                )}
              >
                <div className="flex items-center gap-4">
                  <span
                    className={cn(
                      'w-8 h-8 rounded-lg flex items-center justify-center font-mono text-[13px] border flex-shrink-0',
                      selected === i && !answered
                        ? 'bg-[#2D6A4F] text-white border-[#2D6A4F]'
                        : answered && i === q.correct
                        ? 'bg-[#52796F] text-white border-[#52796F]'
                        : answered && selected === i
                        ? 'bg-[#C0392B] text-white border-[#C0392B]'
                        : 'bg-white text-[#6B6B6B] border-[#E2E2E2]'
                    )}
                  >
                    {String.fromCharCode(65 + i)}
                  </span>
                  <span className="text-[15px] text-[#1A1A1A]">{opt}</span>
                </div>
                {answered && i === q.correct && (
                  <CheckCircle size={18} className="text-[#52796F]" />
                )}
                {answered && selected === i && i !== q.correct && (
                  <XCircle size={18} className="text-[#C0392B]" />
                )}
              </div>
            ))}
          </div>

          {/* Explanation */}
          {answered && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 rounded-xl bg-[#F0F0EE] border border-[#E2E2E2]"
            >
              <div className="flex items-start gap-3">
                <HelpCircle size={18} className="text-[#2D6A4F] mt-0.5 flex-shrink-0" />
                <div>
                  <span className="font-mono text-[11px] text-[#2D6A4F] uppercase tracking-[0.03em] block mb-1">
                    Explanation
                  </span>
                  <p className="text-[14px] text-[#1A1A1A] leading-relaxed">{q.explanation}</p>
                </div>
              </div>
            </motion.div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Footer */}
      <div className="flex items-center justify-between pt-6 border-t border-[#E2E2E2]">
        <button
          onClick={() => {
            if (current > 0) {
              setCurrent(c => c - 1);
              setSelected(answers[current - 1] ?? null);
              setAnswered(current - 1 in answers);
            }
          }}
          disabled={current === 0}
          className="flex items-center gap-2 text-[14px] text-[#6B6B6B] hover:text-[#2D6A4F] disabled:opacity-30 transition-colors font-medium"
        >
          <ArrowLeft size={16} /> Previous
        </button>

        {!answered ? (
          <button
            onClick={handleSubmit}
            disabled={selected === null}
            className="bg-[#2D6A4F] text-white text-[14px] font-semibold px-6 py-2.5 rounded-lg hover:bg-[#245C43] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Submit Answer
          </button>
        ) : (
          <button
            onClick={handleNext}
            className="flex items-center gap-2 bg-[#2D6A4F] text-white text-[14px] font-semibold px-6 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors"
            style={{ boxShadow: '0 4px 12px rgba(45,106,79,0.25)' }}
          >
            {current === questions.length - 1 ? 'Finish Quiz' : 'Next Question'}
            <ArrowRight size={16} />
          </button>
        )}
      </div>
    </div>
  );
}