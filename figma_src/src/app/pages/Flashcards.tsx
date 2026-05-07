import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card } from '../components/ui-kit';
import { ArrowLeft, ArrowRight, RotateCcw, Shuffle, Check } from 'lucide-react';
import { cn } from '../components/ui-kit';
import { Link } from 'react-router';

const allCards = [
  {
    id: 1,
    question: 'What is the primary function of the Calvin Cycle?',
    answer:
      'The Calvin Cycle uses ATP and NADPH from the light-dependent reactions to convert CO₂ into G3P (glyceraldehyde-3-phosphate), the precursor to glucose. It occurs in the stroma of the chloroplast.',
    topic: 'Photosynthesis',
  },
  {
    id: 2,
    question: 'Where do light-dependent reactions occur in the chloroplast?',
    answer:
      'Light-dependent reactions occur in the thylakoid membranes. The stacked grana provide a large surface area for the embedded photosystems and electron transport chains.',
    topic: 'Photosynthesis',
  },
  {
    id: 3,
    question: 'What is the role of chlorophyll in photosynthesis?',
    answer:
      'Chlorophyll absorbs light energy primarily in the blue (430nm) and red (680nm) wavelengths. This energy excites electrons to higher energy levels, powering the light-dependent reactions.',
    topic: 'Photosynthesis',
  },
  {
    id: 4,
    question: 'What is photolysis and what does it produce?',
    answer:
      'Photolysis is the light-driven splitting of water molecules in Photosystem II. It produces electrons (to replace those lost by chlorophyll), protons (H⁺ ions), and oxygen (released as a byproduct).',
    topic: 'Photosynthesis',
  },
];

type Rating = 'still' | 'almost' | 'got';

export function Flashcards() {
  const [cards, setCards] = useState(allCards);
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [direction, setDirection] = useState(0);
  const [ratings, setRatings] = useState<Record<number, Rating>>({});

  const card = cards[index];
  const progress = ((index + 1) / cards.length) * 100;

  const counts = {
    got: Object.values(ratings).filter(r => r === 'got').length,
    almost: Object.values(ratings).filter(r => r === 'almost').length,
    still: Object.values(ratings).filter(r => r === 'still').length,
  };

  const goNext = () => {
    if (index < cards.length - 1) {
      setFlipped(false);
      setDirection(1);
      setTimeout(() => setIndex(i => i + 1), 0);
    }
  };

  const goPrev = () => {
    if (index > 0) {
      setFlipped(false);
      setDirection(-1);
      setTimeout(() => setIndex(i => i - 1), 0);
    }
  };

  const handleRate = (rating: Rating) => {
    setRatings(prev => ({ ...prev, [card.id]: rating }));
    setTimeout(goNext, 300);
  };

  const handleShuffle = () => {
    const shuffled = [...cards].sort(() => Math.random() - 0.5);
    setCards(shuffled);
    setIndex(0);
    setFlipped(false);
    setRatings({});
  };

  const handleRestart = () => {
    setIndex(0);
    setFlipped(false);
    setRatings({});
    setCards(allCards);
  };

  const variants = {
    enter: (d: number) => ({ x: d > 0 ? 280 : -280, opacity: 0, scale: 0.94 }),
    center: { x: 0, opacity: 1, scale: 1 },
    exit: (d: number) => ({ x: d < 0 ? 280 : -280, opacity: 0, scale: 0.94 }),
  };

  return (
    <div className="max-w-[640px] mx-auto flex flex-col items-center animate-in fade-in duration-500">
      {/* Header */}
      <div className="w-full mb-8 text-center">
        <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] mb-4">
          Flashcard Review
        </h1>
        <div className="flex items-center justify-center gap-4">
          <div className="flex-1 max-w-[360px] h-[6px] bg-[#E2E2E2] rounded-full overflow-hidden">
            <div
              className="h-full bg-[#2D6A4F] rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="font-mono text-[12px] text-[#6B6B6B] tracking-[0.03em] whitespace-nowrap">
            Card {index + 1} of {cards.length}
          </span>
        </div>
      </div>

      {/* Card */}
      <div className="relative w-full" style={{ height: 380 }}>
        <AnimatePresence initial={false} custom={direction} mode="wait">
          <motion.div
            key={`${card.id}-${flipped}`}
            custom={direction}
            variants={variants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ type: 'spring', stiffness: 350, damping: 35 }}
            className="absolute inset-0 w-full h-full"
          >
            <div
              className="w-full h-full bg-white rounded-[20px] border-[1.5px] border-[#E2E2E2] flex flex-col items-center justify-between p-8 cursor-pointer select-none"
              style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.04)' }}
              onClick={() => !flipped && setFlipped(true)}
            >
              {!flipped ? (
                /* FRONT */
                <>
                  <span className="font-mono text-[11px] uppercase tracking-[0.06em] text-[#6B6B6B]">
                    Question
                  </span>
                  <div className="flex-1 flex items-center justify-center px-4">
                    <p className="font-serif text-[22px] font-medium text-[#1A1A1A] text-center leading-snug">
                      {card.question}
                    </p>
                  </div>
                  <button className="text-[#2D6A4F] text-[13px] font-semibold hover:underline">
                    Tap to reveal →
                  </button>
                </>
              ) : (
                /* BACK */
                <>
                  <span className="font-mono text-[11px] uppercase tracking-[0.06em] text-[#2D6A4F]">
                    Answer
                  </span>
                  <div className="flex-1 flex flex-col items-center justify-center px-4 gap-4">
                    <p className="text-[15px] text-[#1A1A1A] text-center leading-relaxed">
                      {card.answer}
                    </p>
                    <p className="text-[13px] text-[#6B6B6B] text-center italic font-serif">
                      {card.question}
                    </p>
                  </div>

                  {/* Response buttons */}
                  <div
                    className="w-full grid grid-cols-3 gap-2"
                    onClick={e => e.stopPropagation()}
                  >
                    <button
                      onClick={() => handleRate('still')}
                      className="flex flex-col items-center gap-1 py-3 rounded-xl bg-[#E2E2E2] text-[#1A1A1A] hover:bg-[#D1D1D1] transition-colors"
                    >
                      <span className="text-[12px] font-semibold">Still learning</span>
                    </button>
                    <button
                      onClick={() => handleRate('almost')}
                      className="flex flex-col items-center gap-1 py-3 rounded-xl bg-[#F0F0EE] text-[#52796F] hover:bg-[#E2E2E2] transition-colors"
                    >
                      <span className="text-[12px] font-semibold">Almost there</span>
                    </button>
                    <button
                      onClick={() => handleRate('got')}
                      className="flex flex-col items-center gap-1 py-3 rounded-xl bg-[#D8E8E0] text-[#2D6A4F] hover:bg-[#C5DCD0] transition-colors"
                    >
                      <Check size={14} />
                      <span className="text-[12px] font-semibold">Got it ✓</span>
                    </button>
                  </div>
                </>
              )}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Nav Controls */}
      <div className="flex items-center justify-between w-full mt-8 px-4">
        <button
          onClick={goPrev}
          disabled={index === 0}
          className="text-[#6B6B6B] hover:text-[#2D6A4F] disabled:opacity-30 transition-colors p-2"
        >
          <ArrowLeft size={22} />
        </button>

        <div className="flex gap-6">
          <button
            onClick={handleShuffle}
            className="flex items-center gap-1.5 font-mono text-[12px] text-[#6B6B6B] hover:text-[#2D6A4F] transition-colors tracking-[0.03em]"
          >
            <Shuffle size={14} /> Shuffle
          </button>
          <button
            onClick={handleRestart}
            className="flex items-center gap-1.5 font-mono text-[12px] text-[#6B6B6B] hover:text-[#2D6A4F] transition-colors tracking-[0.03em]"
          >
            <RotateCcw size={14} /> Restart
          </button>
        </div>

        <button
          onClick={goNext}
          disabled={index === cards.length - 1}
          className="text-[#6B6B6B] hover:text-[#2D6A4F] disabled:opacity-30 transition-colors p-2"
        >
          <ArrowRight size={22} />
        </button>
      </div>

      {/* Stats Footer */}
      <div className="mt-6 font-mono text-[12px] text-[#6B6B6B] flex items-center gap-4 tracking-[0.03em]">
        <span className="text-[#2D6A4F] font-medium">Got it: {counts.got}</span>
        <span className="text-[#E2E2E2]">·</span>
        <span>Almost: {counts.almost}</span>
        <span className="text-[#E2E2E2]">·</span>
        <span>Still learning: {counts.still}</span>
      </div>

      {/* Link back */}
      <div className="mt-6">
        <Link to="/app/notes/1" className="text-[13px] text-[#6B6B6B] hover:text-[#2D6A4F] transition-colors">
          ← Back to notes
        </Link>
      </div>

      <style>{`
        .perspective-1000 { perspective: 1000px; }
      `}</style>
    </div>
  );
}