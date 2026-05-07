import React, { useState } from 'react';
import Masonry, { ResponsiveMasonry } from 'react-responsive-masonry';
import { Card } from '../components/ui-kit';
import { FileText, CreditCard, GraduationCap, ArrowRight, Bookmark, Search, X } from 'lucide-react';
import { cn } from '../components/ui-kit';

const allBookmarks = [
  {
    type: 'note',
    title: 'Photosynthesis Overview',
    subject: 'Biology',
    meta: '2 days ago',
    content:
      'Photosynthesis is the process used by plants, algae, and some bacteria to convert light energy into chemical energy stored in glucose...',
  },
  {
    type: 'flashcard',
    title: 'Calvin Cycle Products',
    subject: 'Biology',
    meta: '12 cards',
    content: 'What are the main outputs of the Calvin Cycle? G3P is the direct product, which is used to build glucose and other compounds.',
  },
  {
    type: 'quiz',
    title: 'Plant Biology Quiz',
    subject: 'Biology',
    meta: '80% score',
    content: 'Q5: Which pigment absorbs light energy for photosynthesis? Chlorophyll, specifically chlorophyll-a, is the primary pigment.',
  },
  {
    type: 'note',
    title: "Newton's Laws of Motion",
    subject: 'Physics',
    meta: '1 week ago',
    content:
      '1. An object at rest stays at rest. 2. F = ma. 3. Every action has an equal and opposite reaction. These fundamental laws govern classical mechanics.',
  },
  {
    type: 'flashcard',
    title: 'Periodic Table Groups',
    subject: 'Chemistry',
    meta: '24 cards',
    content:
      'Group 1 elements are Alkali Metals. Group 17 are Halogens. Group 18 are Noble Gases. Each group shares similar chemical properties.',
  },
  {
    type: 'note',
    title: 'World War II: Key Events',
    subject: 'History',
    meta: '3 days ago',
    content:
      'The war began in September 1939 with Germany\'s invasion of Poland. Key events include D-Day (1944), Battle of Stalingrad, and the atomic bombings of Japan.',
  },
  {
    type: 'quiz',
    title: 'Organic Chemistry MCQ',
    subject: 'Chemistry',
    meta: '65% score',
    content: 'Q2: Which mechanism describes the reaction of a primary alkyl halide with NaOH? The SN2 mechanism proceeds via backside attack.',
  },
];

const filters = ['All', 'Notes', 'Flashcards', 'Quiz Questions'];

export function Bookmarks() {
  const [activeFilter, setActiveFilter] = useState('All');
  const [search, setSearch] = useState('');
  const [revisionMode, setRevisionMode] = useState(false);
  const [revisionIndex, setRevisionIndex] = useState(0);

  const typeMap: Record<string, string> = {
    'Notes': 'note',
    'Flashcards': 'flashcard',
    'Quiz Questions': 'quiz',
  };

  const filtered = allBookmarks.filter(b => {
    const matchesFilter = activeFilter === 'All' || b.type === typeMap[activeFilter];
    const matchesSearch =
      !search ||
      b.title.toLowerCase().includes(search.toLowerCase()) ||
      b.subject.toLowerCase().includes(search.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const borderColor = (type: string) => {
    if (type === 'note') return 'border-l-[#2D6A4F]';
    if (type === 'flashcard') return 'border-l-[#52796F]';
    return 'border-l-[#6B6B6B]';
  };

  const typeIcon = (type: string) => {
    if (type === 'note') return <FileText size={12} className="text-[#2D6A4F]" />;
    if (type === 'flashcard') return <CreditCard size={12} className="text-[#52796F]" />;
    return <GraduationCap size={12} className="text-[#6B6B6B]" />;
  };

  const typeLabel = (type: string) => {
    if (type === 'note') return 'Note';
    if (type === 'flashcard') return 'Flashcard';
    return 'Quiz';
  };

  // Revision mode
  if (revisionMode && filtered.length > 0) {
    const item = filtered[revisionIndex];
    return (
      <div className="fixed inset-0 bg-[#F4F4F2] z-50 flex flex-col items-center justify-center p-8">
        <div className="absolute top-6 right-6 flex items-center gap-3">
          <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
            {revisionIndex + 1} / {filtered.length}
          </span>
          <button
            onClick={() => setRevisionMode(false)}
            className="text-[#6B6B6B] hover:text-[#1A1A1A] transition-colors"
          >
            <X size={22} />
          </button>
        </div>

        <div className="max-w-[680px] w-full">
          <div className={cn('bg-white rounded-xl border border-[#E2E2E2] border-l-[3px] p-8 shadow-sm', borderColor(item.type))}>
            <div className="flex items-center gap-2 mb-4">
              <div className="flex items-center gap-1.5 bg-[#F0F0EE] px-2.5 py-1 rounded-full">
                {typeIcon(item.type)}
                <span className="text-[12px] font-medium text-[#6B6B6B]">{typeLabel(item.type)}</span>
              </div>
              <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">
                {item.subject}
              </span>
            </div>
            <h2 className="font-serif text-[22px] font-medium text-[#1A1A1A] mb-4">
              {item.title}
            </h2>
            <p className="text-[15px] text-[#1A1A1A] leading-relaxed">{item.content}</p>
          </div>

          <div className="flex items-center justify-between mt-8 px-2">
            <button
              onClick={() => setRevisionIndex(i => Math.max(0, i - 1))}
              disabled={revisionIndex === 0}
              className="text-[#6B6B6B] hover:text-[#2D6A4F] disabled:opacity-30 transition-colors text-[14px] font-medium"
            >
              ← Previous
            </button>
            <span className="font-mono text-[12px] text-[#6B6B6B] tracking-[0.03em]">
              {revisionIndex + 1} of {filtered.length}
            </span>
            <button
              onClick={() =>
                setRevisionIndex(i => Math.min(filtered.length - 1, i + 1))
              }
              disabled={revisionIndex === filtered.length - 1}
              className="text-[#2D6A4F] hover:underline disabled:opacity-30 transition-colors text-[14px] font-medium"
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
            Your Bookmarks
          </h1>
          <p className="text-[15px] text-[#6B6B6B] mt-1">
            Quick access to your saved study materials.
          </p>
        </div>

        {/* Revision Mode toggle */}
        <div className="flex items-center gap-3 mt-2">
          <span className="text-[13px] font-medium text-[#6B6B6B]">Revision Mode</span>
          <button
            onClick={() => {
              setRevisionMode(true);
              setRevisionIndex(0);
            }}
            className={cn(
              'relative w-10 h-[22px] rounded-full transition-colors flex-shrink-0',
              revisionMode ? 'bg-[#2D6A4F]' : 'bg-[#E2E2E2]'
            )}
          >
            <span
              className={cn(
                'absolute top-[3px] left-[3px] w-4 h-4 bg-white rounded-full shadow transition-transform',
                revisionMode && 'translate-x-[18px]'
              )}
            />
          </button>
        </div>
      </div>

      {/* Search + Filters */}
      <div className="flex flex-col md:flex-row gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search bookmarks..."
            className="w-full pl-9 pr-4 h-10 bg-white border border-[#E2E2E2] rounded-lg text-[14px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
          />
        </div>

        <div className="flex gap-2 flex-wrap">
          {filters.map(f => (
            <button
              key={f}
              onClick={() => setActiveFilter(f)}
              className={cn(
                'px-4 py-2 rounded-full text-[13px] font-medium transition-colors border',
                activeFilter === f
                  ? 'bg-[#2D6A4F] text-white border-[#2D6A4F]'
                  : 'bg-white text-[#6B6B6B] border-[#E2E2E2] hover:border-[#2D6A4F] hover:text-[#2D6A4F]'
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Empty State */}
      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <svg
            width="80"
            height="80"
            viewBox="0 0 80 80"
            fill="none"
            className="mb-6 opacity-50"
          >
            <circle cx="40" cy="32" r="20" stroke="#E2E2E2" strokeWidth="2" />
            <path d="M25 54 Q40 64 55 54" stroke="#E2E2E2" strokeWidth="2" strokeLinecap="round" />
            <path d="M34 28 L34 36 L40 32 L46 36 L46 28" stroke="#E2E2E2" strokeWidth="1.5" strokeLinejoin="round" />
          </svg>
          <h3 className="font-serif italic text-[20px] text-[#1A1A1A] mb-3">
            Save things here as you study.
          </h3>
          <p className="text-[14px] text-[#6B6B6B] max-w-xs">
            Bookmark notes, flashcards, and quiz questions to find them here.
          </p>
        </div>
      ) : (
        <ResponsiveMasonry columnsCountBreakPoints={{ 350: 1, 750: 2, 1000: 3 }}>
          <Masonry gutter="20px">
            {filtered.map((item, i) => (
              <Card
                key={i}
                className={cn(
                  'p-5 flex flex-col gap-3 border-l-[3px]',
                  borderColor(item.type)
                )}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-1.5 bg-[#F0F0EE] px-2.5 py-1 rounded-full">
                    {typeIcon(item.type)}
                    <span className="text-[12px] font-medium text-[#6B6B6B]">
                      {typeLabel(item.type)}
                    </span>
                  </div>
                  <Bookmark size={16} className="text-[#2D6A4F]" fill="#2D6A4F" />
                </div>

                <div>
                  <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-1">
                    {item.title}
                  </h3>
                  <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em] block mb-2">
                    {item.subject} · {item.meta}
                  </span>
                  <p className="text-[13px] text-[#6B6B6B] line-clamp-3 leading-relaxed">
                    {item.content}
                  </p>
                </div>

                <div className="pt-3 mt-auto border-t border-[#E2E2E2]">
                  <button className="text-[#2D6A4F] text-[13px] font-semibold hover:underline flex items-center gap-1">
                    View <ArrowRight size={13} />
                  </button>
                </div>
              </Card>
            ))}
          </Masonry>
        </ResponsiveMasonry>
      )}
    </div>
  );
}
