import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { Card } from '../components/ui-kit';
import { FileText, Search, Plus, ArrowRight, Grid3x3, List } from 'lucide-react';
import { cn } from '../components/ui-kit';

const notes = [
  {
    id: '1',
    title: 'Photosynthesis & Plant Biology',
    subject: 'Biology',
    date: '2h ago',
    preview: 'Photosynthesis is the process by which green plants use sunlight to synthesize food...',
    flashcards: 12,
    hasQuiz: true,
  },
  {
    id: '2',
    title: 'Organic Chemistry: Alkanes & Alkenes',
    subject: 'Chemistry',
    date: 'Yesterday',
    preview: 'Alkanes are saturated hydrocarbons with the general formula CₙH₂ₙ₊₂. They are relatively unreactive...',
    flashcards: 18,
    hasQuiz: true,
  },
  {
    id: '3',
    title: 'World War II: Key Events & Timeline',
    subject: 'History',
    date: '2 days ago',
    preview: 'World War II began in September 1939 with Germany\'s invasion of Poland. The conflict lasted until 1945...',
    flashcards: 24,
    hasQuiz: false,
  },
  {
    id: '4',
    title: "Newton's Laws of Motion",
    subject: 'Physics',
    date: '3 days ago',
    preview: 'The three laws of motion describe the relationship between a body and the forces acting upon it...',
    flashcards: 8,
    hasQuiz: true,
  },
  {
    id: '5',
    title: 'Cell Division: Mitosis & Meiosis',
    subject: 'Biology',
    date: '1 week ago',
    preview: 'Mitosis produces two genetically identical daughter cells. Meiosis produces four genetically diverse gametes...',
    flashcards: 20,
    hasQuiz: false,
  },
  {
    id: '6',
    title: 'Thermodynamics: Laws & Applications',
    subject: 'Physics',
    date: '1 week ago',
    preview: 'The first law of thermodynamics states that energy cannot be created or destroyed, only converted...',
    flashcards: 16,
    hasQuiz: true,
  },
];

export function NotesPage() {
  const [search, setSearch] = useState('');
  const [view, setView] = useState<'grid' | 'list'>('grid');
  const [activeSubject, setActiveSubject] = useState('All');
  const navigate = useNavigate();

  const subjects = ['All', ...Array.from(new Set(notes.map(n => n.subject)))];

  const filtered = notes.filter(n => {
    const matchesSubject = activeSubject === 'All' || n.subject === activeSubject;
    const matchesSearch =
      !search ||
      n.title.toLowerCase().includes(search.toLowerCase()) ||
      n.subject.toLowerCase().includes(search.toLowerCase());
    return matchesSubject && matchesSearch;
  });

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
            My Notes
          </h1>
          <p className="text-[15px] text-[#6B6B6B] mt-1">
            {notes.length} notes generated from your content.
          </p>
        </div>
        <button
          onClick={() => navigate('/upload')}
          className="flex items-center gap-2 bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors mt-1"
        >
          <Plus size={16} />
          New Note
        </button>
      </div>

      {/* Controls */}
      <div className="flex flex-col md:flex-row gap-3 items-start md:items-center justify-between">
        <div className="flex gap-2 flex-wrap">
          {subjects.map(s => (
            <button
              key={s}
              onClick={() => setActiveSubject(s)}
              className={cn(
                'px-4 py-1.5 rounded-full text-[13px] font-medium transition-all border',
                activeSubject === s
                  ? 'bg-[#2D6A4F] text-white border-[#2D6A4F]'
                  : 'bg-white text-[#6B6B6B] border-[#E2E2E2] hover:border-[#2D6A4F] hover:text-[#2D6A4F]'
              )}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search notes..."
              className="pl-9 pr-4 h-9 bg-white border border-[#E2E2E2] rounded-lg text-[13px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] w-48 transition-colors"
            />
          </div>
          <div className="flex border border-[#E2E2E2] rounded-lg overflow-hidden">
            {(['grid', 'list'] as const).map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={cn(
                  'p-2 transition-colors',
                  view === v ? 'bg-[#F0F0EE] text-[#1A1A1A]' : 'bg-white text-[#6B6B6B] hover:text-[#1A1A1A]'
                )}
              >
                {v === 'grid' ? <Grid3x3 size={16} /> : <List size={16} />}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Empty State */}
      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="mb-6 opacity-50">
            <rect x="16" y="12" width="48" height="56" rx="6" stroke="#E2E2E2" strokeWidth="2" />
            <line x1="26" y1="28" x2="54" y2="28" stroke="#E2E2E2" strokeWidth="2" strokeLinecap="round" />
            <line x1="26" y1="38" x2="54" y2="38" stroke="#E2E2E2" strokeWidth="2" strokeLinecap="round" />
            <line x1="26" y1="48" x2="42" y2="48" stroke="#E2E2E2" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <h3 className="font-serif italic text-[20px] text-[#1A1A1A] mb-3">
            Your first note is waiting to be made.
          </h3>
          <p className="text-[14px] text-[#6B6B6B] mb-6 max-w-xs">
            Upload any study material and Nudge will generate structured notes for you.
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors"
          >
            Add Study Material
          </button>
        </div>
      ) : view === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {filtered.map(note => (
            <Link key={note.id} to={`/notes/${note.id}`}>
              <Card className="p-5 h-full flex flex-col gap-3 group cursor-pointer hover:border-[#2D6A4F]/30 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="w-9 h-9 rounded-lg bg-[#F4F4F2] flex items-center justify-center">
                    <FileText size={17} className="text-[#2D6A4F]" />
                  </div>
                  <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                    {note.date}
                  </span>
                </div>

                <div>
                  <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] group-hover:text-[#2D6A4F] transition-colors mb-1">
                    {note.title}
                  </h3>
                  <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">
                    {note.subject}
                  </span>
                </div>

                <p className="text-[13px] text-[#6B6B6B] leading-relaxed line-clamp-2 flex-1">
                  {note.preview}
                </p>

                <div className="flex items-center justify-between pt-3 border-t border-[#E2E2E2]">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                      {note.flashcards} cards
                    </span>
                    {note.hasQuiz && (
                      <span className="bg-[#F0F0EE] text-[#6B6B6B] text-[11px] font-medium px-2 py-0.5 rounded-full">
                        Quiz ready
                      </span>
                    )}
                  </div>
                  <span className="text-[#2D6A4F] text-[13px] font-semibold flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    Open <ArrowRight size={13} />
                  </span>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      ) : (
        <Card className="p-0 overflow-hidden">
          <div className="divide-y divide-[#E2E2E2]">
            {filtered.map((note, i) => (
              <Link key={note.id} to={`/notes/${note.id}`}>
                <div
                  className={cn(
                    'flex items-center gap-5 px-6 py-4 hover:bg-[#F0F0EE] transition-colors group',
                    i % 2 === 1 && 'bg-[#F4F4F2]/40'
                  )}
                >
                  <div className="w-9 h-9 rounded-lg bg-[#F4F4F2] flex items-center justify-center flex-shrink-0">
                    <FileText size={17} className="text-[#2D6A4F]" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] group-hover:text-[#2D6A4F] transition-colors truncate">
                      {note.title}
                    </h3>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2 py-0.5 rounded-full">
                        {note.subject}
                      </span>
                      <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                        {note.date}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 flex-shrink-0">
                    <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                      {note.flashcards} cards
                    </span>
                    <ArrowRight
                      size={16}
                      className="text-[#2D6A4F] opacity-0 group-hover:opacity-100 transition-opacity"
                    />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
