import React, { useState } from 'react';
import { Card } from '../components/ui-kit';
import { Search, ArrowRight, Bookmark, Share2, Eye, ChevronRight, X } from 'lucide-react';
import { cn } from '../components/ui-kit';

const historyGroups = [
  {
    date: 'Today · March 4, 2026',
    items: [
      {
        id: 1,
        type: 'note',
        title: 'Photosynthesis & Plant Biology',
        subject: 'Biology',
        source: 'PDF upload',
        time: '2h ago',
        preview: 'Covers light-dependent reactions, the Calvin cycle, and chloroplast structure.',
      },
      {
        id: 2,
        type: 'quiz',
        title: 'Biology Quiz: Photosynthesis',
        subject: 'Biology',
        source: 'Auto-generated',
        time: '1h ago',
        preview: 'Score: 80% · 8 of 10 correct. Strong on light reactions, review Calvin cycle.',
      },
      {
        id: 3,
        type: 'flashcard',
        title: 'Photosynthesis Flashcards',
        subject: 'Biology',
        source: 'From notes',
        time: '45m ago',
        preview: 'Reviewed 12 cards. Got it: 9 · Almost there: 2 · Still learning: 1',
      },
    ],
  },
  {
    date: 'Yesterday · March 3, 2026',
    items: [
      {
        id: 4,
        type: 'note',
        title: 'Organic Chemistry: Alkanes & Alkenes',
        subject: 'Chemistry',
        source: 'Text input',
        time: '6h ago',
        preview: 'Structural formulas, IUPAC naming conventions, physical and chemical properties.',
      },
      {
        id: 5,
        type: 'pyq',
        title: 'Chemistry PYQ Analysis',
        subject: 'Chemistry',
        source: 'PYQ upload',
        time: '8h ago',
        preview: 'Analyzed 148 questions. Top topics: Organic reactions, Equilibrium, Electrochemistry.',
      },
    ],
  },
  {
    date: 'March 2, 2026',
    items: [
      {
        id: 6,
        type: 'note',
        title: "Newton's Laws of Motion",
        subject: 'Physics',
        source: 'Audio recording',
        time: '2 days ago',
        preview: 'Comprehensive coverage of all three laws with worked examples and diagrams.',
      },
      {
        id: 7,
        type: 'quiz',
        title: 'Physics Quiz: Mechanics',
        subject: 'Physics',
        source: 'Auto-generated',
        time: '2 days ago',
        preview: 'Score: 90% · 9 of 10 correct. Excellent understanding of Newtonian mechanics.',
      },
    ],
  },
];

function TypeBadge({ type }: { type: string }) {
  const config = {
    note: { bg: 'bg-[#2D6A4F]', text: 'text-white', label: 'Note' },
    flashcard: { bg: 'bg-[#52796F]', text: 'text-white', label: 'Flashcard' },
    quiz: { bg: 'bg-[#D8E8E0]', text: 'text-[#2D6A4F]', label: 'Quiz' },
    pyq: { bg: 'bg-[#F0F0EE]', text: 'text-[#6B6B6B]', label: 'PYQ' },
  }[type] || { bg: 'bg-[#F0F0EE]', text: 'text-[#6B6B6B]', label: type };

  const initials = config.label.charAt(0);

  return (
    <div
      className={cn(
        'w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 font-mono text-[12px] font-medium',
        config.bg,
        config.text
      )}
    >
      {initials}
    </div>
  );
}

type VersionItem = {
  version: string;
  date: string;
  preview: string;
};

const versions: VersionItem[] = [
  { version: 'v3 (Current)', date: '2h ago', preview: 'Added Calvin cycle details and key concept callouts.' },
  { version: 'v2', date: '4h ago', preview: 'Expanded light reactions section with thylakoid membrane diagram.' },
  { version: 'v1', date: '6h ago', preview: 'Initial generation from uploaded PDF.' },
];

export function History() {
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('All');
  const [sort] = useState('Newest first');
  const [panelOpen, setPanelOpen] = useState(false);
  const [selectedVersion, setSelectedVersion] = useState(0);

  const typeFilter: Record<string, string> = {
    Notes: 'note',
    Flashcards: 'flashcard',
    Quizzes: 'quiz',
    PYQ: 'pyq',
  };

  const filteredGroups = historyGroups.map(g => ({
    ...g,
    items: g.items.filter(item => {
      const matchesFilter = filter === 'All' || item.type === typeFilter[filter];
      const matchesSearch =
        !search ||
        item.title.toLowerCase().includes(search.toLowerCase()) ||
        item.subject.toLowerCase().includes(search.toLowerCase());
      return matchesFilter && matchesSearch;
    }),
  })).filter(g => g.items.length > 0);

  return (
    <div className="flex gap-0 animate-in fade-in duration-500 relative">
      {/* Main Content */}
      <div className={cn('flex-1 space-y-8 transition-all duration-300', panelOpen && 'mr-[360px]')}>
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
              Your Study Journal
            </h1>
            <p className="text-[15px] text-[#6B6B6B] mt-1">
              A log of everything you've studied and created.
            </p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-col md:flex-row gap-3 items-start md:items-center justify-between">
          <div className="flex gap-2 flex-wrap">
            {['All', 'Notes', 'Flashcards', 'Quizzes', 'PYQ'].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={cn(
                  'px-4 py-1.5 rounded-full text-[13px] font-medium transition-all border',
                  filter === f
                    ? 'bg-[#2D6A4F] text-white border-[#2D6A4F]'
                    : 'bg-white text-[#6B6B6B] border-[#E2E2E2] hover:border-[#2D6A4F] hover:text-[#2D6A4F]'
                )}
              >
                {f}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <div className="relative">
              <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]" />
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search history..."
                className="pl-9 pr-4 h-9 bg-white border border-[#E2E2E2] rounded-lg text-[13px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] w-48 transition-colors"
              />
            </div>
            <select className="h-9 px-3 bg-white border border-[#E2E2E2] rounded-lg text-[13px] text-[#1A1A1A] focus:outline-none focus:border-[#2D6A4F] cursor-pointer">
              <option>Newest first</option>
              <option>Oldest first</option>
            </select>
          </div>
        </div>

        {/* Empty State */}
        {filteredGroups.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="mb-6 opacity-50">
              <rect x="16" y="16" width="48" height="48" rx="8" stroke="#E2E2E2" strokeWidth="2" />
              <line x1="30" y1="30" x2="30" y2="50" stroke="#E2E2E2" strokeWidth="2" strokeLinecap="round" />
              <line x1="30" y1="30" x2="50" y2="30" stroke="#E2E2E2" strokeWidth="2" strokeLinecap="round" />
              <circle cx="50" cy="50" r="4" stroke="#E2E2E2" strokeWidth="2" />
            </svg>
            <h3 className="font-serif italic text-[20px] text-[#1A1A1A] mb-3">
              Your study journal starts today.
            </h3>
            <p className="text-[14px] text-[#6B6B6B] max-w-xs">
              Every note, flashcard, and quiz will appear here as you study.
            </p>
          </div>
        ) : (
          <div className="space-y-8">
            {filteredGroups.map(group => (
              <div key={group.date}>
                {/* Date Header */}
                <div className="flex items-center gap-4 mb-4">
                  <span className="font-mono text-[11px] text-[#6B6B6B] uppercase tracking-[0.03em] whitespace-nowrap">
                    {group.date}
                  </span>
                  <div className="flex-1 h-[1px] bg-[#E2E2E2]" />
                </div>

                {/* Items */}
                <div className="space-y-3">
                  {group.items.map(item => (
                    <Card key={item.id} className="p-5 flex items-center gap-4">
                      <TypeBadge type={item.type} />

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-0.5">
                          <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] truncate">
                            {item.title}
                          </h3>
                        </div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2 py-0.5 rounded-full">
                            {item.subject}
                          </span>
                          <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                            {item.source}
                          </span>
                          <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                            · {item.time}
                          </span>
                        </div>
                        <p className="text-[13px] text-[#6B6B6B] line-clamp-1">{item.preview}</p>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-1 flex-shrink-0">
                        <button
                          title="View"
                          className="w-8 h-8 rounded-lg flex items-center justify-center text-[#6B6B6B] hover:text-[#2D6A4F] hover:bg-[#F0F0EE] transition-colors"
                        >
                          <Eye size={16} />
                        </button>
                        <button
                          title="Bookmark"
                          className="w-8 h-8 rounded-lg flex items-center justify-center text-[#6B6B6B] hover:text-[#2D6A4F] hover:bg-[#F0F0EE] transition-colors"
                        >
                          <Bookmark size={16} />
                        </button>
                        <button
                          title="Export"
                          className="w-8 h-8 rounded-lg flex items-center justify-center text-[#6B6B6B] hover:text-[#2D6A4F] hover:bg-[#F0F0EE] transition-colors"
                        >
                          <Share2 size={16} />
                        </button>
                        {item.type === 'note' && (
                          <button
                            title="Version history"
                            onClick={() => setPanelOpen(true)}
                            className="w-8 h-8 rounded-lg flex items-center justify-center text-[#6B6B6B] hover:text-[#2D6A4F] hover:bg-[#F0F0EE] transition-colors"
                          >
                            <ChevronRight size={16} />
                          </button>
                        )}
                      </div>
                    </Card>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Version History Side Panel */}
      {panelOpen && (
        <div className="fixed right-0 top-0 h-full w-[360px] bg-white border-l border-[#E2E2E2] shadow-[0_8px_32px_rgba(0,0,0,0.10)] z-40 flex flex-col">
          <div className="flex items-center justify-between px-5 py-4 border-b border-[#E2E2E2]">
            <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A]">Version History</h3>
            <button
              onClick={() => setPanelOpen(false)}
              className="text-[#6B6B6B] hover:text-[#1A1A1A] transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-5 space-y-3">
            {versions.map((v, i) => (
              <button
                key={i}
                onClick={() => setSelectedVersion(i)}
                className={cn(
                  'w-full text-left p-4 rounded-xl border transition-all',
                  selectedVersion === i
                    ? 'border-[#2D6A4F] bg-[#D8E8E0]/20'
                    : 'border-[#E2E2E2] hover:border-[#2D6A4F]/30 hover:bg-[#F0F0EE]'
                )}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-[14px] text-[#1A1A1A]">{v.version}</span>
                  <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                    {v.date}
                  </span>
                </div>
                <p className="text-[13px] text-[#6B6B6B] leading-relaxed">{v.preview}</p>
              </button>
            ))}
          </div>

          <div className="p-5 border-t border-[#E2E2E2]">
            <p className="text-[13px] text-[#6B6B6B] mb-3">
              Preview: {versions[selectedVersion].preview}
            </p>
            <button className="w-full bg-[#2D6A4F] text-white text-[14px] font-semibold py-2.5 rounded-lg hover:bg-[#245C43] transition-colors">
              Restore this version
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
