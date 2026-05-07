import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { Card } from '../components/ui-kit';
import { GraduationCap, Clock, TrendingUp, Plus, Search, ArrowRight } from 'lucide-react';
import { cn } from '../components/ui-kit';

const quizzes = [
  {
    id: '1',
    title: 'Photosynthesis & Plant Biology',
    subject: 'Biology',
    questions: 10,
    lastScore: 80,
    attempted: true,
    date: '2h ago',
  },
  {
    id: '2',
    title: 'Organic Chemistry Fundamentals',
    subject: 'Chemistry',
    questions: 12,
    lastScore: 65,
    attempted: true,
    date: 'Yesterday',
  },
  {
    id: '3',
    title: "Newton's Laws & Motion",
    subject: 'Physics',
    questions: 8,
    lastScore: null,
    attempted: false,
    date: '2 days ago',
  },
  {
    id: '4',
    title: 'World War II Key Events',
    subject: 'History',
    questions: 15,
    lastScore: 90,
    attempted: true,
    date: '3 days ago',
  },
  {
    id: '5',
    title: 'Cell Division: Mitosis & Meiosis',
    subject: 'Biology',
    questions: 10,
    lastScore: null,
    attempted: false,
    date: '1 week ago',
  },
];

function ScoreBadge({ score }: { score: number }) {
  if (score >= 80)
    return (
      <span className="font-mono text-[11px] font-medium bg-[#D8E8E0] text-[#2D6A4F] px-2 py-0.5 rounded-full">
        {score}%
      </span>
    );
  if (score >= 60)
    return (
      <span className="font-mono text-[11px] font-medium bg-[#F0F0EE] text-[#6B6B6B] px-2 py-0.5 rounded-full">
        {score}%
      </span>
    );
  return (
    <span className="font-mono text-[11px] font-medium bg-red-50 text-[#C0392B] px-2 py-0.5 rounded-full">
      {score}%
    </span>
  );
}

export function Quizzes() {
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('All');
  const navigate = useNavigate();

  const filtered = quizzes.filter(q => {
    const matchesFilter =
      filter === 'All' ||
      (filter === 'Attempted' && q.attempted) ||
      (filter === 'New' && !q.attempted);
    const matchesSearch =
      !search ||
      q.title.toLowerCase().includes(search.toLowerCase()) ||
      q.subject.toLowerCase().includes(search.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
            Quizzes
          </h1>
          <p className="text-[15px] text-[#6B6B6B] mt-1">
            Test yourself and track your knowledge over time.
          </p>
        </div>
        <button
          onClick={() => navigate('/upload')}
          className="flex items-center gap-2 bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors mt-1"
        >
          <Plus size={16} />
          Generate Quiz
        </button>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-5">
        {[
          { label: 'Total Quizzes', value: quizzes.length, icon: GraduationCap },
          { label: 'Avg Score', value: '78%', icon: TrendingUp },
          { label: 'Time Saved', value: '4.2h', icon: Clock },
        ].map((s, i) => (
          <Card key={i} className="p-5 flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-[#F4F4F2] flex items-center justify-center flex-shrink-0">
              <s.icon size={18} className="text-[#2D6A4F]" />
            </div>
            <div>
              <p className="text-[13px] text-[#6B6B6B]">{s.label}</p>
              <p className="font-serif text-[22px] font-medium text-[#1A1A1A] leading-tight">
                {s.value}
              </p>
            </div>
          </Card>
        ))}
      </div>

      {/* Controls */}
      <div className="flex flex-col md:flex-row gap-3 items-start md:items-center justify-between">
        <div className="flex gap-2">
          {['All', 'New', 'Attempted'].map(f => (
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
        <div className="relative">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search quizzes..."
            className="pl-9 pr-4 h-9 bg-white border border-[#E2E2E2] rounded-lg text-[13px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] w-48 transition-colors"
          />
        </div>
      </div>

      {/* Quiz Cards */}
      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="mb-6 opacity-50">
            <circle cx="40" cy="40" r="28" stroke="#E2E2E2" strokeWidth="2" />
            <text x="40" y="48" textAnchor="middle" fill="#E2E2E2" fontSize="24" fontFamily="serif">?</text>
          </svg>
          <h3 className="font-serif italic text-[20px] text-[#1A1A1A] mb-3">
            No quizzes here yet.
          </h3>
          <p className="text-[14px] text-[#6B6B6B] mb-6 max-w-xs">
            Upload study material and Nudge will generate quizzes for you.
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors"
          >
            Upload Material
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map(quiz => (
            <Card key={quiz.id} className="p-5 flex items-center gap-5 group">
              <div className="w-10 h-10 rounded-full bg-[#F4F4F2] flex items-center justify-center flex-shrink-0">
                <GraduationCap size={18} className="text-[#2D6A4F]" />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] truncate">
                    {quiz.title}
                  </h3>
                  {!quiz.attempted && (
                    <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2 py-0.5 rounded-full flex-shrink-0">
                      New
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <span className="bg-[#F0F0EE] text-[#6B6B6B] text-[11px] font-medium px-2 py-0.5 rounded-full">
                    {quiz.subject}
                  </span>
                  <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                    {quiz.questions} questions
                  </span>
                  <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                    {quiz.date}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-4 flex-shrink-0">
                {quiz.lastScore !== null && <ScoreBadge score={quiz.lastScore} />}
                <Link to="/quizzes/start">
                  <button className="bg-[#2D6A4F] text-white text-[13px] font-semibold px-4 py-2 rounded-lg hover:bg-[#245C43] transition-colors flex items-center gap-1.5">
                    {quiz.attempted ? 'Retake' : 'Start'}
                    <ArrowRight size={13} />
                  </button>
                </Link>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
