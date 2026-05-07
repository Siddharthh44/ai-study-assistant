import React from 'react';
import { Card, Button, Badge } from '../components/ui-kit';
import {
  FileText,
  Clock,
  TrendingUp,
  Flame,
  ArrowRight,
  Plus,
} from 'lucide-react';
import { Link, useNavigate } from 'react-router';

export function Dashboard() {
  const navigate = useNavigate();

  const stats = [
    { label: 'Notes Created', value: '14', icon: FileText },
    { label: 'Flashcards Due', value: '7', icon: Clock, badge: 'Review Today' },
    { label: 'Quiz Average', value: '73%', icon: TrendingUp },
    { label: 'Study Streak', value: '6 days', icon: Flame },
  ];

  const recentNotes = [
    { title: 'Photosynthesis & Plant Biology', subject: 'Biology', date: '2h ago', id: '1' },
    { title: 'Organic Chemistry: Alkanes', subject: 'Chemistry', date: 'Yesterday', id: '2' },
    { title: 'World War II: Key Events', subject: 'History', date: '2 days ago', id: '3' },
  ];

  const dueFlashcards = [
    { topic: 'Cellular Respiration', count: 12, id: 1 },
    { topic: 'Periodic Table Groups', count: 24, id: 2 },
    { topic: 'Calculus Derivatives', count: 8, id: 3 },
  ];

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Page Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
            Good morning, Arjun.
          </h1>
          <p className="text-[15px] text-[#6B6B6B] mt-1">
            You have 3 topics to review and a quiz waiting.
          </p>
        </div>
        <Button onClick={() => navigate('/app/upload')} className="flex items-center gap-2 mt-1">
          <Plus size={16} />
          Add New Material
        </Button>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        {stats.map((stat, i) => (
          <Card key={i} className="p-5 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="text-[13px] text-[#6B6B6B]">{stat.label}</span>
              <div className="w-8 h-8 rounded-lg bg-[#F4F4F2] flex items-center justify-center">
                <stat.icon size={16} className="text-[#2D6A4F]" />
              </div>
            </div>
            <div className="flex items-end justify-between">
              <span className="font-serif text-[28px] font-medium text-[#1A1A1A] leading-none">
                {stat.value}
              </span>
              {stat.badge && (
                <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">
                  {stat.badge}
                </span>
              )}
            </div>
          </Card>
        ))}
      </div>

      {/* Two-column row */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Recently Generated Notes (60%) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-serif text-[22px] font-medium text-[#1A1A1A]">
              Recently Generated Notes
            </h2>
            <Link to="/app/notes" className="text-[13px] text-[#2D6A4F] hover:underline">
              View all
            </Link>
          </div>

          <div className="space-y-3">
            {recentNotes.map(note => (
              <Card
                key={note.id}
                className="p-4 flex items-center justify-between group"
              >
                <div className="flex items-center gap-4">
                  <div className="w-9 h-9 rounded-lg bg-[#F4F4F2] flex items-center justify-center flex-shrink-0">
                    <FileText size={18} className="text-[#2D6A4F]" />
                  </div>
                  <div>
                    <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A]">
                      {note.title}
                    </h3>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">
                        {note.subject}
                      </span>
                      <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                        {note.date}
                      </span>
                    </div>
                  </div>
                </div>
                <Link
                  to={`/app/notes/${note.id}`}
                  className="text-[13px] text-[#2D6A4F] font-semibold hover:underline flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap"
                >
                  Open <ArrowRight size={14} />
                </Link>
              </Card>
            ))}
          </div>
        </div>

        {/* Due for Revision (40%) */}
        <div className="lg:col-span-5 space-y-4">
          <h2 className="font-serif text-[22px] font-medium text-[#1A1A1A]">
            Due for Revision Today
          </h2>
          <Card className="p-0 overflow-hidden">
            <div className="divide-y divide-[#E2E2E2]">
              {dueFlashcards.map(item => (
                <div
                  key={item.id}
                  className="p-4 hover:bg-[#F0F0EE] transition-colors flex items-center justify-between"
                >
                  <div>
                    <p className="text-[15px] font-medium text-[#1A1A1A]">{item.topic}</p>
                    <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                      {item.count} cards
                    </span>
                  </div>
                  <Link to="/app/flashcards">
                    <button className="bg-[#2D6A4F] text-white text-xs font-semibold px-4 py-1.5 rounded-lg hover:bg-[#245C43] transition-colors">
                      Start →
                    </button>
                  </Link>
                </div>
              ))}
            </div>
            <div className="p-3 bg-[#F4F4F2] border-t border-[#E2E2E2] text-center">
              <Link to="/app/flashcards" className="text-[13px] text-[#2D6A4F] hover:underline">
                View all revision tasks
              </Link>
            </div>
          </Card>
        </div>
      </div>

      {/* Continue where you left off */}
      <Card className="border-l-[3px] border-l-[#2D6A4F] flex flex-col md:flex-row md:items-center justify-between gap-4 p-6">
        <div>
          <div className="font-mono text-[11px] text-[#2D6A4F] uppercase tracking-[0.03em] mb-1">
            Continue where you left off
          </div>
          <h3 className="font-serif text-[22px] font-medium text-[#1A1A1A]">
            Photosynthesis &amp; Plant Biology
          </h3>
          <div className="flex items-center gap-2 mt-2">
            <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">
              Biology
            </span>
            <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
              Last edited 2h ago
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <Link to="/app/notes/1">
            <Button variant="secondary">View Notes</Button>
          </Link>
          <Link to="/app/quizzes/start">
            <Button>Generate Quiz</Button>
          </Link>
        </div>
      </Card>
    </div>
  );
}