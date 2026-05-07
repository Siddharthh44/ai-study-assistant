import React, { useState } from 'react';
import { useParams, Link } from 'react-router';
import { Button, Badge, Card } from '../components/ui-kit';
import { Pencil, Bookmark, Share2, ThumbsUp, ThumbsDown, ArrowRight, FileText } from 'lucide-react';
import { cn } from '../components/ui-kit';

export function NoteView() {
  const { id } = useParams();
  const [title, setTitle] = useState('Photosynthesis & Plant Biology');
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [bookmarked, setBookmarked] = useState(false);
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in duration-500">
      {/* LEFT: Note Document (65% → 8 cols) */}
      <div className="lg:col-span-8 space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-3">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              {isEditingTitle ? (
                <input
                  type="text"
                  value={title}
                  onChange={e => setTitle(e.target.value)}
                  onBlur={() => setIsEditingTitle(false)}
                  autoFocus
                  className="font-serif text-[28px] font-medium text-[#1A1A1A] w-full bg-transparent border-b-2 border-[#2D6A4F] focus:outline-none pb-1"
                />
              ) : (
                <h1
                  onClick={() => setIsEditingTitle(true)}
                  className="font-serif text-[28px] font-medium text-[#1A1A1A] cursor-text hover:bg-[#F0F0EE] rounded px-1 -ml-1 transition-colors"
                >
                  {title}
                </h1>
              )}
              <div className="flex items-center gap-3 mt-2">
                <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">
                  Biology
                </span>
                <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                  Generated 2h ago · March 4, 2026
                </span>
              </div>
            </div>

            {/* Icon Buttons */}
            <div className="flex items-center gap-1">
              {[
                { icon: Pencil, label: 'Edit', onClick: () => setIsEditingTitle(true) },
                { icon: Bookmark, label: 'Bookmark', onClick: () => setBookmarked(b => !b) },
                { icon: Share2, label: 'Export', onClick: () => {} },
              ].map(({ icon: Icon, label, onClick }) => (
                <button
                  key={label}
                  onClick={onClick}
                  title={label}
                  className={cn(
                    'w-9 h-9 rounded-lg flex items-center justify-center transition-colors',
                    label === 'Bookmark' && bookmarked
                      ? 'text-[#2D6A4F] bg-[#D8E8E0]'
                      : 'text-[#6B6B6B] hover:text-[#2D6A4F] hover:bg-[#F0F0EE]'
                  )}
                >
                  <Icon size={18} />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Note Content Card */}
        <Card className="p-10 min-h-[600px]">
          <h2 className="font-serif text-[22px] font-medium text-[#1A1A1A] mb-4">
            1. The Process of Photosynthesis
          </h2>
          <p className="text-[15px] text-[#1A1A1A] leading-relaxed mb-6">
            Photosynthesis is the process by which green plants and some other organisms use
            sunlight to synthesize foods with the help of{' '}
            <span className="bg-[#D8E8E0] px-1 rounded">chlorophyll</span>. The general
            equation is:{' '}
            <span className="font-mono bg-[#F0F0EE] px-1.5 py-0.5 rounded text-[13px]">
              6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂
            </span>
          </p>

          {/* Callout Block */}
          <div className="my-6 pl-4 border-l-[3px] border-l-[#2D6A4F] bg-[#D8E8E0]/25 p-4 rounded-r-lg">
            <h4 className="font-serif text-[17px] font-medium text-[#2D6A4F] mb-1.5">
              Key Concept: Chlorophyll
            </h4>
            <p className="text-[14px] text-[#1A1A1A] leading-relaxed">
              Chlorophyll is the green pigment found in the chloroplasts of algae and plants.
              It absorbs light energy primarily in the blue and red wavelengths, enabling
              photosynthesis to proceed.
            </p>
          </div>

          <h3 className="font-serif text-[17px] text-[#6B6B6B] font-medium mt-8 mb-3">
            1.1 Light-Dependent Reactions
          </h3>
          <ul className="space-y-2 mb-6 pl-1">
            {[
              <>Occur in the <span className="bg-[#D8E8E0] px-1 rounded">thylakoid membranes</span>.</>,
              'Convert light energy into chemical energy (ATP and NADPH).',
              'Release oxygen as a byproduct through photolysis of water.',
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-3 text-[15px] text-[#1A1A1A] leading-relaxed">
                <span className="w-1.5 h-1.5 rounded-sm bg-[#2D6A4F] mt-2 flex-shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <h3 className="font-serif text-[17px] text-[#6B6B6B] font-medium mt-6 mb-3">
            1.2 The Calvin Cycle (Light-Independent)
          </h3>
          <p className="text-[15px] text-[#1A1A1A] leading-relaxed mb-4">
            The Calvin cycle takes place in the{' '}
            <span className="bg-[#D8E8E0] px-1 rounded">stroma</span> of the chloroplast.
            It uses ATP and NADPH to convert CO₂ into{' '}
            <span className="bg-[#D8E8E0] px-1 rounded">G3P</span> (glyceraldehyde-3-phosphate),
            which is later used to build glucose.
          </p>

          <h2 className="font-serif text-[22px] font-medium text-[#1A1A1A] mt-8 mb-4">
            2. Factors Affecting Photosynthesis
          </h2>
          <ul className="space-y-2 pl-1">
            {[
              'Light intensity — increases rate up to a saturation point.',
              'CO₂ concentration — higher levels can boost the rate.',
              'Temperature — optimal range is 25–35°C for most plants.',
              'Water availability — essential as a reactant and for stomatal opening.',
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-3 text-[15px] text-[#1A1A1A] leading-relaxed">
                <span className="w-1.5 h-1.5 rounded-sm bg-[#2D6A4F] mt-2 flex-shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </Card>

        {/* Feedback row */}
        <div className="flex items-center justify-between px-1">
          <div className="flex items-center gap-4">
            <span className="text-[13px] text-[#6B6B6B]">Was this helpful?</span>
            <button
              onClick={() => setFeedback('up')}
              className={cn(
                'transition-colors',
                feedback === 'up' ? 'text-[#2D6A4F]' : 'text-[#6B6B6B] hover:text-[#2D6A4F]'
              )}
            >
              <ThumbsUp size={18} />
            </button>
            <button
              onClick={() => setFeedback('down')}
              className={cn(
                'transition-colors',
                feedback === 'down' ? 'text-[#C0392B]' : 'text-[#6B6B6B] hover:text-[#C0392B]'
              )}
            >
              <ThumbsDown size={18} />
            </button>
          </div>
          <button className="text-[13px] text-[#2D6A4F] hover:underline font-medium">
            Edit this note
          </button>
        </div>
      </div>

      {/* RIGHT: Smart Sidebar (35% → 4 cols) */}
      <div className="lg:col-span-4 space-y-4 sticky top-0">
        {/* Key Concepts */}
        <Card className="p-5">
          <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-3">Key Concepts</h3>
          <div className="flex flex-wrap gap-2">
            {['Chlorophyll', 'Thylakoid', 'Stroma', 'ATP', 'NADPH', 'Calvin Cycle', 'Photolysis', 'G3P'].map(
              tag => (
                <span
                  key={tag}
                  className="bg-[#D8E8E0] text-[#2D6A4F] text-[12px] font-medium px-2.5 py-1 rounded-full cursor-pointer hover:bg-[#2D6A4F] hover:text-white transition-colors"
                >
                  {tag}
                </span>
              )
            )}
          </div>
        </Card>

        {/* Flashcards Ready */}
        <Card className="p-5 bg-[#D8E8E0]/10">
          <div className="flex items-start justify-between mb-2">
            <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A]">Flashcards Ready</h3>
            <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">
              12 cards
            </span>
          </div>
          <p className="text-[13px] text-[#6B6B6B] mb-4">
            Review the key terms from this note.
          </p>
          <Link to="/app/flashcards">
            <Button variant="secondary" fullWidth>
              View Flashcards →
            </Button>
          </Link>
        </Card>

        {/* Quiz Available */}
        <Card className="p-5 border-[#2D6A4F] bg-[#D8E8E0]/5">
          <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-2">
            Quiz Available
          </h3>
          <p className="text-[13px] text-[#6B6B6B] mb-4">
            Test your knowledge on this topic. 10 questions ready.
          </p>
          <Link to="/app/quizzes/start">
            <Button fullWidth>Start Quiz →</Button>
          </Link>
        </Card>

        {/* Export */}
        <Card className="p-5">
          <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-3">Export</h3>
          <div className="grid grid-cols-2 gap-2">
            <Button variant="secondary" size="sm" className="flex items-center justify-center gap-2">
              <FileText size={14} /> PDF
            </Button>
            <Button variant="secondary" size="sm" className="flex items-center justify-center gap-2">
              <FileText size={14} /> TXT
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}