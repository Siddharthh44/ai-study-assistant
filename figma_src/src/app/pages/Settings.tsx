import React, { useState } from 'react';
import { Card } from '../components/ui-kit';
import { Check, ChevronDown } from 'lucide-react';
import { cn } from '../components/ui-kit';

const navItems = [
  { id: 'profile', label: 'Profile', icon: '👤' },
  { id: 'learning', label: 'Learning Preferences', icon: '🎓' },
  { id: 'ai', label: 'AI Behavior', icon: '🤖' },
  { id: 'notifications', label: 'Notifications', icon: '🔔' },
  { id: 'data', label: 'Data & Storage', icon: '📦' },
  { id: 'security', label: 'Account & Security', icon: '🔐' },
  { id: 'about', label: 'About Nudge', icon: 'ℹ️' },
];

function Toggle({ checked, onChange }: { checked: boolean; onChange: () => void }) {
  return (
    <button
      onClick={onChange}
      className={cn(
        'relative w-10 h-[22px] rounded-full transition-colors flex-shrink-0',
        checked ? 'bg-[#2D6A4F]' : 'bg-[#E2E2E2]'
      )}
    >
      <span
        className={cn(
          'absolute top-[3px] left-[3px] w-4 h-4 bg-white rounded-full shadow transition-transform duration-200',
          checked && 'translate-x-[18px]'
        )}
      />
    </button>
  );
}

function SettingRow({
  label,
  sub,
  control,
  last = false,
}: {
  label: string;
  sub?: string;
  control: React.ReactNode;
  last?: boolean;
}) {
  return (
    <div className={cn('flex items-center justify-between gap-6 py-4', !last && 'border-b border-[#E2E2E2]')}>
      <div className="min-w-0">
        <p className="text-[15px] font-medium text-[#1A1A1A]">{label}</p>
        {sub && <p className="text-[13px] text-[#6B6B6B] mt-0.5">{sub}</p>}
      </div>
      <div className="flex-shrink-0">{control}</div>
    </div>
  );
}

function SegmentedControl({
  options,
  value,
  onChange,
}: {
  options: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex border border-[#E2E2E2] rounded-lg overflow-hidden">
      {options.map(opt => (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          className={cn(
            'px-4 py-1.5 text-[13px] font-medium transition-colors',
            value === opt
              ? 'bg-[#2D6A4F] text-white'
              : 'bg-white text-[#6B6B6B] hover:bg-[#F0F0EE]'
          )}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}

function Stepper({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center border border-[#E2E2E2] rounded-full overflow-hidden">
      <button
        onClick={() => onChange(Math.max(1, value - 1))}
        className="w-8 h-8 flex items-center justify-center text-[#6B6B6B] hover:bg-[#F0F0EE] transition-colors text-[16px]"
      >
        –
      </button>
      <span className="font-mono text-[13px] text-[#1A1A1A] w-8 text-center tracking-[0.03em]">
        {value}
      </span>
      <button
        onClick={() => onChange(value + 1)}
        className="w-8 h-8 flex items-center justify-center text-[#6B6B6B] hover:bg-[#F0F0EE] transition-colors text-[16px]"
      >
        +
      </button>
    </div>
  );
}

function ProfileSection() {
  const [name, setName] = useState('Arjun Sharma');
  const [email] = useState('arjun@example.com');
  const [institution, setInstitution] = useState('IIT Preparation Institute');
  const [exam, setExam] = useState('JEE Advanced');
  const [examDate, setExamDate] = useState('2026-05-20');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div className="space-y-6">
      {/* Avatar */}
      <div className="flex items-center gap-5">
        <div className="w-16 h-16 rounded-full bg-[#2D6A4F] flex items-center justify-center flex-shrink-0">
          <span className="font-serif text-[22px] font-medium text-white">AR</span>
        </div>
        <div>
          <p className="text-[15px] font-medium text-[#1A1A1A]">{name}</p>
          <button className="text-[13px] text-[#2D6A4F] hover:underline mt-0.5">
            Change photo
          </button>
        </div>
      </div>

      <Card className="p-6 space-y-4">
        <div>
          <label className="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">Full Name</label>
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            className="w-full h-10 px-4 bg-white border border-[#E2E2E2] rounded-lg text-[14px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
          />
        </div>

        <div>
          <label className="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">Email</label>
          <div className="flex items-center gap-3">
            <input
              value={email}
              readOnly
              className="flex-1 h-10 px-4 bg-[#F4F4F2] border border-[#E2E2E2] rounded-lg text-[14px] text-[#6B6B6B] focus:outline-none"
            />
            <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-semibold px-3 py-1.5 rounded-full whitespace-nowrap">
              Verified ✓
            </span>
          </div>
        </div>

        <div>
          <label className="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">Institution</label>
          <input
            value={institution}
            onChange={e => setInstitution(e.target.value)}
            className="w-full h-10 px-4 bg-white border border-[#E2E2E2] rounded-lg text-[14px] focus:outline-none focus:border-[#2D6A4F] transition-colors"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">Exam Goal</label>
            <div className="relative">
              <select
                value={exam}
                onChange={e => setExam(e.target.value)}
                className="w-full h-10 px-4 bg-white border border-[#E2E2E2] rounded-lg text-[14px] focus:outline-none focus:border-[#2D6A4F] appearance-none cursor-pointer"
              >
                {['JEE Advanced', 'NEET', 'UPSC', 'CAT', 'GATE', 'Other'].map(o => (
                  <option key={o}>{o}</option>
                ))}
              </select>
              <ChevronDown
                size={14}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-[#6B6B6B] pointer-events-none"
              />
            </div>
          </div>
          <div>
            <label className="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">Exam Date</label>
            <input
              type="date"
              value={examDate}
              onChange={e => setExamDate(e.target.value)}
              className="w-full h-10 px-4 bg-white border border-[#E2E2E2] rounded-lg text-[14px] focus:outline-none focus:border-[#2D6A4F] cursor-pointer"
            />
          </div>
        </div>
      </Card>

      <div className="flex items-center gap-4">
        <button
          onClick={handleSave}
          className="flex items-center gap-2 bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors"
        >
          {saved && <Check size={16} />}
          {saved ? 'Saved!' : 'Save Changes →'}
        </button>
        <p className="text-[13px] text-[#6B6B6B]">Last updated 3 days ago</p>
      </div>
    </div>
  );
}

function AIBehaviorSection() {
  const [noteLength, setNoteLength] = useState('Balanced');
  const [tone, setTone] = useState('Academic');
  const [autoFlashcards, setAutoFlashcards] = useState(true);
  const [autoQuiz, setAutoQuiz] = useState(false);
  const [reminders, setReminders] = useState(true);
  const [questionsPerQuiz, setQuestionsPerQuiz] = useState(10);
  const [reminderTime, setReminderTime] = useState('09:00');
  const [difficulty, setDifficulty] = useState(65);

  return (
    <div className="space-y-5">
      <Card className="p-6">
        <h4 className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-1">Note Generation</h4>
        <p className="text-[13px] text-[#6B6B6B] mb-4">Control how Nudge generates your study notes.</p>
        <SettingRow
          label="Note length"
          sub="How much detail should your notes include?"
          control={
            <SegmentedControl
              options={['Brief', 'Balanced', 'Detailed']}
              value={noteLength}
              onChange={setNoteLength}
            />
          }
        />
        <SettingRow
          label="Writing tone"
          sub="Style of language used in generated notes."
          control={
            <div className="relative">
              <select
                value={tone}
                onChange={e => setTone(e.target.value)}
                className="h-9 pl-4 pr-8 bg-white border border-[#E2E2E2] rounded-lg text-[13px] focus:outline-none focus:border-[#2D6A4F] appearance-none cursor-pointer"
              >
                {['Academic', 'Conversational', 'Mixed'].map(o => <option key={o}>{o}</option>)}
              </select>
              <ChevronDown size={13} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#6B6B6B] pointer-events-none" />
            </div>
          }
          last
        />
      </Card>

      <Card className="p-6">
        <h4 className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-1">Flashcards & Quizzes</h4>
        <p className="text-[13px] text-[#6B6B6B] mb-4">Customize auto-generation settings.</p>
        <SettingRow
          label="Auto-generate flashcards"
          sub="Create flashcards from every new note automatically."
          control={<Toggle checked={autoFlashcards} onChange={() => setAutoFlashcards(v => !v)} />}
        />
        <SettingRow
          label="Auto-generate quiz"
          sub="Generate a quiz whenever flashcards are created."
          control={<Toggle checked={autoQuiz} onChange={() => setAutoQuiz(v => !v)} />}
        />
        <SettingRow
          label="Questions per quiz"
          sub="Default number of questions in auto-generated quizzes."
          control={<Stepper value={questionsPerQuiz} onChange={setQuestionsPerQuiz} />}
        />
        <SettingRow
          label="Default difficulty"
          sub="Slider from easy (0) to challenging (100)."
          control={
            <div className="flex items-center gap-3">
              <span className="font-mono text-[11px] text-[#6B6B6B]">Easy</span>
              <div className="w-36 relative">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={difficulty}
                  onChange={e => setDifficulty(Number(e.target.value))}
                  className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #2D6A4F ${difficulty}%, #E2E2E2 ${difficulty}%)`,
                  }}
                />
              </div>
              <span className="font-mono text-[11px] text-[#6B6B6B]">Hard</span>
            </div>
          }
          last
        />
      </Card>

      <Card className="p-6">
        <h4 className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-1">Reminders</h4>
        <p className="text-[13px] text-[#6B6B6B] mb-4">Get gentle nudges to keep your study streak going.</p>
        <SettingRow
          label="Daily study reminder"
          sub="Receive a daily notification to review your flashcards."
          control={<Toggle checked={reminders} onChange={() => setReminders(v => !v)} />}
        />
        <SettingRow
          label="Reminder time"
          sub="What time should Nudge remind you to study?"
          control={
            <input
              type="time"
              value={reminderTime}
              onChange={e => setReminderTime(e.target.value)}
              disabled={!reminders}
              className="h-9 px-3 bg-white border border-[#E2E2E2] rounded-lg text-[13px] focus:outline-none focus:border-[#2D6A4F] disabled:opacity-40 cursor-pointer"
            />
          }
          last
        />
      </Card>
    </div>
  );
}

export function Settings() {
  const [activeSection, setActiveSection] = useState('profile');

  return (
    <div className="animate-in fade-in duration-500">
      <div className="mb-8">
        <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
          Settings
        </h1>
      </div>

      <div className="flex gap-8">
        {/* Left Nav */}
        <aside className="w-[240px] flex-shrink-0">
          <ul className="space-y-0.5">
            {navItems.map(item => (
              <li key={item.id}>
                <button
                  onClick={() => setActiveSection(item.id)}
                  className={cn(
                    'w-full flex items-center gap-3 h-10 px-4 rounded-lg transition-all text-[14px] font-medium relative',
                    activeSection === item.id
                      ? 'bg-[#D8E8E0]/50 text-[#2D6A4F]'
                      : 'text-[#6B6B6B] hover:bg-[#F0F0EE] hover:text-[#1A1A1A]'
                  )}
                >
                  {activeSection === item.id && (
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-6 bg-[#2D6A4F] rounded-r-full" />
                  )}
                  <span>{item.icon}</span>
                  <span>{item.label}</span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        {/* Right Content */}
        <div className="flex-1 min-w-0">
          {activeSection === 'profile' && <ProfileSection />}
          {activeSection === 'ai' && <AIBehaviorSection />}
          {activeSection === 'learning' && (
            <div className="space-y-6">
              <div>
                <h2 className="font-serif text-[22px] font-medium text-[#1A1A1A] mb-1">
                  Learning Preferences
                </h2>
                <p className="text-[14px] text-[#6B6B6B]">
                  Customize how Nudge supports your learning style.
                </p>
              </div>
              <Card className="p-6">
                <SettingRow
                  label="Spaced repetition"
                  sub="Use SRS algorithms to schedule your flashcard reviews."
                  control={<Toggle checked={true} onChange={() => {}} />}
                />
                <SettingRow
                  label="Show answer hints"
                  sub="Show partial hints before revealing flashcard answers."
                  control={<Toggle checked={false} onChange={() => {}} />}
                />
                <SettingRow
                  label="Progress tracking"
                  sub="Track which topics you've reviewed and how well."
                  control={<Toggle checked={true} onChange={() => {}} />}
                  last
                />
              </Card>
            </div>
          )}
          {(activeSection === 'notifications' ||
            activeSection === 'data' ||
            activeSection === 'security' ||
            activeSection === 'about') && (
            <div className="space-y-6">
              <h2 className="font-serif text-[22px] font-medium text-[#1A1A1A]">
                {navItems.find(n => n.id === activeSection)?.label}
              </h2>
              <Card className="p-6">
                <p className="text-[15px] text-[#6B6B6B]">
                  This section is coming soon. We're working on it!
                </p>
              </Card>
            </div>
          )}

          {/* Delete account */}
          <div className="mt-12 pt-6 border-t border-[#E2E2E2]">
            <button className="text-[13px] text-[#6B6B6B] hover:text-[#C0392B] transition-colors">
              Delete my account
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
