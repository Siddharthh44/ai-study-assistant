import React, { useState } from 'react';
import { Card, Badge, Button } from '../components/ui-kit';
import { cn } from '../components/ui-kit';
import { FileText, Search, Brain, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';

const topics = [
  { name: 'Thermodynamics', frequency: 14, importance: 'High' },
  { name: 'Electrostatics', frequency: 11, importance: 'Medium' },
  { name: 'Organic Chemistry', frequency: 9, importance: 'High' },
  { name: 'Genetics', frequency: 8, importance: 'Low' },
  { name: 'Wave Optics', frequency: 12, importance: 'High' },
  { name: 'Equilibrium', frequency: 7, importance: 'Medium' },
];

const predictions = [
  { topic: 'Wave Optics', confidence: 'high', reason: 'Consistent pattern every 2 years.' },
  { topic: 'Equilibrium', confidence: 'medium', reason: 'High weightage in recent trends.' },
  { topic: 'Plant Physiology', confidence: 'medium', reason: 'Expected due to syllabus changes.' },
  { topic: 'Electrochemistry', confidence: 'high', reason: 'Appeared in 6 of last 8 papers.' },
  { topic: 'Modern Physics', confidence: 'low', reason: 'Low frequency but flagged by AI.' },
];

const pyqs = [
  {
    year: '2023',
    subject: 'Physics',
    text: 'Calculate the capacitance of a parallel plate capacitor with a dielectric slab inserted...',
    answer: 'The capacitance with dielectric: C = κε₀A/d where κ is the dielectric constant.',
  },
  {
    year: '2022',
    subject: 'Chemistry',
    text: 'Explain the mechanism of SN2 reaction with an example.',
    answer: 'SN2 reactions proceed through a backside attack, with inversion of configuration...',
  },
  {
    year: '2023',
    subject: 'Biology',
    text: 'Describe the process of DNA replication and the enzymes involved.',
    answer: 'DNA replication is semi-conservative. Key enzymes: Helicase, Primase, DNA Pol III...',
  },
];

export function PYQAnalysis() {
  const [activeSubject, setActiveSubject] = useState('All');
  const [search, setSearch] = useState('');
  const [expandedPYQ, setExpandedPYQ] = useState<number | null>(null);

  const dotColor = (conf: string) => {
    if (conf === 'high') return '#2D6A4F';
    if (conf === 'medium') return '#52796F';
    return '#E2E2E2';
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
          Exam Intelligence
        </h1>
        <p className="text-[15px] text-[#6B6B6B] mt-1">
          Based on your uploaded previous year papers.
        </p>
      </div>

      {/* Subject Tabs */}
      <div className="flex gap-2 flex-wrap">
        {['All', 'Physics', 'Chemistry', 'Biology'].map(subj => (
          <button
            key={subj}
            onClick={() => setActiveSubject(subj)}
            className={cn(
              'px-5 py-2 rounded-full text-[13px] font-medium transition-all',
              activeSubject === subj
                ? 'bg-[#2D6A4F] text-white'
                : 'bg-white border border-[#E2E2E2] text-[#6B6B6B] hover:border-[#2D6A4F] hover:text-[#2D6A4F]'
            )}
          >
            {subj}
          </button>
        ))}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {[
          { label: 'Questions Analyzed', value: '1,240', icon: FileText },
          { label: 'High-Frequency Topics', value: '14', icon: TrendingUp },
          { label: 'Predicted Important Areas', value: '5', icon: Brain },
        ].map((s, i) => (
          <Card key={i} className={cn('p-5 flex items-center justify-between', i === 2 && 'border-[#2D6A4F] bg-[#D8E8E0]/10')}>
            <div>
              <span className={cn('text-[13px] font-medium block mb-1', i === 2 ? 'text-[#2D6A4F]' : 'text-[#6B6B6B]')}>
                {s.label}
              </span>
              <span className="font-serif text-[28px] font-medium text-[#1A1A1A] leading-none">
                {s.value}
              </span>
            </div>
            <div className={cn('w-10 h-10 rounded-full flex items-center justify-center', i === 2 ? 'bg-[#D8E8E0]' : 'bg-[#F0F0EE]')}>
              <s.icon size={18} className="text-[#2D6A4F]" />
            </div>
          </Card>
        ))}
      </div>

      {/* Two columns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Frequency List */}
        <Card className="p-6">
          <h3 className="font-serif text-[22px] font-medium text-[#1A1A1A] mb-5">
            Topic Frequency
          </h3>
          <div className="space-y-4">
            {topics.map((topic, i) => (
              <div key={i} className="flex items-center gap-4">
                <span className="w-36 text-[13px] font-medium text-[#1A1A1A] truncate flex-shrink-0">
                  {topic.name}
                </span>
                <div className="flex-1 h-[6px] bg-[#F0F0EE] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${(topic.frequency / 15) * 100}%`,
                      backgroundColor:
                        topic.importance === 'High'
                          ? '#2D6A4F'
                          : topic.importance === 'Medium'
                          ? '#52796F'
                          : '#E2E2E2',
                    }}
                  />
                </div>
                <span className="font-mono text-[11px] text-[#6B6B6B] w-24 text-right tracking-[0.03em] flex-shrink-0">
                  Asked {topic.frequency}×
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* AI Predictions */}
        <Card
          className="p-6 relative overflow-hidden"
          style={{ boxShadow: '0 0 0 2px #2D6A4F, 0 1px 4px rgba(0,0,0,0.06)' }}
        >
          <div className="absolute top-4 right-4 opacity-[0.06]">
            <Brain size={120} className="text-[#2D6A4F]" />
          </div>
          <div className="flex items-center gap-2 mb-5">
            <span className="bg-[#2D6A4F] text-white text-[11px] font-semibold uppercase tracking-wider px-3 py-1 rounded-full">
              AI Insight
            </span>
            <span className="font-mono text-[11px] text-[#6B6B6B] uppercase tracking-[0.03em]">
              High Probability Topics
            </span>
          </div>

          <div className="space-y-5 relative z-10">
            {predictions.map((pred, i) => (
              <div key={i} className="flex gap-3">
                <span className="font-serif text-[22px] text-[#2D6A4F]/20 select-none leading-none mt-0.5">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <div>
                  <div className="flex items-center gap-2 mb-0.5">
                    <h4 className="font-serif text-[17px] font-medium text-[#1A1A1A]">
                      {pred.topic}
                    </h4>
                    {/* Confidence dots */}
                    <div className="flex gap-0.5">
                      {[0, 1, 2].map(d => (
                        <div
                          key={d}
                          className="w-1.5 h-1.5 rounded-full"
                          style={{
                            backgroundColor:
                              pred.confidence === 'high'
                                ? '#2D6A4F'
                                : pred.confidence === 'medium' && d < 2
                                ? '#52796F'
                                : d === 0 && pred.confidence === 'low'
                                ? '#52796F'
                                : '#E2E2E2',
                          }}
                        />
                      ))}
                    </div>
                  </div>
                  <p className="text-[13px] text-[#6B6B6B]">{pred.reason}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* PYQ Browser */}
      <div>
        <h3 className="font-serif text-[22px] font-medium text-[#1A1A1A] mb-4">
          Browse Questions
        </h3>
        <Card className="p-0 overflow-hidden">
          <div className="p-4 bg-[#F4F4F2] border-b border-[#E2E2E2] flex gap-3">
            <div className="relative flex-1">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#6B6B6B]" />
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search within questions..."
                className="w-full pl-9 pr-4 h-10 bg-white border border-[#E2E2E2] rounded-lg text-[14px] text-[#1A1A1A] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
              />
            </div>
          </div>
          <div className="divide-y divide-[#E2E2E2]">
            {pyqs.map((q, i) => (
              <div key={i} className="hover:bg-[#F0F0EE] transition-colors">
                <div
                  className="p-5 cursor-pointer"
                  onClick={() => setExpandedPYQ(expandedPYQ === i ? null : i)}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-mono text-[11px] font-medium bg-[#F0F0EE] text-[#6B6B6B] px-2 py-0.5 rounded-full tracking-[0.03em]">
                      {q.year}
                    </span>
                    <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2 py-0.5 rounded-full">
                      {q.subject}
                    </span>
                  </div>
                  <div className="flex items-start justify-between gap-4">
                    <p className="text-[14px] text-[#1A1A1A] font-medium leading-relaxed">
                      {q.text}
                    </p>
                    <div className="text-[#6B6B6B] flex-shrink-0 mt-0.5">
                      {expandedPYQ === i ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </div>
                  </div>
                </div>
                {expandedPYQ === i && (
                  <div className="px-5 pb-5 pt-0">
                    <div className="pl-4 border-l-[3px] border-l-[#2D6A4F] bg-[#D8E8E0]/20 p-3 rounded-r-lg">
                      <p className="font-mono text-[11px] text-[#2D6A4F] uppercase tracking-[0.03em] mb-1">
                        Answer
                      </p>
                      <p className="text-[14px] text-[#1A1A1A] leading-relaxed">{q.answer}</p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
