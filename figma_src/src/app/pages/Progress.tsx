import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Card, Button } from '../components/ui-kit';
import { GraduationCap, TrendingUp, Flame, BookOpen, ArrowRight } from 'lucide-react';
import { Link } from 'react-router';

const scoreData = [
  { name: 'Mon', score: 65 },
  { name: 'Tue', score: 70 },
  { name: 'Wed', score: 68 },
  { name: 'Thu', score: 75 },
  { name: 'Fri', score: 82 },
  { name: 'Sat', score: 85 },
  { name: 'Sun', score: 80 },
];

const topicData = [
  { name: 'Biology', score: 85, fill: '#2D6A4F' },
  { name: 'Math', score: 82, fill: '#2D6A4F' },
  { name: 'Chemistry', score: 65, fill: '#52796F' },
  { name: 'History', score: 70, fill: '#52796F' },
  { name: 'Physics', score: 45, fill: '#E2E2E2' },
];

const quizHistory = [
  { title: 'Photosynthesis Quiz', date: 'Mar 4', score: 80, total: 10 },
  { title: 'Organic Chemistry', date: 'Mar 3', score: 65, total: 10 },
  { title: 'Calculus Derivatives', date: 'Mar 2', score: 90, total: 10 },
  { title: 'World War II', date: 'Mar 1', score: 55, total: 10 },
];

function ScoreBadge({ pct }: { pct: number }) {
  if (pct >= 80)
    return (
      <span className="font-mono text-[11px] font-medium bg-[#D8E8E0] text-[#2D6A4F] px-2 py-0.5 rounded-full">
        {pct}%
      </span>
    );
  if (pct >= 60)
    return (
      <span className="font-mono text-[11px] font-medium bg-[#F0F0EE] text-[#6B6B6B] px-2 py-0.5 rounded-full">
        {pct}%
      </span>
    );
  return (
    <span className="font-mono text-[11px] font-medium bg-red-50 text-[#C0392B] px-2 py-0.5 rounded-full">
      {pct}%
    </span>
  );
}

export function Progress() {
  const stats = [
    { label: 'Quizzes Taken', value: '24', icon: GraduationCap },
    { label: 'Average Score', value: '78%', icon: TrendingUp },
    { label: 'Cards Reviewed', value: '142', icon: BookOpen },
    { label: 'Study Streak', value: '12 days', icon: Flame },
  ];

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
          Your Progress
        </h1>
        <p className="text-[15px] text-[#6B6B6B] mt-1">
          Track your performance and identify areas for improvement.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        {stats.map((stat, i) => (
          <Card key={i} className="p-5 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="text-[13px] text-[#6B6B6B]">{stat.label}</span>
              <div className="w-8 h-8 rounded-lg bg-[#F4F4F2] flex items-center justify-center">
                <stat.icon size={16} className="text-[#2D6A4F]" />
              </div>
            </div>
            <span className="font-serif text-[28px] font-medium text-[#1A1A1A] leading-none">
              {stat.value}
            </span>
          </Card>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Line Chart */}
        <Card className="p-6">
          <h3 className="font-serif text-[22px] font-medium text-[#1A1A1A] mb-6">
            Score Trend
          </h3>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={scoreData}>
                <CartesianGrid strokeDasharray="0" horizontal stroke="#E2E2E2" vertical={false} />
                <XAxis
                  dataKey="name"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#6B6B6B', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  dy={8}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#6B6B6B', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  domain={[50, 100]}
                />
                <Tooltip
                  contentStyle={{
                    background: '#FFFFFF',
                    border: '1px solid #E2E2E2',
                    borderRadius: 8,
                    boxShadow: '0 4px 16px rgba(0,0,0,0.06)',
                    fontFamily: 'Plus Jakarta Sans',
                    fontSize: 13,
                  }}
                  labelStyle={{ color: '#6B6B6B', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                  itemStyle={{ color: '#2D6A4F', fontWeight: 600 }}
                />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#2D6A4F"
                  strokeWidth={2.5}
                  dot={{ fill: '#2D6A4F', r: 4, strokeWidth: 0 }}
                  activeDot={{ r: 6, strokeWidth: 0, fill: '#2D6A4F' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Topic Bars */}
        <Card className="p-6">
          <h3 className="font-serif text-[22px] font-medium text-[#1A1A1A] mb-6">
            Performance by Topic
          </h3>
          <div className="space-y-4">
            {topicData.map(topic => (
              <div key={topic.name} className="flex items-center gap-4">
                <span className="w-20 text-[13px] font-medium text-[#1A1A1A] flex-shrink-0">
                  {topic.name}
                </span>
                <div className="flex-1 h-2 bg-[#F0F0EE] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${topic.score}%`, backgroundColor: topic.fill }}
                  />
                </div>
                <span className="font-mono text-[11px] text-[#6B6B6B] w-10 text-right tracking-[0.03em]">
                  {topic.score}%
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Focus Card */}
      <Card className="border-l-[3px] border-l-[#2D6A4F] p-6">
        <h3 className="font-serif text-[22px] font-medium text-[#1A1A1A] mb-4">
          Where to focus next 🌱
        </h3>
        <div className="flex flex-wrap gap-4">
          {[
            { topic: 'Physics: Optics', accuracy: '45%' },
            { topic: 'Chemistry: Alkanes', accuracy: '65%' },
            { topic: 'Biology: Respiration', accuracy: '70%' },
          ].map(item => (
            <div
              key={item.topic}
              className="flex items-center gap-3 bg-white border border-[#E2E2E2] rounded-full px-4 py-2"
            >
              <span className="text-[14px] font-medium text-[#1A1A1A]">{item.topic}</span>
              <span className="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2 py-0.5 rounded-full">
                {item.accuracy}
              </span>
              <button className="text-[13px] text-[#2D6A4F] font-semibold hover:underline">
                Review →
              </button>
            </div>
          ))}
        </div>
      </Card>

      {/* Quiz History Table */}
      <Card className="p-0 overflow-hidden">
        <div className="px-6 py-4 border-b border-[#E2E2E2]">
          <h3 className="font-serif text-[22px] font-medium text-[#1A1A1A]">
            Recent Quiz History
          </h3>
        </div>
        <div className="divide-y divide-[#E2E2E2]">
          {quizHistory.map((quiz, i) => (
            <div
              key={i}
              className={`flex items-center justify-between px-6 py-4 ${i % 2 === 0 ? 'bg-white' : 'bg-[#F4F4F2]'}`}
            >
              <div>
                <p className="text-[15px] font-medium text-[#1A1A1A]">{quiz.title}</p>
                <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                  {quiz.date}
                </span>
              </div>
              <div className="flex items-center gap-4">
                <ScoreBadge pct={quiz.score} />
                <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                  {Math.round((quiz.score / 100) * quiz.total)}/{quiz.total}
                </span>
                <Link to="/quizzes/results" className="text-[13px] text-[#2D6A4F] hover:underline">
                  View →
                </Link>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
