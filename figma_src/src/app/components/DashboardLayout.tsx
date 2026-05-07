import React, { useState, useRef, useEffect } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router';
import {
  House,
  Files,
  Cards,
  GraduationCap,
  TrendUp,
  FileText,
  BookmarkSimple,
  ClockCounterClockwise,
  Gear,
  Bell,
  MagnifyingGlass,
  Export,
  X,
  Check,
  Lightning,
} from '@phosphor-icons/react';
import { cn } from './ui-kit';
import { motion, AnimatePresence } from 'motion/react';
import { Link } from 'react-router';

const notifications = [
  {
    id: 1,
    type: 'flashcard',
    title: '7 flashcards due for review',
    sub: 'Cellular Respiration · Biology',
    time: '2m ago',
    read: false,
    action: 'Review now',
  },
  {
    id: 2,
    type: 'quiz',
    title: 'New quiz ready: Organic Chemistry',
    sub: 'Based on your latest notes',
    time: '1h ago',
    read: false,
    action: 'Start quiz',
  },
  {
    id: 3,
    type: 'insight',
    title: 'Study insight ready',
    sub: "You've reviewed 24 cards this week — great pace!",
    time: '3h ago',
    read: false,
    action: 'View insight',
  },
  {
    id: 4,
    type: 'flashcard',
    title: 'Periodic Table review due',
    sub: 'Chemistry · 24 cards',
    time: 'Yesterday',
    read: true,
    action: 'Review',
  },
];

function NotificationPanel({ onClose }: { onClose: () => void }) {
  const [items, setItems] = useState(notifications);

  const markAllRead = () => setItems(prev => prev.map(n => ({ ...n, read: true })));

  const typeIcon = (type: string) => {
    if (type === 'flashcard') return <Cards size={13} className="text-[#2D6A4F]" />;
    if (type === 'quiz') return <GraduationCap size={13} className="text-[#52796F]" />;
    return <Lightning size={13} className="text-white" />;
  };

  const typeBg = (type: string) => {
    if (type === 'flashcard') return 'bg-[#D8E8E0]';
    if (type === 'quiz') return 'bg-[#F0F0EE]';
    return 'bg-[#2D6A4F]';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -8, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.97 }}
      transition={{ duration: 0.18, ease: 'easeOut' }}
      className="absolute top-[calc(100%+8px)] right-0 w-[380px] bg-white z-50 overflow-hidden"
      style={{
        boxShadow: '0 8px 32px rgba(0,0,0,0.10)',
        borderRadius: '0 0 16px 16px',
        border: '1px solid #E2E2E2',
        borderTop: 'none',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 pt-5 pb-3 border-b border-[#E2E2E2]">
        <span className="font-serif text-[17px] font-medium text-[#1A1A1A]">Nudges</span>
        <div className="flex items-center gap-3">
          <button
            onClick={markAllRead}
            className="text-[#2D6A4F] text-[12px] font-semibold hover:underline"
          >
            Mark all as read
          </button>
          <button onClick={onClose} className="text-[#6B6B6B] hover:text-[#1A1A1A] transition-colors">
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Items */}
      <div className="divide-y divide-[#E2E2E2] max-h-[420px] overflow-y-auto">
        {items.map((n, i) => (
          <div
            key={n.id}
            className={cn('px-5 py-4 flex gap-3', n.read ? 'opacity-55' : '')}
          >
            {/* Dot + line */}
            <div className="flex flex-col items-center pt-[3px] flex-shrink-0 gap-1">
              <div
                className={cn(
                  'w-2 h-2 rounded-full flex-shrink-0',
                  !n.read ? 'bg-[#2D6A4F]' : 'bg-transparent'
                )}
              />
              {i < items.length - 1 && (
                <div className="w-[1px] flex-1 bg-[#E2E2E2]" style={{ minHeight: 24 }} />
              )}
            </div>

            {/* Icon badge */}
            <div
              className={cn(
                'w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0',
                typeBg(n.type)
              )}
            >
              {typeIcon(n.type)}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-2">
                <p className="text-[14px] font-medium text-[#1A1A1A] leading-tight">
                  {n.title}
                </p>
                <span className="font-mono text-[10px] text-[#6B6B6B] whitespace-nowrap tracking-[0.03em]">
                  {n.time}
                </span>
              </div>
              <p className="text-[12px] text-[#6B6B6B] mt-0.5 mb-2">{n.sub}</p>
              <button className="text-[11px] font-semibold text-[#2D6A4F] border border-[#2D6A4F] rounded-full px-3 py-0.5 hover:bg-[#D8E8E0] transition-colors">
                {n.action}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-[#E2E2E2] text-center">
        <Link
          to="/app/settings"
          onClick={onClose}
          className="text-[12px] text-[#6B6B6B] hover:text-[#2D6A4F] transition-colors"
        >
          Notification settings →
        </Link>
      </div>
    </motion.div>
  );
}

export function DashboardLayout() {
  const [notifOpen, setNotifOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const notifRef = useRef<HTMLDivElement>(null);
  const location = useLocation();

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setNotifOpen(false);
      }
    };
    if (notifOpen) document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [notifOpen]);

  // Breadcrumb from /app/* path
  const segments = location.pathname.replace('/app', '').split('/').filter(Boolean);
  const breadcrumb =
    segments.length === 0
      ? 'Dashboard'
      : segments.map(s => s.charAt(0).toUpperCase() + s.slice(1).replace(/-/g, ' ')).join(' / ');

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <div className="flex h-screen bg-[#F4F4F2] overflow-hidden font-sans">
      {/* ── Sidebar ── */}
      <aside className="w-[220px] flex-shrink-0 h-full border-r border-[#E2E2E2] bg-[#F4F4F2] flex flex-col justify-between">
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="h-14 flex items-center px-6 border-b border-[#E2E2E2]">
            <span className="font-serif text-[20px] font-medium text-[#1A1A1A]">Nudge</span>
            <span className="ml-[2px] w-[7px] h-[7px] rounded-full bg-[#2D6A4F] mb-[1px]" />
          </div>

          {/* Nav */}
          <nav className="flex-1 overflow-y-auto py-3">
            <ul className="space-y-0.5">
              <NavItem to="/app" icon={House} label="Home" exact />
              <NavItem to="/app/notes" icon={Files} label="My Notes" />
              <NavItem to="/app/flashcards" icon={Cards} label="Flashcards" />
              <NavItem to="/app/quizzes" icon={GraduationCap} label="Quizzes" />
              <NavItem to="/app/progress" icon={TrendUp} label="Progress" />
              <NavItem to="/app/pyq-analysis" icon={FileText} label="PYQ Analysis" />
              <NavItem to="/app/bookmarks" icon={BookmarkSimple} label="Bookmarks" />
              <NavItem to="/app/history" icon={ClockCounterClockwise} label="History" />
              <NavItem to="/app/export" icon={Export} label="Export" />
            </ul>
          </nav>

          {/* Bottom */}
          <div className="border-t border-[#E2E2E2] p-3">
            <NavItem to="/app/settings" icon={Gear} label="Settings" />
            <div className="mt-3 flex items-center gap-3 px-4 py-2">
              <div className="w-8 h-8 rounded-full bg-[#2D6A4F] flex items-center justify-center flex-shrink-0">
                <span className="font-mono text-[11px] text-white tracking-wider">AR</span>
              </div>
              <div className="flex flex-col min-w-0">
                <span className="text-[14px] font-medium text-[#1A1A1A] truncate">Arjun R.</span>
                <Link to="/login" className="text-[11px] text-[#6B6B6B] hover:text-[#2D6A4F] transition-colors">
                  Sign out
                </Link>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Top Bar */}
        <header className="h-14 flex items-center justify-between px-6 bg-[#F4F4F2] border-b border-[#E2E2E2] flex-shrink-0 relative">
          <span className="font-mono text-[11px] text-[#6B6B6B] uppercase tracking-[0.03em]">
            {breadcrumb}
          </span>

          <div className="flex items-center gap-3">
            {/* Search */}
            {searchOpen ? (
              <input
                autoFocus
                placeholder="Search..."
                className="h-8 px-3 bg-white border border-[#E2E2E2] rounded-lg text-[13px] focus:outline-none focus:border-[#2D6A4F] w-48 transition-all"
                onBlur={() => setSearchOpen(false)}
              />
            ) : (
              <button
                onClick={() => setSearchOpen(true)}
                className="text-[#6B6B6B] hover:text-[#2D6A4F] transition-colors p-1"
              >
                <MagnifyingGlass size={20} />
              </button>
            )}

            {/* Bell */}
            <div className="relative" ref={notifRef}>
              <button
                onClick={() => setNotifOpen(prev => !prev)}
                className="relative text-[#6B6B6B] hover:text-[#2D6A4F] transition-colors p-1"
              >
                <Bell size={20} />
                {unreadCount > 0 && (
                  <span className="absolute top-0.5 right-0.5 w-2 h-2 bg-[#C0392B] rounded-full border-2 border-[#F4F4F2]" />
                )}
              </button>
              <AnimatePresence>
                {notifOpen && <NotificationPanel onClose={() => setNotifOpen(false)} />}
              </AnimatePresence>
            </div>

            {/* Avatar */}
            <div className="w-8 h-8 rounded-full bg-[#2D6A4F] flex items-center justify-center cursor-pointer">
              <span className="font-mono text-[11px] text-white tracking-wider">AR</span>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto">
          <div className="px-12 py-10 max-w-[1100px] mx-auto pb-20">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}

function NavItem({
  to,
  icon: Icon,
  label,
  exact,
}: {
  to: string;
  icon: React.ElementType;
  label: string;
  exact?: boolean;
}) {
  return (
    <li>
      <NavLink
        to={to}
        end={exact}
        className={({ isActive }) =>
          cn(
            'flex items-center gap-3 h-10 px-4 mx-2 rounded-lg transition-all duration-150 group relative text-[14px]',
            isActive
              ? 'bg-[#D8E8E0]/50 text-[#2D6A4F]'
              : 'text-[#6B6B6B] hover:bg-[#F0F0EE] hover:text-[#1A1A1A]'
          )
        }
      >
        {({ isActive }) => (
          <>
            {isActive && (
              <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-6 bg-[#2D6A4F] rounded-r-full -ml-2" />
            )}
            <Icon
              size={20}
              weight="regular"
              className={cn(
                'transition-colors flex-shrink-0',
                isActive ? 'text-[#2D6A4F]' : 'text-[#6B6B6B] group-hover:text-[#1A1A1A]'
              )}
            />
            <span className="font-sans font-medium">{label}</span>
          </>
        )}
      </NavLink>
    </li>
  );
}
