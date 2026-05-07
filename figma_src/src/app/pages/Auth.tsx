import React, { useState } from 'react';
import { useNavigate } from 'react-router';
import { motion, AnimatePresence } from 'motion/react';
import { Eye, EyeOff, Check } from 'lucide-react';
import { cn } from '../components/ui-kit';

export function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleAuth = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      navigate('/app');
    }, 900);
  };

  return (
    <div className="flex h-screen w-full overflow-hidden font-sans">
      {/* ── Left Panel ── */}
      <div className="w-[45%] bg-[#2D6A4F] hidden md:flex flex-col justify-between p-16 relative overflow-hidden">
        {/* Decorative orbs */}
        <div className="absolute -top-24 -right-24 w-[400px] h-[400px] rounded-full bg-white opacity-[0.04] blur-3xl pointer-events-none" />
        <div className="absolute -bottom-16 -left-16 w-[300px] h-[300px] rounded-full bg-white opacity-[0.04] blur-3xl pointer-events-none" />

        <div />

        <div className="max-w-[420px] z-10">
          <h1 className="font-serif text-[48px] font-medium text-white leading-tight mb-10">
            "Your notes.<br />Your pace.<br />Your way."
          </h1>

          <ul className="space-y-5">
            {[
              'AI-generated notes from any content',
              'Flashcards and quizzes in seconds',
              'Tracks what you need to review',
            ].map((item, i) => (
              <li key={i} className="flex items-start gap-4">
                <div className="w-5 h-5 rounded-full border border-white/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Check size={11} className="text-white/70" />
                </div>
                <span className="text-white/80 text-[16px]">{item}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Wordmark */}
        <div className="flex items-center gap-1 z-10">
          <span className="font-serif text-[20px] font-medium text-white">Nudge</span>
          <span className="w-[7px] h-[7px] rounded-full bg-[#D8E8E0] mb-[1px]" />
        </div>
      </div>

      {/* ── Right Panel ── */}
      <div className="flex-1 bg-white flex flex-col items-center justify-center p-8 md:p-16">
        <AnimatePresence mode="wait">
          <motion.div
            key={isLogin ? 'login' : 'signup'}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            transition={{ duration: 0.3 }}
            className="w-full max-w-[400px]"
          >
            {/* Heading */}
            <div className="mb-8 text-center">
              <h2 className="font-serif text-[32px] font-medium text-[#1A1A1A] mb-2 leading-tight">
                {isLogin ? 'Welcome back' : 'Create your account'}
              </h2>
              <p className="text-[15px] text-[#6B6B6B]">
                {isLogin
                  ? "Let's pick up where you left off."
                  : 'Start your learning journey today.'}
              </p>
            </div>

            {/* Form */}
            <form onSubmit={handleAuth} className="space-y-4">
              {!isLogin && (
                <input
                  type="text"
                  placeholder="Full Name"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  required
                  className="w-full h-11 px-4 bg-white border-[1.5px] border-[#E2E2E2] rounded-lg text-[15px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
                />
              )}

              <input
                type="email"
                placeholder="Email address"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                className="w-full h-11 px-4 bg-white border-[1.5px] border-[#E2E2E2] rounded-lg text-[15px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
              />

              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  className="w-full h-11 px-4 pr-11 bg-white border-[1.5px] border-[#E2E2E2] rounded-lg text-[15px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[#6B6B6B] hover:text-[#2D6A4F] transition-colors"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>

              {!isLogin && (
                <input
                  type="password"
                  placeholder="Confirm Password"
                  required
                  className="w-full h-11 px-4 bg-white border-[1.5px] border-[#E2E2E2] rounded-lg text-[15px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
                />
              )}

              {isLogin && (
                <div className="flex justify-end">
                  <button
                    type="button"
                    className="text-[13px] text-[#2D6A4F] hover:underline font-medium"
                  >
                    Forgot password?
                  </button>
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className={cn(
                  'w-full h-12 bg-[#2D6A4F] text-white text-[14px] font-semibold rounded-lg transition-all',
                  loading
                    ? 'opacity-80 cursor-not-allowed'
                    : 'hover:bg-[#245C43]',
                  !loading && 'hover:shadow-[0_4px_12px_rgba(45,106,79,0.25)]'
                )}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="white" strokeWidth="3" strokeOpacity="0.3" />
                      <path d="M12 2a10 10 0 0 1 10 10" stroke="white" strokeWidth="3" strokeLinecap="round" />
                    </svg>
                    Signing in...
                  </span>
                ) : isLogin ? (
                  'Sign in →'
                ) : (
                  'Create account →'
                )}
              </button>
            </form>

            {/* Divider */}
            <div className="my-6 flex items-center gap-4">
              <div className="h-[1px] flex-1 bg-[#E2E2E2]" />
              <span className="text-[12px] text-[#6B6B6B] font-medium uppercase tracking-wider">or</span>
              <div className="h-[1px] flex-1 bg-[#E2E2E2]" />
            </div>

            {/* Google */}
            <button
              type="button"
              className="w-full h-12 flex items-center justify-center gap-3 bg-white border-[1.5px] border-[#E2E2E2] rounded-lg text-[14px] font-semibold text-[#1A1A1A] hover:bg-[#F0F0EE] transition-colors"
            >
              <svg viewBox="0 0 24 24" className="w-5 h-5" aria-hidden="true">
                <path
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  fill="#4285F4"
                />
                <path
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  fill="#34A853"
                />
                <path
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  fill="#FBBC05"
                />
                <path
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  fill="#EA4335"
                />
              </svg>
              Continue with Google
            </button>

            {/* Toggle */}
            <div className="mt-8 text-center">
              <span className="text-[14px] text-[#6B6B6B]">
                {isLogin ? 'New here? ' : 'Already have an account? '}
              </span>
              <button
                type="button"
                onClick={() => setIsLogin(v => !v)}
                className="text-[14px] text-[#2D6A4F] font-semibold hover:underline"
              >
                {isLogin ? 'Create a free account →' : 'Sign in →'}
              </button>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}