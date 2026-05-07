import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router';
import { Button, Card } from '../components/ui-kit';
import { FileText, FileUp, Image, Mic, BookOpen, X, Check } from 'lucide-react';
import { cn } from '../components/ui-kit';
import { motion, AnimatePresence } from 'motion/react';

const tabs = [
  { id: 'text', label: 'Text', icon: FileText },
  { id: 'file', label: 'PDF / Word', icon: FileUp },
  { id: 'image', label: 'Image', icon: Image },
  { id: 'audio', label: 'Audio', icon: Mic },
  { id: 'pyq', label: 'PYQ', icon: BookOpen },
];

const processingOptions = [
  { id: 'summary', label: 'Generate Summary', defaultChecked: true },
  { id: 'notes', label: 'Create Structured Notes', defaultChecked: true },
  { id: 'flashcards', label: 'Generate Flashcards', defaultChecked: true },
  { id: 'quiz', label: 'Create Quiz', defaultChecked: false },
];

export function Upload() {
  const [activeTab, setActiveTab] = useState('text');
  const [text, setText] = useState('');
  const [subject, setSubject] = useState('');
  const [tags, setTags] = useState('');
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [checked, setChecked] = useState<Record<string, boolean>>({
    summary: true,
    notes: true,
    flashcards: true,
    quiz: false,
  });
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const toggleCheck = (id: string) =>
    setChecked(prev => ({ ...prev, [id]: !prev[id] }));

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setUploadedFile(file.name);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) setUploadedFile(file.name);
  };

  const startRecording = () => {
    setIsRecording(true);
    setRecordingTime(0);
    timerRef.current = setInterval(() => setRecordingTime(t => t + 1), 1000);
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
    setUploadedFile('voice-recording.webm');
  };

  const formatTime = (s: number) =>
    `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

  return (
    <div className="max-w-[800px] mx-auto animate-in fade-in duration-500">
      <div className="mb-8">
        <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
          Add New Study Material
        </h1>
        <p className="text-[15px] text-[#6B6B6B] mt-1">
          Upload a file, paste text, or record a voice note — we'll handle the rest.
        </p>
      </div>

      {/* Tab Pills */}
      <div className="flex gap-2 flex-wrap mb-6">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-full border text-[13px] font-medium transition-all',
              activeTab === tab.id
                ? 'bg-[#2D6A4F] text-white border-[#2D6A4F]'
                : 'bg-white text-[#1A1A1A] border-[#E2E2E2] hover:bg-[#F0F0EE]'
            )}
          >
            <tab.icon size={15} />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Input Area */}
      <Card className="mb-6 p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'text' && (
            <motion.div
              key="text"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col gap-4"
            >
              <div className="relative">
                <textarea
                  className="w-full min-h-[220px] p-4 text-[15px] text-[#1A1A1A] placeholder:text-[#6B6B6B] border border-[#E2E2E2] rounded-lg focus:outline-none focus:border-[#2D6A4F] resize-none bg-white transition-colors leading-relaxed"
                  placeholder="Paste your notes, textbook excerpt, or any study content here..."
                  value={text}
                  onChange={e => setText(e.target.value)}
                />
                <span className="absolute bottom-3 right-3 font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                  {text.length} chars
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <input
                  type="text"
                  placeholder="Subject / Topic"
                  value={subject}
                  onChange={e => setSubject(e.target.value)}
                  className="h-10 px-4 bg-white border border-[#E2E2E2] rounded-lg text-[14px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
                />
                <input
                  type="text"
                  placeholder="Add Tags (comma separated)"
                  value={tags}
                  onChange={e => setTags(e.target.value)}
                  className="h-10 px-4 bg-white border border-[#E2E2E2] rounded-lg text-[14px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
                />
              </div>
            </motion.div>
          )}

          {activeTab === 'file' && (
            <motion.div
              key="file"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {uploadedFile ? (
                <div className="flex items-center gap-4 p-4 bg-[#F4F4F2] rounded-lg border border-[#E2E2E2]">
                  <div className="w-10 h-10 rounded-lg bg-[#D8E8E0] flex items-center justify-center">
                    <FileUp size={18} className="text-[#2D6A4F]" />
                  </div>
                  <span className="flex-1 text-[14px] font-medium text-[#1A1A1A]">
                    {uploadedFile}
                  </span>
                  <button
                    onClick={() => setUploadedFile(null)}
                    className="text-[#6B6B6B] hover:text-[#C0392B] transition-colors"
                  >
                    <X size={18} />
                  </button>
                </div>
              ) : (
                <div
                  onDragOver={e => e.preventDefault()}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className="flex flex-col items-center justify-center border-2 border-dashed border-[#E2E2E2] rounded-xl bg-[#F4F4F2] p-12 cursor-pointer hover:bg-[#F0F0EE] hover:border-[#2D6A4F] transition-all min-h-[220px]"
                >
                  <div className="w-16 h-16 rounded-full bg-[#D8E8E0] flex items-center justify-center mb-4">
                    <FileUp size={28} className="text-[#2D6A4F]" />
                  </div>
                  <p className="text-[15px] font-medium text-[#1A1A1A] mb-1">
                    Drag and drop your file here
                  </p>
                  <p className="text-[14px] text-[#6B6B6B]">
                    or{' '}
                    <span className="text-[#2D6A4F] font-semibold hover:underline">
                      browse files
                    </span>
                  </p>
                  <p className="font-mono text-[11px] text-[#6B6B6B] mt-3 tracking-[0.03em]">
                    PDF, DOCX, TXT · Up to 10MB
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.doc,.docx,.txt"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'image' && (
            <motion.div
              key="image"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center border-2 border-dashed border-[#E2E2E2] rounded-xl bg-[#F4F4F2] p-12 cursor-pointer hover:bg-[#F0F0EE] hover:border-[#2D6A4F] transition-all min-h-[220px]"
            >
              <div className="w-16 h-16 rounded-full bg-[#D8E8E0] flex items-center justify-center mb-4">
                <Image size={28} className="text-[#2D6A4F]" />
              </div>
              <p className="text-[15px] font-medium text-[#1A1A1A] mb-1">
                Upload an image of your notes
              </p>
              <p className="text-[14px] text-[#6B6B6B]">
                or{' '}
                <span className="text-[#2D6A4F] font-semibold hover:underline">browse files</span>
              </p>
              <p className="font-mono text-[11px] text-[#6B6B6B] mt-3 tracking-[0.03em]">
                JPG, PNG, HEIC · Up to 10MB
              </p>
            </motion.div>
          )}

          {activeTab === 'audio' && (
            <motion.div
              key="audio"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-2 gap-6 min-h-[220px]"
            >
              {/* Upload */}
              <div className="flex flex-col items-center justify-center border-2 border-dashed border-[#E2E2E2] rounded-xl bg-[#F4F4F2] p-8 cursor-pointer hover:bg-[#F0F0EE] transition-colors">
                <FileUp size={28} className="text-[#2D6A4F] mb-3" />
                <p className="text-[14px] font-medium text-[#1A1A1A] mb-1 text-center">
                  Upload audio file
                </p>
                <p className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
                  MP3, WAV, M4A
                </p>
              </div>

              {/* Record */}
              <div className="flex flex-col items-center justify-center gap-4 p-8">
                <div className="relative">
                  {isRecording && (
                    <div className="absolute inset-0 rounded-full bg-[#2D6A4F]/20 animate-ping" />
                  )}
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={cn(
                      'w-20 h-20 rounded-full flex items-center justify-center transition-all',
                      isRecording
                        ? 'bg-[#C0392B] hover:bg-[#A93226]'
                        : 'bg-[#2D6A4F] hover:bg-[#245C43]'
                    )}
                  >
                    <Mic size={28} className="text-white" />
                  </button>
                </div>
                <div className="text-center">
                  {isRecording ? (
                    <>
                      <p className="font-mono text-[17px] text-[#2D6A4F] font-medium tracking-[0.03em]">
                        {formatTime(recordingTime)}
                      </p>
                      <p className="text-[13px] text-[#6B6B6B] mt-1">Recording... tap to stop</p>
                    </>
                  ) : (
                    <p className="text-[14px] text-[#6B6B6B]">Tap to Record</p>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'pyq' && (
            <motion.div
              key="pyq"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center border-2 border-dashed border-[#E2E2E2] rounded-xl bg-[#F4F4F2] p-12 cursor-pointer hover:bg-[#F0F0EE] hover:border-[#2D6A4F] transition-all min-h-[220px]"
            >
              <div className="w-16 h-16 rounded-full bg-[#D8E8E0] flex items-center justify-center mb-4">
                <BookOpen size={28} className="text-[#2D6A4F]" />
              </div>
              <p className="text-[15px] font-medium text-[#1A1A1A] mb-1">
                Upload Previous Year Papers
              </p>
              <p className="text-[14px] text-[#6B6B6B]">
                or{' '}
                <span className="text-[#2D6A4F] font-semibold hover:underline">browse files</span>
              </p>
              <p className="font-mono text-[11px] text-[#6B6B6B] mt-3 tracking-[0.03em]">
                PDF, DOCX · Up to 20MB
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Processing Options */}
      <Card className="mb-6 p-6">
        <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-4">
          What do you want Nudge to do?
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {processingOptions.map(opt => (
            <label
              key={opt.id}
              className="flex items-center gap-3 cursor-pointer group"
              onClick={() => toggleCheck(opt.id)}
            >
              <div
                className={cn(
                  'w-4 h-4 rounded flex items-center justify-center border-[1.5px] transition-colors flex-shrink-0',
                  checked[opt.id]
                    ? 'bg-[#2D6A4F] border-[#2D6A4F]'
                    : 'bg-white border-[#E2E2E2] group-hover:border-[#2D6A4F]'
                )}
              >
                {checked[opt.id] && <Check size={11} strokeWidth={3} className="text-white" />}
              </div>
              <span
                className={cn(
                  'text-[14px]',
                  checked[opt.id] ? 'text-[#1A1A1A] font-medium' : 'text-[#6B6B6B]'
                )}
              >
                {opt.label}
              </span>
            </label>
          ))}
        </div>
      </Card>

      {/* CTA */}
      <div>
        <Button
          size="lg"
          fullWidth
          className="h-14 text-[16px]"
          onClick={() => navigate('/processing')}
        >
          Process with AI →
        </Button>
        <p className="mt-3 text-[13px] text-[#6B6B6B] text-center">
          This usually takes 10–30 seconds
        </p>
      </div>
    </div>
  );
}