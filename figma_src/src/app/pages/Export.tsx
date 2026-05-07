import React, { useState } from 'react';
import { Card } from '../components/ui-kit';
import { FileText, Check, ChevronDown, ChevronUp, Download } from 'lucide-react';
import { cn } from '../components/ui-kit';
import { motion, AnimatePresence } from 'motion/react';

const contentItems = [
  { id: 1, title: 'Photosynthesis & Plant Biology', type: 'Note' },
  { id: 2, title: 'Organic Chemistry: Alkanes', type: 'Note' },
  { id: 3, title: 'Biology Quiz: Photosynthesis', type: 'Quiz' },
  { id: 4, title: "Newton's Laws of Motion", type: 'Note' },
  { id: 5, title: 'Photosynthesis Flashcards', type: 'Flashcard' },
  { id: 6, title: 'World War II: Key Events', type: 'Note' },
];

const formats = [
  {
    id: 'pdf',
    label: 'PDF Document',
    desc: 'Formatted, print-ready PDF with your custom header.',
    icon: '📄',
  },
  {
    id: 'txt',
    label: 'Plain Text',
    desc: 'Clean .txt file for importing into any app.',
    icon: '📝',
  },
  {
    id: 'anki',
    label: 'Anki Deck (.apkg)',
    desc: 'Export flashcards directly into Anki.',
    icon: '🃏',
  },
];

const pdfOptions = [
  { id: 'cover', label: 'Include cover page', checked: true },
  { id: 'toc', label: 'Table of contents', checked: true },
  { id: 'concepts', label: 'Key concepts appendix', checked: false },
  { id: 'pages', label: 'Page numbers', checked: true },
];

const fontSizes = ['Small', 'Medium', 'Large'];

export function ExportPage() {
  const [selected, setSelected] = useState<Set<number>>(new Set([1, 2, 3]));
  const [format, setFormat] = useState('pdf');
  const [pdfChecked, setPdfChecked] = useState<Record<string, boolean>>(
    Object.fromEntries(pdfOptions.map(o => [o.id, o.checked]))
  );
  const [fontSize, setFontSize] = useState('Medium');
  const [headerText, setHeaderText] = useState('');
  const [pdfExpanded, setPdfExpanded] = useState(true);
  const [success, setSuccess] = useState(false);

  const toggleSelect = (id: number) => {
    setSelected(prev => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  };

  const selectAll = () => setSelected(new Set(contentItems.map(i => i.id)));
  const clearAll = () => setSelected(new Set());

  const handleDownload = () => {
    setSuccess(true);
    setTimeout(() => setSuccess(false), 4000);
  };

  return (
    <div className="max-w-[800px] mx-auto space-y-6 animate-in fade-in duration-500">
      {/* Success Banner */}
      <AnimatePresence>
        {success && (
          <motion.div
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-[#52796F] text-white px-6 py-3 rounded-xl shadow-lg"
          >
            <Check size={18} />
            <span className="text-[14px] font-semibold">✓ Your export is downloading!</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <div>
        <h1 className="font-serif text-[32px] font-medium text-[#1A1A1A] leading-tight">
          Export Your Content
        </h1>
        <p className="text-[15px] text-[#6B6B6B] mt-1">
          Choose what to export and in what format.
        </p>
      </div>

      {/* Content Selector */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A]">
            What do you want to export?
          </h3>
          <div className="flex items-center gap-3">
            <button
              onClick={selectAll}
              className="text-[#2D6A4F] text-[13px] font-medium hover:underline"
            >
              Select all
            </button>
            <span className="text-[#E2E2E2]">·</span>
            <span className="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">
              {selected.size} selected
            </span>
          </div>
        </div>

        <div className="space-y-1">
          {contentItems.map(item => (
            <label
              key={item.id}
              onClick={() => toggleSelect(item.id)}
              className="flex items-center gap-4 px-3 py-3 rounded-lg cursor-pointer hover:bg-[#F0F0EE] transition-colors"
            >
              <div
                className={cn(
                  'w-4 h-4 rounded border-[1.5px] flex items-center justify-center transition-colors flex-shrink-0',
                  selected.has(item.id)
                    ? 'bg-[#2D6A4F] border-[#2D6A4F]'
                    : 'bg-white border-[#E2E2E2] hover:border-[#2D6A4F]'
                )}
              >
                {selected.has(item.id) && <Check size={10} strokeWidth={3} className="text-white" />}
              </div>
              <span className="flex-1 text-[14px] font-medium text-[#1A1A1A]">{item.title}</span>
              <span className="bg-[#F0F0EE] text-[#6B6B6B] text-[11px] font-medium px-2.5 py-0.5 rounded-full">
                {item.type}
              </span>
            </label>
          ))}
        </div>
      </Card>

      {/* Format Selector */}
      <div>
        <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-4">Select Format</h3>
        <div className="grid grid-cols-3 gap-4">
          {formats.map(f => (
            <button
              key={f.id}
              onClick={() => setFormat(f.id)}
              className={cn(
                'p-5 rounded-xl border text-left transition-all',
                format === f.id
                  ? 'border-[#2D6A4F] bg-[#D8E8E0]/25'
                  : 'border-[#E2E2E2] bg-white hover:border-[#2D6A4F]/30'
              )}
            >
              <span className="text-2xl mb-3 block">{f.icon}</span>
              <p className="font-serif text-[17px] font-medium text-[#1A1A1A] mb-1">{f.label}</p>
              <p className="text-[12px] text-[#6B6B6B] leading-relaxed">{f.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* PDF Options */}
      {format === 'pdf' && (
        <Card className="p-0 overflow-hidden">
          <button
            onClick={() => setPdfExpanded(e => !e)}
            className="w-full flex items-center justify-between px-6 py-4 hover:bg-[#F0F0EE] transition-colors"
          >
            <h3 className="font-serif text-[17px] font-medium text-[#1A1A1A]">PDF Options</h3>
            {pdfExpanded ? (
              <ChevronUp size={18} className="text-[#6B6B6B]" />
            ) : (
              <ChevronDown size={18} className="text-[#6B6B6B]" />
            )}
          </button>

          <AnimatePresence>
            {pdfExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="px-6 pb-6 space-y-5 border-t border-[#E2E2E2] pt-5">
                  {/* Checkboxes */}
                  <div className="grid grid-cols-2 gap-3">
                    {pdfOptions.map(opt => (
                      <label
                        key={opt.id}
                        onClick={() => setPdfChecked(prev => ({ ...prev, [opt.id]: !prev[opt.id] }))}
                        className="flex items-center gap-3 cursor-pointer"
                      >
                        <div
                          className={cn(
                            'w-4 h-4 rounded border-[1.5px] flex items-center justify-center transition-colors flex-shrink-0',
                            pdfChecked[opt.id]
                              ? 'bg-[#2D6A4F] border-[#2D6A4F]'
                              : 'bg-white border-[#E2E2E2]'
                          )}
                        >
                          {pdfChecked[opt.id] && (
                            <Check size={10} strokeWidth={3} className="text-white" />
                          )}
                        </div>
                        <span className="text-[14px] text-[#1A1A1A]">{opt.label}</span>
                      </label>
                    ))}
                  </div>

                  {/* Font Size Toggle */}
                  <div>
                    <p className="text-[13px] font-medium text-[#6B6B6B] mb-2">Font Size</p>
                    <div className="flex border border-[#E2E2E2] rounded-lg overflow-hidden w-fit">
                      {fontSizes.map(s => (
                        <button
                          key={s}
                          onClick={() => setFontSize(s)}
                          className={cn(
                            'px-5 py-2 text-[13px] font-medium transition-colors',
                            fontSize === s
                              ? 'bg-[#2D6A4F] text-white'
                              : 'bg-white text-[#6B6B6B] hover:bg-[#F0F0EE]'
                          )}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Custom Header */}
                  <div>
                    <p className="text-[13px] font-medium text-[#6B6B6B] mb-2">Custom Header</p>
                    <input
                      type="text"
                      value={headerText}
                      onChange={e => setHeaderText(e.target.value)}
                      placeholder="e.g. Arjun's Study Notes · Biology"
                      className="w-full h-10 px-4 bg-white border border-[#E2E2E2] rounded-lg text-[14px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>
      )}

      {/* Preview + Download */}
      <div className="grid grid-cols-5 gap-6">
        {/* Preview thumbnail */}
        <div className="col-span-2">
          <p className="text-[13px] font-medium text-[#6B6B6B] mb-2">Preview</p>
          <div className="bg-[#F4F4F2] border border-[#E2E2E2] rounded-xl p-4 aspect-[3/4] flex flex-col">
            <div className="flex-1 bg-white rounded-lg p-3 shadow-sm flex flex-col gap-2">
              <div className="h-2 bg-[#2D6A4F] rounded-full w-3/4" />
              <div className="h-1.5 bg-[#E2E2E2] rounded-full w-1/2" />
              <div className="h-[1px] bg-[#E2E2E2] my-1" />
              <div className="space-y-1">
                {[0.9, 0.7, 0.8, 0.6, 0.75].map((w, i) => (
                  <div
                    key={i}
                    className="h-1 bg-[#E2E2E2] rounded-full"
                    style={{ width: `${w * 100}%` }}
                  />
                ))}
              </div>
            </div>
            <div className="mt-2 text-center">
              <span className="font-serif text-[11px] text-[#6B6B6B]">Nudge●</span>
            </div>
          </div>
          <p className="font-mono text-[11px] text-[#6B6B6B] mt-2 tracking-[0.03em]">
            {selected.size * 2} pages · Est. {(selected.size * 0.2).toFixed(1)} MB
          </p>
        </div>

        {/* Download */}
        <div className="col-span-3 flex flex-col justify-end gap-4">
          <div className="bg-[#F4F4F2] rounded-xl p-4">
            <p className="text-[13px] font-medium text-[#1A1A1A] mb-1">Export Summary</p>
            <div className="space-y-1">
              <div className="flex justify-between text-[13px]">
                <span className="text-[#6B6B6B]">Items selected</span>
                <span className="font-medium text-[#1A1A1A]">{selected.size}</span>
              </div>
              <div className="flex justify-between text-[13px]">
                <span className="text-[#6B6B6B]">Format</span>
                <span className="font-medium text-[#1A1A1A]">
                  {formats.find(f => f.id === format)?.label}
                </span>
              </div>
              {format === 'pdf' && (
                <div className="flex justify-between text-[13px]">
                  <span className="text-[#6B6B6B]">Font size</span>
                  <span className="font-medium text-[#1A1A1A]">{fontSize}</span>
                </div>
              )}
            </div>
          </div>

          <button
            onClick={handleDownload}
            disabled={selected.size === 0}
            className="w-full flex items-center justify-center gap-2 bg-[#2D6A4F] text-white text-[14px] font-semibold py-3.5 rounded-lg hover:bg-[#245C43] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ boxShadow: selected.size > 0 ? '0 4px 12px rgba(45,106,79,0.25)' : 'none' }}
          >
            <Download size={18} />
            Download Export →
          </button>
          <p className="text-[12px] text-[#6B6B6B] text-center">
            Nothing is uploaded to external servers.
          </p>
        </div>
      </div>
    </div>
  );
}
