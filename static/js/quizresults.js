(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData || !pageData.result) return;

  const result = pageData.result;
  const scoreValue = document.getElementById("scoreValue");
  const percentValue = document.getElementById("percentValue");
  const scoreBar = document.getElementById("scoreBar");
  const feedbackText = document.getElementById("feedbackText");
  const weakTopicsWrapper = document.getElementById("weakTopicsWrapper");
  const wrongAnswersAccordionWrapper = document.getElementById("wrongAnswersAccordionWrapper");
  const reviewNotesBtn = document.getElementById("reviewNotesBtn");
  const retryQuizBtn = document.getElementById("retryQuizBtn");
  const exportResultsBtn = document.getElementById("exportResultsBtn");

  scoreValue.textContent = result.score + " / " + result.total_questions;
  percentValue.textContent = result.percent + "%";
  scoreBar.style.setProperty("--target-width", result.percent + "%");
  feedbackText.textContent = result.feedback;

  weakTopicsWrapper.innerHTML = (result.weak_topics || [])
    .map(function (topic) {
      return '<span class="bg-[#F0F0EE] text-[#6B6B6B] text-[12px] font-medium px-3 py-1.5 rounded-full select-none">' +
        window.NudgeApp.escapeHtml(topic.name) + " · " + topic.accuracy + "%</span>";
    })
    .join("") || '<span class="bg-[#D8E8E0] text-[#2D6A4F] text-[12px] font-medium px-3 py-1.5 rounded-full select-none">Strong across all topics</span>';

  wrongAnswersAccordionWrapper.innerHTML = (result.wrong_answers || [])
    .map(function (item, index) {
      return (
        '<div class="bg-white border border-[#E2E2E2] p-5 border-l-[3px] border-l-[#C0392B] rounded-r-xl shadow-sm">' +
        '<div class="flex items-start justify-between gap-4 mb-4"><h4 class="font-serif text-[17px] font-medium text-[#1A1A1A] leading-snug">' +
        window.NudgeApp.escapeHtml(item.question) + '</h4><i data-lucide="x-circle" class="text-[#C0392B] flex-shrink-0 mt-0.5 w-[18px] h-[18px]"></i></div>' +
        '<div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4"><div class="p-3 bg-red-50 rounded-lg border border-red-100"><p class="font-mono text-[10px] text-[#C0392B] uppercase tracking-[0.03em] mb-1">Your answer</p><p class="text-[13px] text-[#1A1A1A] font-medium">' +
        window.NudgeApp.escapeHtml(item.yours) + '</p></div><div class="p-3 bg-[#D8E8E0]/30 rounded-lg border border-[#D8E8E0]"><p class="font-mono text-[10px] text-[#2D6A4F] uppercase tracking-[0.03em] mb-1">Correct answer</p><p class="text-[13px] text-[#1A1A1A] font-medium">' +
        window.NudgeApp.escapeHtml(item.correct) + '</p></div></div><div class="text-[13px] text-[#6B6B6B] leading-relaxed"><span class="font-semibold text-[#2D6A4F]">Explanation: </span>' +
        window.NudgeApp.escapeHtml(item.explanation || "No explanation provided.") + "</div></div>"
      );
    })
    .join("") || '<div class="bg-white border border-[#E2E2E2] p-5 rounded-r-xl shadow-sm text-[14px] text-[#6B6B6B]">No incorrect answers to review. Nicely done.</div>';

  reviewNotesBtn.addEventListener("click", function () {
    if (result.note_id) window.NudgeApp.navigate("/notes/" + result.note_id);
  });
  retryQuizBtn.addEventListener("click", function () {
    if (result.quiz_id) window.NudgeApp.navigate("/quiz/" + result.quiz_id);
  });
  exportResultsBtn.addEventListener("click", function () {
    if (result.note_id) window.NudgeApp.download("/export/json/" + result.note_id);
  });

  window.NudgeApp.createIconSet();
})();
