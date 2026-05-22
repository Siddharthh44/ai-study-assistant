(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData) return;

  function renderRecentNotes(items) {
    const container = document.getElementById("recentNotesContainer");
    if (!container) return;
    container.innerHTML = items.length
      ? items
          .map(function (item) {
            return (
              '<div class="bg-white rounded-xl border border-[#E2E2E2] p-4 flex items-center justify-between shadow-sm group cursor-pointer" data-note-id="' + item.id + '">' +
              '<div class="flex items-center gap-4">' +
              '<div class="w-9 h-9 rounded-lg bg-[#F4F4F2] flex items-center justify-center flex-shrink-0"><i data-lucide="file-text" class="text-[#2D6A4F] w-4 h-4"></i></div>' +
              "<div><h3 class=\"font-serif text-[17px] font-medium text-[#1A1A1A]\">" + window.NudgeApp.escapeHtml(item.title) + "</h3>" +
              '<div class="flex items-center gap-2 mt-1"><span class="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">' + window.NudgeApp.escapeHtml(item.subject) + '</span>' +
              '<span class="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">' + window.NudgeApp.escapeHtml(item.date) + "</span></div></div></div>" +
              '<a href="/notes/' + item.id + '" class="text-[13px] text-[#2D6A4F] font-semibold hover:underline flex items-center gap-1 md:opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">Open <i data-lucide="arrow-right" class="w-3.5 h-3.5"></i></a>' +
              "</div>"
            );
          })
          .join("")
      : '<div class="bg-white rounded-xl border border-[#E2E2E2] p-6 text-[14px] text-[#6B6B6B]">Upload your first study material to populate this dashboard.</div>';

    container.querySelectorAll("[data-note-id]").forEach(function (card) {
      card.addEventListener("click", function () {
        window.NudgeApp.navigate("/notes/" + card.getAttribute("data-note-id"));
      });
    });
  }

  function renderRevision(items) {
    const container = document.getElementById("revisionTasksContainer");
    if (!container) return;
    container.innerHTML = items.length
      ? items
          .map(function (item) {
            return (
              '<div class="p-4 hover:bg-[#F0F0EE] transition-colors flex items-center justify-between">' +
              "<div><p class=\"text-[15px] font-medium text-[#1A1A1A]\">" + window.NudgeApp.escapeHtml(item.title) + '</p>' +
              '<span class="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">' + item.cards + " cards</span></div>" +
              '<button data-note-id="' + item.note_id + '" class="bg-[#2D6A4F] text-white text-xs font-semibold px-4 py-1.5 rounded-lg hover:bg-[#245C43] transition-colors">Start →</button></div>'
            );
          })
          .join("")
      : '<div class="p-4 text-[14px] text-[#6B6B6B]">No cards are due right now.</div>';

    container.querySelectorAll("[data-note-id]").forEach(function (button) {
      button.addEventListener("click", function () {
        window.NudgeApp.navigate("/flashcards?note_id=" + button.getAttribute("data-note-id"));
      });
    });
  }

  const summary = pageData.summary || {};
  window.NudgeApp.setText("dashboardGreeting", "Good morning, " + (pageData.greeting_name || "there") + ".");
  window.NudgeApp.setText("dashboardSubheading", "You have " + (summary.flashcards_due || 0) + " flashcards due and " + (pageData.recent_notes || []).length + " recent notes ready.");
  window.NudgeApp.setText("notesCreatedValue", String(summary.notes_created || 0));
  window.NudgeApp.setText("flashcardsDueValue", String(summary.flashcards_due || 0));
  window.NudgeApp.setText("quizAverageValue", (summary.quiz_average || 0) + "%");
  window.NudgeApp.setText("studyStreakValue", (summary.study_streak || 0) + " days");

  const uploadButtons = ["addNewMaterialBtn", "viewAllNotesLink", "viewAllRevisionLink", "continueGenerateQuizBtn"];
  uploadButtons.forEach(function (id) {
    const node = document.getElementById(id);
    if (!node) return;
    node.addEventListener("click", function (event) {
      if (node.tagName === "A") event.preventDefault();
      const target = id === "viewAllNotesLink" ? "/notes" : id === "viewAllRevisionLink" ? "/flashcards" : "/upload";
      if (id === "continueGenerateQuizBtn" && pageData.continue_note) {
        window.NudgeApp.navigate("/quizzes");
        return;
      }
      window.NudgeApp.navigate(target);
    });
  });

  const continueNote = pageData.continue_note;
  if (continueNote) {
    window.NudgeApp.setText("continueTitle", continueNote.title);
    window.NudgeApp.setText("continueSubject", continueNote.subject);
    window.NudgeApp.setText("continueEdited", "Last edited " + continueNote.date);
    document.getElementById("continueViewNotesBtn").addEventListener("click", function () {
      window.NudgeApp.navigate("/notes/" + continueNote.id);
    });
  }

  renderRecentNotes(pageData.recent_notes || []);
  renderRevision(pageData.revision_tasks || []);
  window.NudgeApp.createIconSet();
})();
