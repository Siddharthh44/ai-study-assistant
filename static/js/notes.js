(function () {
  const pageData = window.PAGE_DATA || {};

  if (pageData.items) {
    initNotesIndex(pageData);
  }
  if (pageData.note) {
    initNoteDetail(pageData);
  }

  function initNotesIndex(data) {
    let searchString = "";
    let layoutView = "grid";
    let activeSubjectFilter = "All";

    const notesCountLabel = document.getElementById("notesCountLabel");
    const subjectPillsWrapper = document.getElementById("subjectPillsWrapper");
    const searchInput = document.getElementById("searchInput");
    const btnViewGrid = document.getElementById("btnViewGrid");
    const btnViewList = document.getElementById("btnViewList");
    const emptyStateTarget = document.getElementById("emptyStateTarget");
    const gridDisplayContainer = document.getElementById("gridDisplayContainer");
    const listDisplayContainer = document.getElementById("listDisplayContainer");
    const listDisplayInnerRowGroup = document.getElementById("listDisplayInnerRowGroup");
    const newNoteButtons = document.querySelectorAll("[data-notes-upload]");

    notesCountLabel.textContent = data.items.length + " notes generated from your content.";
    searchInput.addEventListener("input", function (event) {
      searchString = event.target.value.trim().toLowerCase();
      render();
    });
    btnViewGrid.addEventListener("click", function () {
      layoutView = "grid";
      render();
    });
    btnViewList.addEventListener("click", function () {
      layoutView = "list";
      render();
    });
    newNoteButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        window.NudgeApp.navigate("/upload");
      });
    });

    function renderPills() {
      subjectPillsWrapper.innerHTML = data.subjects
        .map(function (subject) {
          const active = activeSubjectFilter === subject;
          return '<button data-subject="' + subject + '" class="px-4 py-1.5 rounded-full text-[13px] font-medium transition-all border focus:outline-none ' +
            (active ? 'bg-[#2D6A4F] text-white border-[#2D6A4F]' : 'bg-white text-[#6B6B6B] border-[#E2E2E2] hover:border-[#2D6A4F] hover:text-[#2D6A4F]') +
            '">' + window.NudgeApp.escapeHtml(subject) + "</button>";
        })
        .join("");
      subjectPillsWrapper.querySelectorAll("[data-subject]").forEach(function (button) {
        button.addEventListener("click", function () {
          activeSubjectFilter = button.getAttribute("data-subject");
          render();
        });
      });
    }

    function filtered() {
      return data.items.filter(function (note) {
        const matchesSubject = activeSubjectFilter === "All" || note.subject === activeSubjectFilter;
        const matchesSearch = !searchString || note.title.toLowerCase().includes(searchString) || note.subject.toLowerCase().includes(searchString);
        return matchesSubject && matchesSearch;
      });
    }

    function renderGrid(items) {
      gridDisplayContainer.innerHTML = items
        .map(function (note) {
          return (
            '<div class="bg-white rounded-xl border border-[#E2E2E2] p-5 shadow-sm hover:border-[#2D6A4F] transition-colors cursor-pointer" data-open-note="' + note.id + '">' +
            '<div class="flex items-start justify-between gap-3 mb-4"><div><div class="flex items-center gap-2 mb-2"><span class="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">' +
            window.NudgeApp.escapeHtml(note.subject) + '</span><span class="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">' + window.NudgeApp.escapeHtml(note.date) + '</span></div>' +
            '<h3 class="font-serif text-[18px] font-medium text-[#1A1A1A] leading-snug">' + window.NudgeApp.escapeHtml(note.title) + '</h3></div>' +
            '<button class="text-[#6B6B6B] hover:text-[#2D6A4F] focus:outline-none" data-delete-note="' + note.id + '"><i data-lucide="trash-2" class="w-4 h-4"></i></button></div>' +
            '<p class="text-[14px] text-[#6B6B6B] leading-relaxed mb-5">' + window.NudgeApp.escapeHtml(note.preview) + '</p>' +
            '<div class="flex items-center justify-between text-[13px]"><span class="text-[#6B6B6B]">' + note.flashcards + ' flashcards</span>' +
            '<button class="text-[#2D6A4F] font-semibold hover:underline focus:outline-none">Open →</button></div></div>'
          );
        })
        .join("");
    }

    function renderList(items) {
      listDisplayInnerRowGroup.innerHTML = items
        .map(function (note) {
          return (
            '<div class="p-4 flex items-center justify-between gap-4" data-open-note="' + note.id + '">' +
            '<div><h3 class="font-serif text-[18px] font-medium text-[#1A1A1A]">' + window.NudgeApp.escapeHtml(note.title) + '</h3>' +
            '<div class="flex items-center gap-2 mt-1"><span class="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">' + window.NudgeApp.escapeHtml(note.subject) + '</span>' +
            '<span class="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em]">' + window.NudgeApp.escapeHtml(note.date) + '</span></div></div>' +
            '<div class="flex items-center gap-3"><span class="text-[13px] text-[#6B6B6B]">' + note.flashcards + ' cards</span>' +
            '<button class="text-[#6B6B6B] hover:text-[#C0392B] focus:outline-none" data-delete-note="' + note.id + '"><i data-lucide="trash-2" class="w-4 h-4"></i></button></div></div>'
          );
        })
        .join("");
    }

    async function deleteNote(noteId) {
      await window.NudgeApp.jsonFetch("/api/notes/" + noteId, { method: "DELETE" });
      const index = data.items.findIndex(function (item) {
        return String(item.id) === String(noteId);
      });
      if (index >= 0) data.items.splice(index, 1);
      notesCountLabel.textContent = data.items.length + " notes generated from your content.";
      render();
    }

    function bindActions() {
      document.querySelectorAll("[data-open-note]").forEach(function (node) {
        node.addEventListener("click", function (event) {
          if (event.target.closest("[data-delete-note]")) return;
          window.NudgeApp.navigate("/notes/" + node.getAttribute("data-open-note"));
        });
      });
      document.querySelectorAll("[data-delete-note]").forEach(function (button) {
        button.addEventListener("click", function (event) {
          event.stopPropagation();
          deleteNote(button.getAttribute("data-delete-note"));
        });
      });
      window.NudgeApp.createIconSet();
    }

    function render() {
      renderPills();
      const items = filtered();
      emptyStateTarget.classList.toggle("hidden", items.length > 0);
      if (layoutView === "grid") {
        btnViewGrid.className = "p-2 transition-colors bg-[#F0F0EE] text-[#1A1A1A]";
        btnViewList.className = "p-2 transition-colors text-[#6B6B6B]";
        gridDisplayContainer.classList.remove("hidden");
        listDisplayContainer.classList.add("hidden");
        renderGrid(items);
      } else {
        btnViewList.className = "p-2 transition-colors bg-[#F0F0EE] text-[#1A1A1A]";
        btnViewGrid.className = "p-2 transition-colors text-[#6B6B6B]";
        listDisplayContainer.classList.remove("hidden");
        gridDisplayContainer.classList.add("hidden");
        renderList(items);
      }
      bindActions();
    }

    render();
  }

  function initNoteDetail(data) {
    const note = data.note;
    const noteContentCard = document.getElementById("noteContentCard");
    if (!noteContentCard) return;

    const titleText = document.getElementById("titleText");
    const titleInput = document.getElementById("titleInput");
    const noteSubjectBadge = document.getElementById("noteSubjectBadge");
    const noteDateText = document.getElementById("noteDateText");
    const bookmarkBtn = document.getElementById("bookmarkBtn");
    const editTitleBtn = document.getElementById("editTitleBtn");
    const conceptsCapsulesContainer = document.getElementById("conceptsCapsulesContainer");
    const flashcardsCountBadge = document.getElementById("flashcardsCountBadge");
    const flashcardsLaunchBtn = document.getElementById("flashcardsLaunchBtn");
    const quizLaunchBtn = document.getElementById("quizLaunchBtn");
    const exportPdfBtn = document.getElementById("exportPdfBtn");
    const exportTxtBtn = document.getElementById("exportTxtBtn");
    const editNoteBtn = document.getElementById("editNoteBtn");
    const feedbackUpBtn = document.getElementById("feedbackUpBtn");
    const feedbackDownBtn = document.getElementById("feedbackDownBtn");

    titleText.textContent = note.title;
    titleInput.value = note.title;
    noteSubjectBadge.textContent = note.subject;
    noteDateText.textContent = "Generated " + note.date + " · " + note.generated_at;
    flashcardsCountBadge.textContent = data.flashcards.length + " cards";
    noteContentCard.innerHTML = note.notes_html || '<p class="text-[15px] text-[#1A1A1A] leading-relaxed">Your generated note will appear here once processing finishes.</p>';
    conceptsCapsulesContainer.innerHTML = note.key_concepts
      .map(function (concept) {
        return '<span class="bg-[#F4F4F2] text-[#1A1A1A] text-[13px] px-3 py-2 rounded-full border border-[#E2E2E2]">' + window.NudgeApp.escapeHtml(concept.term) + "</span>";
      })
      .join("");

    function syncBookmark(bookmarked) {
      bookmarkBtn.className = bookmarked
        ? "w-9 h-9 rounded-lg flex items-center justify-center text-[#2D6A4F] bg-[#F0F0EE] transition-colors focus:outline-none"
        : "w-9 h-9 rounded-lg flex items-center justify-center text-[#6B6B6B] hover:text-[#2D6A4F] hover:bg-[#F0F0EE] transition-colors focus:outline-none";
    }

    async function saveTitle() {
      titleInput.classList.add("hidden");
      titleText.classList.remove("hidden");
      titleText.textContent = titleInput.value.trim() || note.title;
      const updated = await window.NudgeApp.jsonFetch("/api/notes/" + note.id, {
        method: "PATCH",
        body: JSON.stringify({ title: titleInput.value.trim() }),
      });
      const updatedNote = updated.note || updated;
      note.title = updatedNote.title;
      titleText.textContent = updatedNote.title;
    }

    syncBookmark(note.bookmarked);

    titleText.addEventListener("click", function () {
      titleText.classList.add("hidden");
      titleInput.classList.remove("hidden");
      titleInput.focus();
    });
    titleInput.addEventListener("blur", saveTitle);
    titleInput.addEventListener("keydown", function (event) {
      if (event.key === "Enter") saveTitle();
    });
    editNoteBtn.addEventListener("click", function () {
      titleText.click();
    });
    editTitleBtn.addEventListener("click", function () {
      titleText.click();
    });
    bookmarkBtn.addEventListener("click", async function () {
      const response = await window.NudgeApp.jsonFetch("/api/bookmarks/toggle", {
        method: "POST",
        body: JSON.stringify({ content_type: "note", content_id: note.id }),
      });
      note.bookmarked = response.bookmarked;
      syncBookmark(note.bookmarked);
    });
    flashcardsLaunchBtn.addEventListener("click", function () {
      window.NudgeApp.navigate("/flashcards?note_id=" + note.id);
    });
    quizLaunchBtn.addEventListener("click", function () {
      if (data.quiz) window.NudgeApp.navigate("/quiz/" + data.quiz.id);
    });
    exportPdfBtn.addEventListener("click", function () {
      window.NudgeApp.download("/export/pdf/" + note.id);
    });
    exportTxtBtn.addEventListener("click", function () {
      window.NudgeApp.download("/export/txt/" + note.id);
    });
    feedbackUpBtn.addEventListener("click", function () {
      feedbackUpBtn.classList.add("text-[#2D6A4F]");
      feedbackDownBtn.classList.remove("text-[#C0392B]");
    });
    feedbackDownBtn.addEventListener("click", function () {
      feedbackDownBtn.classList.add("text-[#C0392B]");
      feedbackUpBtn.classList.remove("text-[#2D6A4F]");
    });
    window.NudgeApp.createIconSet();
  }
})();
