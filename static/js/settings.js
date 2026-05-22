(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData) return;

  const navItems = [
    { id: "profile", label: "Profile", icon: "👤" },
    { id: "learning", label: "Learning Preferences", icon: "🎓" },
    { id: "ai", label: "AI Behavior", icon: "🤖" },
    { id: "data", label: "Data & Storage", icon: "📦" },
  ];

  const state = {
    activeSection: "profile",
    profile: pageData.profile,
    ai: pageData.ai,
    learning: pageData.learning,
    export: pageData.export,
    theme: pageData.theme,
  };

  const navItemsList = document.getElementById("navItemsList");
  const settingsDynamicContentArea = document.getElementById("settingsDynamicContentArea");

  function renderNavigationDrawer() {
    navItemsList.innerHTML = navItems
      .map(function (item) {
        const active = state.activeSection === item.id;
        return '<li><button data-section="' + item.id + '" class="' +
          (active
            ? "bg-[#D8E8E0]/50 text-[#2D6A4F] w-full flex items-center gap-3 h-10 px-4 rounded-lg transition-all text-[14px] font-medium relative text-left focus:outline-none"
            : "text-[#6B6B6B] hover:bg-[#F0F0EE] hover:text-[#1A1A1A] w-full flex items-center gap-3 h-10 px-4 rounded-lg transition-all text-[14px] font-medium relative text-left focus:outline-none") +
          '"><span class="select-none">' + item.icon + "</span><span>" + item.label + "</span></button></li>";
      })
      .join("");

    navItemsList.querySelectorAll("[data-section]").forEach(function (button) {
      button.addEventListener("click", function () {
        state.activeSection = button.getAttribute("data-section");
        renderNavigationDrawer();
        renderSection();
      });
    });
  }

  function renderSection() {
    if (state.activeSection === "profile") {
      settingsDynamicContentArea.innerHTML =
        '<div class="space-y-6"><div class="bg-white border border-[#E2E2E2] rounded-xl p-6 space-y-4 shadow-sm">' +
        field("Full Name", "settingsFullName", state.profile.name) +
        field("Institution", "settingsInstitution", state.profile.institution) +
        field("Exam", "settingsExam", state.profile.exam) +
        field("Exam Date", "settingsExamDate", state.profile.exam_date, "date") +
        '<button id="saveProfileBtn" class="bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors">Save Changes →</button></div></div>';
      document.getElementById("saveProfileBtn").addEventListener("click", saveProfile);
    } else if (state.activeSection === "learning") {
      settingsDynamicContentArea.innerHTML =
        '<div class="bg-white border border-[#E2E2E2] rounded-xl p-6 space-y-4 shadow-sm">' +
        checkbox("Enable spaced repetition", "learningSpaced", state.learning.spaced_repetition) +
        checkbox("Show hints during revision", "learningHints", state.learning.show_hints) +
        checkbox("Track progress analytics", "learningProgress", state.learning.progress_tracking) +
        '<button id="saveLearningBtn" class="bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors">Save Preferences →</button></div>';
      document.getElementById("saveLearningBtn").addEventListener("click", saveLearning);
    } else if (state.activeSection === "ai") {
      settingsDynamicContentArea.innerHTML =
        '<div class="bg-white border border-[#E2E2E2] rounded-xl p-6 space-y-4 shadow-sm">' +
        selectField("Note Length", "aiNoteLength", ["Short", "Balanced", "Detailed"], state.ai.note_length) +
        selectField("Tone", "aiTone", ["Academic", "Friendly", "Exam-focused"], state.ai.tone) +
        checkbox("Auto-generate flashcards", "aiAutoFlashcards", state.ai.auto_flashcards) +
        checkbox("Auto-generate quizzes", "aiAutoQuiz", state.ai.auto_quiz) +
        field("Questions Per Quiz", "aiQuestionsPerQuiz", state.ai.questions_per_quiz, "number") +
        field("Reminder Time", "aiReminderTime", state.ai.reminder_time, "time") +
        '<button id="saveAiBtn" class="bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors">Save AI Preferences →</button></div>';
      document.getElementById("saveAiBtn").addEventListener("click", saveAi);
    } else {
      settingsDynamicContentArea.innerHTML =
        '<div class="bg-white border border-[#E2E2E2] rounded-xl p-6 space-y-4 shadow-sm"><p class="text-[14px] text-[#6B6B6B]">Reset progress, export defaults, and logout live in this section.</p>' +
        '<div class="flex flex-wrap gap-3"><button id="resetProgressBtn" class="bg-[#F4F4F2] border border-[#E2E2E2] text-[#1A1A1A] text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#F0F0EE] transition-colors">Reset Progress</button>' +
        '<button id="logoutBtn" class="bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors">Logout</button></div></div>';
      document.getElementById("resetProgressBtn").addEventListener("click", async function () {
        await window.NudgeApp.jsonFetch("/api/settings/reset-progress", { method: "POST" });
      });
      document.getElementById("logoutBtn").addEventListener("click", async function () {
        await fetch("/logout", { method: "POST", credentials: "same-origin" });
        window.NudgeApp.navigate("/login");
      });
    }
  }

  function field(label, id, value, type) {
    return '<div><label class="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">' + label + '</label><input id="' + id + '" type="' + (type || "text") + '" value="' + window.NudgeApp.escapeHtml(value || "") + '" class="w-full h-10 px-4 bg-white border border-[#E2E2E2] rounded-lg text-[14px] placeholder:text-[#6B6B6B] focus:outline-none focus:border-[#2D6A4F] transition-colors"></div>';
  }

  function checkbox(label, id, checked) {
    return '<label class="flex items-center gap-3 text-[14px] text-[#1A1A1A]"><input id="' + id + '" type="checkbox" class="accent-[#2D6A4F]" ' + (checked ? "checked" : "") + ">" + label + "</label>";
  }

  function selectField(label, id, options, selected) {
    return '<div><label class="block text-[13px] font-medium text-[#6B6B6B] mb-1.5">' + label + '</label><select id="' + id + '" class="w-full h-10 px-4 bg-white border border-[#E2E2E2] rounded-lg text-[14px] focus:outline-none focus:border-[#2D6A4F] transition-colors">' +
      options.map(function (option) { return '<option ' + (selected === option ? "selected" : "") + ">" + option + "</option>"; }).join("") +
      "</select></div>";
  }

  async function saveProfile() {
    const updated = await window.NudgeApp.jsonFetch("/api/settings", {
      method: "POST",
      body: JSON.stringify({
        full_name: document.getElementById("settingsFullName").value,
        institution: document.getElementById("settingsInstitution").value,
        exam: document.getElementById("settingsExam").value,
        exam_date: document.getElementById("settingsExamDate").value,
      }),
    });
    state.profile = updated.profile;
  }

  async function saveLearning() {
    const updated = await window.NudgeApp.jsonFetch("/api/settings", {
      method: "POST",
      body: JSON.stringify({
        spaced_repetition: document.getElementById("learningSpaced").checked,
        show_hints: document.getElementById("learningHints").checked,
        progress_tracking: document.getElementById("learningProgress").checked,
      }),
    });
    state.learning = updated.learning;
  }

  async function saveAi() {
    const updated = await window.NudgeApp.jsonFetch("/api/settings", {
      method: "POST",
      body: JSON.stringify({
        note_length: document.getElementById("aiNoteLength").value,
        ai_tone: document.getElementById("aiTone").value,
        auto_flashcards: document.getElementById("aiAutoFlashcards").checked,
        auto_quiz: document.getElementById("aiAutoQuiz").checked,
        questions_per_quiz: Number(document.getElementById("aiQuestionsPerQuiz").value || 10),
        reminder_time: document.getElementById("aiReminderTime").value,
      }),
    });
    state.ai = updated.ai;
  }

  renderNavigationDrawer();
  renderSection();
})();
