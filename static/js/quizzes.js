(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData || !pageData.items) return;

  let currentSearchString = "";
  let currentFilterCategory = "All";
  const totalQuizzesStat = document.getElementById("totalQuizzesStat");
  const averageScoreStat = document.getElementById("averageScoreStat");
  const timeSavedStat = document.getElementById("timeSavedStat");
  const filterButtonsWrapper = document.getElementById("filterButtonsWrapper");
  const quizSearchInput = document.getElementById("quizSearchInput");
  const quizCardsDisplayDeck = document.getElementById("quizCardsDisplayDeck");
  const createQuizButton = document.getElementById("generateQuizBtn");

  totalQuizzesStat.textContent = pageData.stats.total_quizzes;
  averageScoreStat.textContent = pageData.stats.average_score + "%";
  timeSavedStat.textContent = pageData.stats.time_saved_hours + "h";
  quizSearchInput.addEventListener("input", function (event) {
    currentSearchString = event.target.value.trim().toLowerCase();
    render();
  });
  createQuizButton.addEventListener("click", function () {
    window.NudgeApp.navigate("/upload");
  });

  function renderFilters() {
    const criteriaCategories = ["All", "New", "Attempted"];
    filterButtonsWrapper.innerHTML = criteriaCategories
      .map(function (criteria) {
        const active = currentFilterCategory === criteria;
        return '<button data-filter="' + criteria + '" class="' +
          (active
            ? "bg-[#2D6A4F] text-white border-[#2D6A4F] px-4 py-1.5 rounded-full text-[13px] font-medium transition-all border focus:outline-none"
            : "bg-white text-[#6B6B6B] border-[#E2E2E2] hover:border-[#2D6A4F] hover:text-[#2D6A4F] px-4 py-1.5 rounded-full text-[13px] font-medium transition-all border focus:outline-none") +
          '">' + criteria + "</button>";
      })
      .join("");
    filterButtonsWrapper.querySelectorAll("[data-filter]").forEach(function (button) {
      button.addEventListener("click", function () {
        currentFilterCategory = button.getAttribute("data-filter");
        render();
      });
    });
  }

  function filteredItems() {
    return pageData.items.filter(function (quiz) {
      const matchesFilter =
        currentFilterCategory === "All" ||
        (currentFilterCategory === "Attempted" && quiz.attempted) ||
        (currentFilterCategory === "New" && !quiz.attempted);
      const matchesSearch =
        !currentSearchString ||
        quiz.title.toLowerCase().includes(currentSearchString) ||
        quiz.subject.toLowerCase().includes(currentSearchString);
      return matchesFilter && matchesSearch;
    });
  }

  function render() {
    renderFilters();
    const items = filteredItems();
    if (!items.length) {
      quizCardsDisplayDeck.innerHTML = '<div class="bg-white rounded-xl border border-[#E2E2E2] p-6 text-[14px] text-[#6B6B6B]">No quizzes match that filter yet.</div>';
      return;
    }

    quizCardsDisplayDeck.innerHTML = items
      .map(function (quiz) {
        return (
          '<div class="bg-white rounded-xl border border-[#E2E2E2] p-5 shadow-sm mb-4">' +
          '<div class="flex items-start justify-between gap-4 mb-4"><div><div class="flex items-center gap-2 mb-2"><span class="bg-[#D8E8E0] text-[#2D6A4F] text-[11px] font-medium px-2.5 py-0.5 rounded-full">' +
          window.NudgeApp.escapeHtml(quiz.subject) + '</span><span class="font-mono text-[11px] text-[#6B6B6B]">' + quiz.date + "</span></div>" +
          '<h3 class="font-serif text-[22px] font-medium text-[#1A1A1A]">' + window.NudgeApp.escapeHtml(quiz.title) + "</h3></div>" +
          (quiz.last_score !== null ? '<span class="font-mono text-[11px] font-medium bg-[#D8E8E0] text-[#2D6A4F] px-2 py-0.5 rounded-full">' + quiz.last_score + "%</span>" : '<span class="font-mono text-[11px] font-medium bg-[#F0F0EE] text-[#6B6B6B] px-2 py-0.5 rounded-full">New</span>') +
          "</div>" +
          '<div class="flex items-center justify-between"><span class="text-[14px] text-[#6B6B6B]">' + quiz.questions + ' questions</span>' +
          '<button data-quiz-id="' + quiz.id + '" class="bg-[#2D6A4F] text-white text-[13px] font-semibold px-4 py-2 rounded-lg hover:bg-[#245C43] transition-colors">' +
          (quiz.attempted ? "Retry Quiz →" : "Start Quiz →") + "</button></div></div>"
        );
      })
      .join("");

    quizCardsDisplayDeck.querySelectorAll("[data-quiz-id]").forEach(function (button) {
      button.addEventListener("click", function () {
        window.NudgeApp.navigate("/quiz/" + button.getAttribute("data-quiz-id"));
      });
    });
  }

  render();
  window.NudgeApp.createIconSet();
})();
