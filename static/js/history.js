(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData || !pageData.items) return;

  const filterPillsGroup = document.getElementById("filterPillsGroup");
  const searchField = document.getElementById("searchField");
  const sortSelector = document.getElementById("sortSelector");
  const journalTimelineTarget = document.getElementById("journalTimelineTarget");
  const emptyStateDisplay = document.getElementById("emptyStateDisplay");
  const categories = ["All", "Notes", "Flashcards", "Quizzes", "Exports"];
  const typeMap = { Notes: "note", Flashcards: "flashcard", Quizzes: "quiz", Exports: "exported" };
  let currentFilter = "All";
  let searchQuery = "";
  let currentSortOrder = "newest";

  searchField.addEventListener("input", function (event) {
    searchQuery = event.target.value.trim().toLowerCase();
    render();
  });
  sortSelector.addEventListener("change", function (event) {
    currentSortOrder = event.target.value;
    render();
  });

  function filtered() {
    const items = pageData.items.filter(function (item) {
      const matchesFilter = currentFilter === "All" || item.content_type === typeMap[currentFilter] || item.event_type === typeMap[currentFilter];
      const matchesSearch = !searchQuery || item.title.toLowerCase().includes(searchQuery) || item.description.toLowerCase().includes(searchQuery);
      return matchesFilter && matchesSearch;
    });
    return items.sort(function (a, b) {
      return currentSortOrder === "oldest"
        ? new Date(a.created_at) - new Date(b.created_at)
        : new Date(b.created_at) - new Date(a.created_at);
    });
  }

  function renderFilters() {
    filterPillsGroup.innerHTML = categories
      .map(function (category) {
        const active = currentFilter === category;
        return '<button data-filter="' + category + '" class="' +
          (active
            ? "bg-[#2D6A4F] text-white border-[#2D6A4F] px-4 py-1.5 rounded-full text-[13px] font-medium transition-all border focus:outline-none"
            : "bg-white text-[#6B6B6B] border-[#E2E2E2] hover:border-[#2D6A4F] hover:text-[#2D6A4F] px-4 py-1.5 rounded-full text-[13px] font-medium transition-all border focus:outline-none") +
          '">' + category + "</button>";
      })
      .join("");
    filterPillsGroup.querySelectorAll("[data-filter]").forEach(function (button) {
      button.addEventListener("click", function () {
        currentFilter = button.getAttribute("data-filter");
        render();
      });
    });
  }

  function render() {
    renderFilters();
    const items = filtered();
    emptyStateDisplay.classList.toggle("hidden", items.length > 0);
    journalTimelineTarget.innerHTML = items
      .map(function (item) {
        return (
          '<div class="bg-white rounded-xl border border-[#E2E2E2] p-5 shadow-sm">' +
          '<div class="flex items-start justify-between gap-4"><div><p class="font-serif text-[20px] font-medium text-[#1A1A1A]">' +
          window.NudgeApp.escapeHtml(item.title) + '</p><p class="text-[13px] text-[#6B6B6B] mt-1">' +
          window.NudgeApp.escapeHtml(item.description) + '</p></div><span class="font-mono text-[11px] text-[#6B6B6B]">' +
          item.date + "</span></div></div>"
        );
      })
      .join("");
  }

  render();
  window.NudgeApp.createIconSet();
})();
