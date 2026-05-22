(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData || !pageData.items) return;

  let activeFilter = "All";
  let searchQuery = "";
  let isRevisionMode = false;
  let revisionIndex = 0;

  const contentFeed = document.getElementById("contentFeed");
  const filterButtonGroup = document.getElementById("filterButtonGroup");
  const searchField = document.getElementById("searchField");
  const toggleRevisionModeBtn = document.getElementById("toggleRevisionModeBtn");
  const toggleCircle = document.getElementById("toggleCircle");
  const revisionOverlay = document.getElementById("revisionOverlay");
  const closeOverlayBtn = document.getElementById("closeOverlayBtn");
  const revisionFocusCard = document.getElementById("revisionFocusCard");
  const prevRevisionBtn = document.getElementById("prevRevisionBtn");
  const nextRevisionBtn = document.getElementById("nextRevisionBtn");
  const overlayHeaderCounter = document.getElementById("overlayHeaderCounter");
  const overlayFooterCounter = document.getElementById("overlayFooterCounter");
  const filterOptions = ["All", "Notes", "Flashcards", "Quiz Questions"];
  const typeMap = { Notes: "note", Flashcards: "flashcard", "Quiz Questions": "quiz" };

  searchField.addEventListener("input", function (event) {
    searchQuery = event.target.value.trim().toLowerCase();
    render();
  });

  toggleRevisionModeBtn.addEventListener("click", function () {
    isRevisionMode = !isRevisionMode;
    toggleRevisionModeBtn.className = isRevisionMode
      ? "relative w-10 h-[22px] rounded-full bg-[#2D6A4F] transition-colors flex-shrink-0 focus:outline-none"
      : "relative w-10 h-[22px] rounded-full bg-[#E2E2E2] transition-colors flex-shrink-0 focus:outline-none";
    toggleCircle.style.transform = isRevisionMode ? "translateX(18px)" : "translateX(0)";
    if (isRevisionMode) openOverlay();
    else revisionOverlay.classList.add("hidden");
  });
  closeOverlayBtn.addEventListener("click", function () {
    revisionOverlay.classList.add("hidden");
    isRevisionMode = false;
    toggleRevisionModeBtn.click();
  });
  prevRevisionBtn.addEventListener("click", function () {
    revisionIndex = Math.max(0, revisionIndex - 1);
    renderOverlay();
  });
  nextRevisionBtn.addEventListener("click", function () {
    revisionIndex = Math.min(filtered().length - 1, revisionIndex + 1);
    renderOverlay();
  });

  function filtered() {
    return pageData.items.filter(function (item) {
      const matchesFilter = activeFilter === "All" || item.content_type === typeMap[activeFilter];
      const matchesSearch = !searchQuery || item.title.toLowerCase().includes(searchQuery) || item.subject.toLowerCase().includes(searchQuery);
      return matchesFilter && matchesSearch;
    });
  }

  function renderFilters() {
    filterButtonGroup.innerHTML = filterOptions
      .map(function (filter) {
        const active = activeFilter === filter;
        return '<button data-filter="' + filter + '" class="px-4 py-2 rounded-full text-[13px] font-medium transition-colors border ' +
          (active ? "bg-[#2D6A4F] text-white border-[#2D6A4F]" : "bg-white text-[#6B6B6B] border-[#E2E2E2] hover:border-[#2D6A4F] hover:text-[#2D6A4F]") +
          '">' + filter + "</button>";
      })
      .join("");
    filterButtonGroup.querySelectorAll("[data-filter]").forEach(function (button) {
      button.addEventListener("click", function () {
        activeFilter = button.getAttribute("data-filter");
        render();
      });
    });
  }

  function render() {
    renderFilters();
    const items = filtered();
    if (!items.length) {
      contentFeed.innerHTML = '<div class="bg-white rounded-xl border border-[#E2E2E2] p-6 text-[14px] text-[#6B6B6B]">No bookmarks match that search yet.</div>';
      return;
    }
    contentFeed.innerHTML =
      '<div class="masonry-grid">' +
      items
        .map(function (item) {
          return (
            '<div class="masonry-item bg-white rounded-xl border border-[#E2E2E2] border-l-[3px] p-5 shadow-sm">' +
            '<div class="flex items-center justify-between gap-4 mb-3"><div><p class="font-serif text-[20px] font-medium text-[#1A1A1A]">' +
            window.NudgeApp.escapeHtml(item.title) + '</p><p class="text-[12px] text-[#6B6B6B] mt-1">' +
            window.NudgeApp.escapeHtml(item.subject) + " · " + window.NudgeApp.escapeHtml(item.meta || "") + "</p></div></div>" +
            '<p class="text-[14px] text-[#6B6B6B] leading-relaxed">' + window.NudgeApp.escapeHtml(item.content) + "</p></div>"
          );
        })
        .join("") +
      "</div>";
  }

  function openOverlay() {
    revisionIndex = 0;
    revisionOverlay.classList.remove("hidden");
    renderOverlay();
  }

  function renderOverlay() {
    const items = filtered();
    const active = items[revisionIndex];
    if (!active) return;
    revisionFocusCard.innerHTML =
      '<p class="font-mono text-[11px] text-[#6B6B6B] tracking-[0.03em] mb-2">' + window.NudgeApp.escapeHtml(active.subject) + '</p>' +
      '<h3 class="font-serif text-[28px] font-medium text-[#1A1A1A] mb-4">' + window.NudgeApp.escapeHtml(active.title) + '</h3>' +
      '<p class="text-[15px] text-[#6B6B6B] leading-relaxed">' + window.NudgeApp.escapeHtml(active.content) + "</p>";
    overlayHeaderCounter.textContent = "Revision Mode";
    overlayFooterCounter.textContent = revisionIndex + 1 + " / " + items.length;
    prevRevisionBtn.disabled = revisionIndex === 0;
    nextRevisionBtn.disabled = revisionIndex === items.length - 1;
  }

  render();
  window.NudgeApp.createIconSet();
})();
