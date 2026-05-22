// ── Card search filter ────────────────────────────────────────────────────
// Usage: initCardSearch('searchInputId', '.card-selector')
function initCardSearch(inputId, cardSelector) {
  const input = document.getElementById(inputId);
  if (!input) return;
  input.addEventListener("input", () => {
    const value = input.value.toLowerCase();
    document.querySelectorAll(cardSelector).forEach(card => {
      card.style.display = card.innerText.toLowerCase().includes(value) ? "" : "none";
    });
  });
}

// ── Sidebar active link highlighter ──────────────────────────────────────
// Adds the given class(es) to any <a> in nav/aside whose href matches the current path.
function highlightActiveNavLink(activeClass = "bg-violet-600") {
  const path = window.location.pathname;
  document.querySelectorAll("nav a, aside a").forEach(link => {
    if (link.getAttribute("href") === path) {
      link.classList.add(...activeClass.split(" "));
    }
  });
}

// ── Toggle switch ─────────────────────────────────────────────────────────
function toggleSwitch(element) {
  element.classList.toggle("active");
}

// ── Empty state checker ───────────────────────────────────────────────────
// Shows #emptyState if no elements matching cardSelector remain visible.
function checkEmptyState(cardSelector, emptyStateId = "emptyState") {
  const remaining = document.querySelectorAll(cardSelector);
  const emptyEl = document.getElementById(emptyStateId);
  if (!emptyEl) return;
  emptyEl.classList.toggle("hidden", remaining.length > 0);
}

// ── Auto-run on DOMContentLoaded ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  highlightActiveNavLink();
});
