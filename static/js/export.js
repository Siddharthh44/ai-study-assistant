(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData || !pageData.items) return;

  const contentItemsContainer = document.getElementById("contentItemsContainer");
  const selectedCountDisplay = document.getElementById("selectedCountDisplay");
  const summarySelectedCount = document.getElementById("summarySelectedCount");
  const formatCardsGrid = document.getElementById("formatCardsGrid");
  const summaryFormatLabel = document.getElementById("summaryFormatLabel");
  const summaryFontLabel = document.getElementById("summaryFontLabel");
  const pdfCheckboxesGrid = document.getElementById("pdfCheckboxesGrid");
  const fontSizeButtonGroup = document.getElementById("fontSizeButtonGroup");
  const customHeaderField = document.getElementById("customHeaderField");
  const downloadExportBtn = document.getElementById("downloadExportBtn");
  const selectAllBtn = document.getElementById("selectAllBtn");
  const liveMetricsDisplay = document.getElementById("liveMetricsDisplay");

  const formats = [
    { id: "pdf", label: "PDF Document", desc: "Formatted PDF export.", icon: "📄" },
    { id: "txt", label: "Plain Text", desc: "Simple text export.", icon: "📝" },
    { id: "json", label: "JSON Data", desc: "Structured export for reuse.", icon: "🧾" },
  ];
  const selectedSet = new Set(pageData.selected_ids || []);
  let activeFormat = (pageData.settings && pageData.settings.export_format) || "pdf";
  let activeFontSize = (pageData.settings && pageData.settings.export_font_size) || "Medium";

  function selectedItems() {
    return pageData.items.filter(function (item) {
      return selectedSet.has(item.selection_key || ((item.content_type || "note") + ":" + item.id));
    });
  }

  function renderItems() {
    contentItemsContainer.innerHTML = pageData.items
      .map(function (item) {
        const selectionKey = item.selection_key || ((item.content_type || "note") + ":" + item.id);
        const checked = selectedSet.has(selectionKey);
        return (
          '<label class="flex items-center justify-between gap-4 py-3 border-b border-[#F4F4F2]">' +
          '<div class="flex items-center gap-3"><input type="checkbox" data-item-key="' + selectionKey + '" ' + (checked ? "checked" : "") + ' class="w-4 h-4 accent-[#2D6A4F]">' +
          '<div><p class="text-[14px] font-medium text-[#1A1A1A]">' + window.NudgeApp.escapeHtml(item.title) + '</p><p class="text-[12px] text-[#6B6B6B]">' +
          window.NudgeApp.escapeHtml(item.type) + "</p></div></div></label>"
        );
      })
      .join("");
    contentItemsContainer.querySelectorAll("[data-item-key]").forEach(function (checkbox) {
      checkbox.addEventListener("change", function () {
        const selectionKey = checkbox.getAttribute("data-item-key");
        if (checkbox.checked) selectedSet.add(selectionKey);
        else selectedSet.delete(selectionKey);
        renderSummary();
      });
    });
  }

  function renderFormats() {
    formatCardsGrid.innerHTML = formats
      .map(function (format) {
        const active = activeFormat === format.id;
        return '<button data-format="' + format.id + '" class="text-left bg-white border rounded-xl p-4 transition-colors ' +
          (active ? "border-[#2D6A4F]" : "border-[#E2E2E2] hover:border-[#2D6A4F]") +
          '"><div class="text-[24px] mb-2">' + format.icon + '</div><p class="font-medium text-[#1A1A1A] text-[14px]">' +
          format.label + '</p><p class="text-[12px] text-[#6B6B6B] mt-1">' + format.desc + "</p></button>";
      })
      .join("");
    formatCardsGrid.querySelectorAll("[data-format]").forEach(function (button) {
      button.addEventListener("click", function () {
        activeFormat = button.getAttribute("data-format");
        renderFormats();
        renderSummary();
      });
    });
  }

  function renderPdfOptions() {
    pdfCheckboxesGrid.innerHTML = [
      "Include cover page",
      "Table of contents",
      "Key concepts appendix",
      "Page numbers",
    ].map(function (label) {
      return '<label class="flex items-center gap-3 text-[14px] text-[#1A1A1A]"><input type="checkbox" checked class="accent-[#2D6A4F]">' + label + "</label>";
    }).join("");
    fontSizeButtonGroup.innerHTML = ["Small", "Medium", "Large"]
      .map(function (label) {
        return '<button data-font-size="' + label + '" class="px-4 py-2 text-[13px] border-r border-[#E2E2E2] last:border-r-0 ' +
          (activeFontSize === label ? "bg-[#D8E8E0] text-[#2D6A4F]" : "bg-white text-[#6B6B6B]") +
          '">' + label + "</button>";
      })
      .join("");
    fontSizeButtonGroup.querySelectorAll("[data-font-size]").forEach(function (button) {
      button.addEventListener("click", function () {
        activeFontSize = button.getAttribute("data-font-size");
        renderPdfOptions();
        renderSummary();
      });
    });
    customHeaderField.value = (pageData.settings && pageData.settings.export_header) || "";
  }

  function renderSummary() {
    const selected = selectedItems();
    selectedCountDisplay.textContent = selected.length + " selected";
    summarySelectedCount.textContent = selected.length;
    summaryFormatLabel.textContent = formats.find(function (format) { return format.id === activeFormat; }).label;
    summaryFontLabel.textContent = activeFontSize;
    liveMetricsDisplay.textContent = Math.max(selected.length, 1) + " item(s) · Ready to export";
  }

  function activeSelection() {
    const selected = selectedItems()[0];
    return selected || pageData.items[0];
  }

  downloadExportBtn.addEventListener("click", function () {
    const item = activeSelection();
    if (!item) return;
    const contentType = item.content_type || (item.type === "Quiz" ? "quiz" : "note");
    window.NudgeApp.download("/export/" + activeFormat + "/" + item.id + "?content_type=" + contentType);
  });
  selectAllBtn.addEventListener("click", function () {
    pageData.items.forEach(function (item) { selectedSet.add(item.selection_key || ((item.content_type || "note") + ":" + item.id)); });
    renderItems();
    renderSummary();
  });

  renderItems();
  renderFormats();
  renderPdfOptions();
  renderSummary();
  window.NudgeApp.createIconSet();
})();
