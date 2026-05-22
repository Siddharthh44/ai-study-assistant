(function () {
  if (!document.getElementById("tabsPillsContainer")) return;

  const tabs = [
    { id: "text", label: "Text", icon: "file-text" },
    { id: "file", label: "PDF / Word", icon: "file-up" },
    { id: "image", label: "Image", icon: "image" },
    { id: "audio", label: "Audio", icon: "mic" },
    { id: "pyq", label: "PYQ", icon: "book-open" },
  ];

  const processingOptions = [
    { id: "summary", label: "Generate Summary" },
    { id: "notes", label: "Create Structured Notes" },
    { id: "flashcards", label: "Generate Flashcards" },
    { id: "quiz", label: "Create Quiz" },
  ];

  const state = {
    activeTab: "text",
    textValue: "",
    uploadedFile: null,
    isRecording: false,
    recordingTime: 0,
    checkedOptions: {
      summary: true,
      notes: true,
      flashcards: true,
      quiz: true,
    },
  };

  let timerIntervalRef = null;

  const tabsPillsContainer = document.getElementById("tabsPillsContainer");
  const textareaInput = document.getElementById("textareaInput");
  const charCounter = document.getElementById("charCounter");
  const hiddenFileInput = document.getElementById("hiddenFileInput");
  const fileUploadWrapper = document.getElementById("fileUploadWrapper");
  const processingOptionsCheckboxGrid = document.getElementById("processingOptionsCheckboxGrid");
  const micActionButton = document.getElementById("micActionButton");
  const recordingPulseRing = document.getElementById("recordingPulseRing");
  const micStatusTextWrapper = document.getElementById("micStatusTextWrapper");
  const submitButton = document.querySelector("button[onclick='executeAiFormSubmission()']");
  const statusLine = document.getElementById("uploadStatusText");

  function setStatus(message, isError) {
    if (!statusLine) return;
    statusLine.textContent = message;
    statusLine.className = isError
      ? "mt-3 text-[13px] text-[#C0392B] text-center"
      : "mt-3 text-[13px] text-[#6B6B6B] text-center";
  }

  function bootstrapUploadWorkspace() {
    renderTabNavigationPills();
    renderProcessingOptionsCheckboxes();
    synchronizeFileUploadBoxView();

    textareaInput.addEventListener("input", function (event) {
      state.textValue = event.target.value;
      charCounter.textContent = state.textValue.length + " chars";
    });

    hiddenFileInput.addEventListener("change", function (event) {
      const file = event.target.files[0];
      if (file) {
        state.uploadedFile = file;
        synchronizeFileUploadBoxView();
      }
    });
  }

  function renderTabNavigationPills() {
    tabsPillsContainer.innerHTML = tabs
      .map(function (tab) {
        const isActive = state.activeTab === tab.id;
        const className = isActive
          ? "bg-[#2D6A4F] text-white border-[#2D6A4F] flex items-center gap-2 px-4 py-2 rounded-full border text-[13px] font-medium transition-all focus:outline-none"
          : "bg-white text-[#1A1A1A] border-[#E2E2E2] hover:bg-[#F0F0EE] flex items-center gap-2 px-4 py-2 rounded-full border text-[13px] font-medium transition-all focus:outline-none";
        return '<button data-tab="' + tab.id + '" class="' + className + '"><i data-lucide="' + tab.icon + '" class="w-[15px] h-[15px]"></i>' + tab.label + "</button>";
      })
      .join("");

    tabsPillsContainer.querySelectorAll("[data-tab]").forEach(function (button) {
      button.addEventListener("click", function () {
        state.activeTab = button.getAttribute("data-tab");
        renderTabNavigationPills();
        document.querySelectorAll(".tab-content").forEach(function (panel) {
          panel.classList.remove("active");
        });
        const target = document.getElementById("panel-" + state.activeTab);
        if (target) target.classList.add("active");
      });
    });

    window.NudgeApp.createIconSet();
  }

  function renderProcessingOptionsCheckboxes() {
    processingOptionsCheckboxGrid.innerHTML = processingOptions
      .map(function (option) {
        const checked = state.checkedOptions[option.id];
        return (
          '<label class="flex items-center gap-3 cursor-pointer group select-none">' +
          '<input type="checkbox" class="hidden" ' +
          (checked ? "checked" : "") +
          ' data-option="' + option.id + '">' +
          '<div class="w-4 h-4 rounded flex items-center justify-center border-[1.5px] transition-colors flex-shrink-0 ' +
          (checked ? "bg-[#2D6A4F] border-[#2D6A4F]" : "bg-white border-[#E2E2E2] group-hover:border-[#2D6A4F]") +
          '">' +
          (checked ? '<i data-lucide="check" class="text-white w-[11px] h-[11px]" style="stroke-width: 3px;"></i>' : "") +
          "</div>" +
          '<span class="text-[14px] ' + (checked ? "text-[#1A1A1A] font-medium" : "text-[#6B6B6B]") + '">' +
          option.label +
          "</span></label>"
        );
      })
      .join("");

    processingOptionsCheckboxGrid.querySelectorAll("[data-option]").forEach(function (checkbox) {
      checkbox.addEventListener("change", function () {
        state.checkedOptions[checkbox.getAttribute("data-option")] = checkbox.checked;
        renderProcessingOptionsCheckboxes();
      });
    });

    window.NudgeApp.createIconSet();
  }

  function synchronizeFileUploadBoxView() {
    if (state.uploadedFile) {
      fileUploadWrapper.innerHTML =
        '<div class="flex items-center gap-4 p-4 bg-[#F4F4F2] rounded-lg border border-[#E2E2E2]">' +
        '<div class="w-10 h-10 rounded-lg bg-[#D8E8E0] flex items-center justify-center flex-shrink-0">' +
        '<i data-lucide="file-up" class="text-[#2D6A4F] w-[18px] h-[18px]"></i></div>' +
        '<span class="flex-1 text-[14px] font-medium text-[#1A1A1A] truncate">' + window.NudgeApp.escapeHtml(state.uploadedFile.name) + "</span>" +
        '<button id="clearSelectedUpload" class="text-[#6B6B6B] hover:text-[#C0392B] transition-colors focus:outline-none"><i data-lucide="x" class="w-[18px] h-[18px]"></i></button>' +
        "</div>";
      document.getElementById("clearSelectedUpload").addEventListener("click", function (event) {
        event.stopPropagation();
        state.uploadedFile = null;
        hiddenFileInput.value = "";
        synchronizeFileUploadBoxView();
      });
    } else {
      fileUploadWrapper.innerHTML =
        '<div id="fileDropZone" class="flex flex-col items-center justify-center border-2 border-dashed border-[#E2E2E2] rounded-xl bg-[#F4F4F2] p-12 cursor-pointer hover:bg-[#F0F0EE] hover:border-[#2D6A4F] transition-all min-h-[220px]">' +
        '<div class="w-16 h-16 rounded-full bg-[#D8E8E0] flex items-center justify-center mb-4">' +
        '<i data-lucide="file-up" class="text-[#2D6A4F] w-7 h-7"></i></div>' +
        '<p class="text-[15px] font-medium text-[#1A1A1A] mb-1">Drag and drop your file here</p>' +
        '<p class="text-[14px] text-[#6B6B6B]">or <span class="text-[#2D6A4F] font-semibold hover:underline">browse files</span></p>' +
        '<p class="font-mono text-[11px] text-[#6B6B6B] mt-3 tracking-[0.03em]">PDF, DOCX, TXT · Up to 10MB</p></div>';
      const dropZone = document.getElementById("fileDropZone");
      dropZone.addEventListener("click", function () {
        hiddenFileInput.click();
      });
      dropZone.addEventListener("dragover", function (event) {
        event.preventDefault();
      });
      dropZone.addEventListener("drop", function (event) {
        event.preventDefault();
        const file = event.dataTransfer.files && event.dataTransfer.files[0];
        if (file) {
          state.uploadedFile = file;
          synchronizeFileUploadBoxView();
        }
      });
    }

    window.NudgeApp.createIconSet();
  }

  function formatTime(totalSeconds) {
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
    const seconds = String(totalSeconds % 60).padStart(2, "0");
    return minutes + ":" + seconds;
  }

  window.handleMicRecordingCycle = function () {
    state.isRecording = false;
    window.clearInterval(timerIntervalRef);
    recordingPulseRing.classList.add("hidden");
    micActionButton.className = "w-20 h-20 rounded-full flex items-center justify-center transition-all bg-[#2D6A4F] hover:bg-[#245C43] focus:outline-none";
    micStatusTextWrapper.innerHTML = '<p class="text-[14px] text-[#6B6B6B]">Audio capture is not available in this deployment. Please upload TXT, PDF, DOCX, or paste text instead.</p>';
    setStatus("Audio upload is not supported in this deployment yet.", true);
  };

  async function submit() {
    const subject = document.getElementById("subjectInput").value.trim();
    const tags = document.getElementById("tagsInput").value.trim();
    const defaultHtml = submitButton.innerHTML;
    const formData = new FormData();
    formData.append("source_type", state.activeTab === "text" ? "text" : "file");
    formData.append("subject", subject);
    formData.append("tags", tags);

    if (state.activeTab === "text") {
      if (!textareaInput.value.trim()) {
        setStatus("Paste some study material before processing.", true);
        return;
      }
      formData.append("text", textareaInput.value.trim());
    } else {
      if (state.activeTab === "audio") {
        setStatus("Audio upload is not supported in this deployment yet. Please use text, PDF, DOCX, or TXT.", true);
        return;
      }
      if (!state.uploadedFile) {
        setStatus("Attach a supported file before processing.", true);
        return;
      }
      formData.append("file", state.uploadedFile);
    }

    window.NudgeApp.toggleButtonState(submitButton, true, "Processing with AI…", defaultHtml);
    setStatus("Uploading and starting the AI pipeline...", false);

    try {
      const payload = await window.NudgeApp.jsonFetch("/process", {
        method: "POST",
        body: formData,
      });
      window.NudgeApp.navigate(payload.redirect_url);
    } catch (error) {
      setStatus(error.message, true);
      window.NudgeApp.toggleButtonState(submitButton, false, null, defaultHtml);
    }
  }

  window.simulateFileBrowsing = function () {
    hiddenFileInput.click();
  };

  window.executeAiFormSubmission = submit;

  bootstrapUploadWorkspace();
  setStatus("This usually takes 10–30 seconds", false);
})();
