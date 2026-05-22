(function () {
  const pageData = window.PAGE_DATA || {};
  const noteId = pageData.note_id;
  const processingMessage = document.getElementById("processingMessage");
  const processingSubtext = document.getElementById("processingSubtext");
  const stepNodes = document.querySelectorAll(".step-node");
  const fluidBlob = document.getElementById("fluidBlob");
  const successCheckWrapper = document.getElementById("successCheckWrapper");
  if (!processingMessage || !processingSubtext) return;

  const steps = [
    { message: "Reading your content...", sub: "Scanning through your material carefully." },
    { message: "Finding the key ideas...", sub: "Identifying important concepts and themes." },
    { message: "Organizing your notes...", sub: "Structuring notes, flashcards, and quiz data." },
    { message: "Finalizing everything...", sub: "Saving your study pack and preparing your results." },
  ];

  let currentStep = 0;
  let carouselInterval = null;
  let pollAttempts = 0;
  let stopped = false;

  function transitionCopy(message, sub) {
    processingMessage.classList.remove("text-active-state", "text-enter-state");
    processingSubtext.classList.remove("text-active-state", "text-enter-state");
    processingMessage.classList.add("text-exit-state");
    processingSubtext.classList.add("text-exit-state");

    window.setTimeout(function () {
      processingMessage.textContent = message;
      processingSubtext.textContent = sub;
      processingMessage.classList.remove("text-exit-state");
      processingSubtext.classList.remove("text-exit-state");
      processingMessage.classList.add("text-active-state");
      processingSubtext.classList.add("text-active-state");
    }, 180);
  }

  function syncStep(index) {
    transitionCopy(steps[index].message, steps[index].sub);
    stepNodes.forEach(function (node, idx) {
      const ring = node.querySelector(".step-indicator-ring");
      const icon = node.querySelector("i");
      const label = node.querySelector(".step-label");
      const active = idx <= index;
      ring.className = active
        ? "step-indicator-ring w-[26px] h-[26px] rounded-full flex items-center justify-center border-2 bg-[#2D6A4F] border-[#2D6A4F] transition-all duration-500"
        : "step-indicator-ring w-[26px] h-[26px] rounded-full flex items-center justify-center border-2 bg-[#F4F4F2] border-[#E2E2E2] transition-all duration-500";
      icon.classList.toggle("hidden", !active);
      label.className = idx === index
        ? "step-label font-mono text-[10px] uppercase tracking-[0.03em] whitespace-nowrap text-[#1A1A1A] font-medium"
        : "step-label font-mono text-[10px] uppercase tracking-[0.03em] whitespace-nowrap text-[#6B6B6B]";
    });
  }

  function finish() {
    if (stopped) return;
    stopped = true;
    window.clearInterval(carouselInterval);
    fluidBlob.style.animation = "none";
    fluidBlob.className = "w-[100px] h-[100px] bg-[#2D6A4F] rounded-full scale-100 transition-all duration-500";
    successCheckWrapper.className = "absolute inset-0 flex items-center justify-center scale-100 opacity-100 transition-all duration-500 ease-out";
    transitionCopy("Your study pack is ready.", "Opening your generated notes now.");
    window.setTimeout(function () {
      window.NudgeApp.navigate("/notes/" + noteId);
    }, 900);
  }

  function setFailure(message) {
    if (stopped) return;
    stopped = true;
    window.clearInterval(carouselInterval);
    fluidBlob.style.animation = "none";
    transitionCopy("We couldn't finish that upload.", message || "Please head back and try again.");
  }

  async function poll() {
    if (stopped) return;

    if (!noteId) {
      setFailure("No upload was attached to this processing session.");
      return;
    }

    pollAttempts += 1;
    if (pollAttempts > 120) {
      setFailure("Processing took longer than expected. Please check your notes list.");
      return;
    }

    try {
      const status = await window.NudgeApp.jsonFetch("/api/processing/" + noteId);
      if (status.status === "ready") {
        finish();
        return;
      }
      if (status.status === "failed") {
        setFailure(status.error_message);
        return;
      }
    } catch (error) {
      setFailure(error.message);
      return;
    }

    window.setTimeout(poll, 1500);
  }

  function rotate() {
    carouselInterval = window.setInterval(function () {
      if (stopped) return;
      currentStep = Math.min(currentStep + 1, steps.length - 1);
      if (currentStep === steps.length - 1) {
        window.clearInterval(carouselInterval);
      }
      syncStep(currentStep);
    }, 1800);
  }

  window.handleCancelNavigation = function () {
    window.NudgeApp.navigate("/upload");
  };

  processingMessage.classList.remove("text-enter-state");
  processingSubtext.classList.remove("text-enter-state");
  syncStep(currentStep);
  rotate();
  poll();
  window.NudgeApp.createIconSet();
})();
