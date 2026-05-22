(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData || !pageData.quiz) return;

  const quiz = pageData.quiz;
  const questions = quiz.questions || [];
  const questionCounter = document.getElementById("questionCounter");
  const timerBadge = document.getElementById("timerBadge");
  const timerText = document.getElementById("timerText");
  const progressBarFill = document.getElementById("progressBarFill");
  const questionNumberPrefix = document.getElementById("questionNumberPrefix");
  const questionTextDisplay = document.getElementById("questionTextDisplay");
  const optionsContainer = document.getElementById("optionsContainer");
  const explanationContainer = document.getElementById("explanationContainer");
  const prevButton = document.getElementById("prevButton");
  const actionButtonWrapper = document.getElementById("actionButtonWrapper");
  const quizTitle = document.getElementById("quizTitle");
  const quizSubject = document.getElementById("quizSubject");

  let currentQuestionIndex = 0;
  let selectedOptionIndex = null;
  let isQuestionAnswered = false;
  let countdownTimeSeconds = quiz.duration_seconds || Math.max(questions.length, 1) * 60;
  const savedAnswersMap = {};
  const draftSelectionsMap = {};
  const startedAt = Date.now();
  let countdownTimerHandle = null;
  let quizSubmitted = false;

  quizTitle.textContent = pageData.title;
  quizSubject.textContent = pageData.subject + " Quiz";

  function formatAndUpdateTimerUI() {
    const minutes = String(Math.floor(countdownTimeSeconds / 60)).padStart(2, "0");
    const seconds = String(countdownTimeSeconds % 60).padStart(2, "0");
    timerText.textContent = minutes + ":" + seconds;
    timerBadge.className = countdownTimeSeconds < 120
      ? "flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-50 text-[#C0392B]"
      : "flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#D8E8E0] text-[#2D6A4F]";
  }

  function startCountdownTimer() {
    if (countdownTimerHandle) return;
    formatAndUpdateTimerUI();
    countdownTimerHandle = window.setInterval(function () {
      if (quizSubmitted) return;
      countdownTimeSeconds = Math.max(0, countdownTimeSeconds - 1);
      formatAndUpdateTimerUI();
      if (countdownTimeSeconds === 0) {
        submitQuiz();
      }
    }, 1000);
  }

  function renderQuestion() {
    const activeQuestion = questions[currentQuestionIndex];
    isQuestionAnswered = Object.prototype.hasOwnProperty.call(savedAnswersMap, currentQuestionIndex);
    if (isQuestionAnswered) {
      selectedOptionIndex = savedAnswersMap[currentQuestionIndex];
    } else if (Object.prototype.hasOwnProperty.call(draftSelectionsMap, currentQuestionIndex)) {
      selectedOptionIndex = draftSelectionsMap[currentQuestionIndex];
    } else {
      selectedOptionIndex = null;
    }

    questionCounter.textContent = "Question " + (currentQuestionIndex + 1) + " of " + questions.length;
    questionNumberPrefix.textContent = "Q" + (currentQuestionIndex + 1) + ".";
    questionTextDisplay.textContent = activeQuestion.question;
    progressBarFill.style.width = ((currentQuestionIndex + 1) / Math.max(questions.length, 1)) * 100 + "%";
    prevButton.disabled = currentQuestionIndex === 0;

    optionsContainer.innerHTML = activeQuestion.options
      .map(function (optionText, idx) {
        const letter = String.fromCharCode(65 + idx);
        let containerStyles = "border-[#E2E2E2] bg-white hover:border-[#2D6A4F] hover:bg-[#D8E8E0]/10";
        let badgeStyles = "bg-white text-[#6B6B6B] border-[#E2E2E2]";
        let iconMarkup = "";

        if (!isQuestionAnswered && selectedOptionIndex === idx) {
          containerStyles = "border-[#2D6A4F] border-l-[4px] bg-[#D8E8E0]/30";
          badgeStyles = "bg-[#2D6A4F] text-white border-[#2D6A4F]";
        } else if (isQuestionAnswered) {
          if (idx === activeQuestion.correct_index) {
            containerStyles = "border-[#52796F] border-l-[4px] bg-[#D8E8E0]/30";
            badgeStyles = "bg-[#52796F] text-white border-[#52796F]";
            iconMarkup = '<i data-lucide="check-circle" class="text-[#52796F] w-[18px] h-[18px]"></i>';
          } else if (selectedOptionIndex === idx) {
            containerStyles = "border-[#C0392B] border-l-[4px] bg-red-50";
            badgeStyles = "bg-[#C0392B] text-white border-[#C0392B]";
            iconMarkup = '<i data-lucide="x-circle" class="text-[#C0392B] w-[18px] h-[18px]"></i>';
          }
        }

        return (
          '<button type="button" data-option-index="' + idx + '" class="w-full text-left p-4 rounded-xl border transition-all pointer-events-auto ' + containerStyles + '">' +
          '<div class="flex items-start gap-4"><div class="w-8 h-8 rounded-full border flex items-center justify-center font-mono text-[12px] flex-shrink-0 ' + badgeStyles + '">' + letter + "</div>" +
          '<div class="flex-1 text-[15px] text-[#1A1A1A] leading-relaxed">' + window.NudgeApp.escapeHtml(optionText) + "</div>" + iconMarkup + "</div></button>"
        );
      })
      .join("");

    optionsContainer.querySelectorAll("[data-option-index]").forEach(function (button) {
      button.addEventListener("click", function () {
        if (isQuestionAnswered) return;
        selectedOptionIndex = Number(button.getAttribute("data-option-index"));
        draftSelectionsMap[currentQuestionIndex] = selectedOptionIndex;
        renderQuestion();
      });
    });

    explanationContainer.classList.toggle("hidden", !isQuestionAnswered);
    explanationContainer.innerHTML = isQuestionAnswered
      ? '<div class="bg-[#F4F4F2] rounded-xl p-4 border border-[#E2E2E2]"><p class="text-[13px] font-medium text-[#2D6A4F] mb-1">Explanation</p><p class="text-[14px] text-[#6B6B6B] leading-relaxed">' + window.NudgeApp.escapeHtml(activeQuestion.explanation || "") + "</p></div>"
      : "";

    actionButtonWrapper.innerHTML = isQuestionAnswered
      ? (currentQuestionIndex === questions.length - 1
          ? '<button type="button" id="finishQuizButton" class="bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors">Finish Quiz &rarr;</button>'
          : '<button type="button" id="nextQuestionButton" class="bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors">Next Question &rarr;</button>')
      : '<button type="button" id="submitAnswerButton" class="bg-[#2D6A4F] text-white text-[14px] font-semibold px-5 py-2.5 rounded-lg hover:bg-[#245C43] transition-colors disabled:opacity-60">Submit Answer</button>';

    const submitAnswerButton = document.getElementById("submitAnswerButton");
    if (submitAnswerButton) {
      submitAnswerButton.disabled = selectedOptionIndex === null;
      submitAnswerButton.addEventListener("click", function () {
        if (selectedOptionIndex === null) return;
        savedAnswersMap[currentQuestionIndex] = selectedOptionIndex;
        delete draftSelectionsMap[currentQuestionIndex];
        renderQuestion();
      });
    }

    const nextQuestionButton = document.getElementById("nextQuestionButton");
    if (nextQuestionButton) {
      nextQuestionButton.addEventListener("click", function () {
        currentQuestionIndex += 1;
        renderQuestion();
      });
    }

    const finishQuizButton = document.getElementById("finishQuizButton");
    if (finishQuizButton) {
      finishQuizButton.addEventListener("click", submitQuiz);
    }

    window.NudgeApp.createIconSet();
  }

  async function submitQuiz() {
    if (quizSubmitted) return;
    quizSubmitted = true;
    if (countdownTimerHandle) {
      window.clearInterval(countdownTimerHandle);
      countdownTimerHandle = null;
    }

    const answers = {};
    Object.keys(savedAnswersMap).forEach(function (questionIndex) {
      const question = questions[Number(questionIndex)];
      answers[question.id] = savedAnswersMap[questionIndex];
    });

    try {
      const response = await window.NudgeApp.jsonFetch("/api/quizzes/" + quiz.id + "/attempts", {
        method: "POST",
        body: JSON.stringify({
          answers: answers,
          completion_seconds: Math.round((Date.now() - startedAt) / 1000),
        }),
      });
      window.NudgeApp.navigate("/quiz-results?attempt_id=" + response.attempt_id);
    } catch (_error) {
      quizSubmitted = false;
      startCountdownTimer();
      actionButtonWrapper.innerHTML = '<p class="text-[13px] text-[#C0392B]">Unable to submit quiz. Please try again.</p>';
    }
  }

  window.navigateToPreviousQuestion = function () {
    currentQuestionIndex = Math.max(0, currentQuestionIndex - 1);
    renderQuestion();
  };

  renderQuestion();
  startCountdownTimer();
  window.NudgeApp.createIconSet();
})();
