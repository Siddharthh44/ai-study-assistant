(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData || !pageData.cards) return;

  const cards = pageData.cards.slice();
  const flashcardInner = document.getElementById("flashcardInner");
  const questionDisplayFront = document.getElementById("questionDisplayFront");
  const questionDisplayBack = document.getElementById("questionDisplayBack");
  const answerDisplay = document.getElementById("answerDisplay");
  const progressBar = document.getElementById("progressBar");
  const cardProgressText = document.getElementById("cardProgressText");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const countGot = document.getElementById("countGot");
  const countAlmost = document.getElementById("countAlmost");
  const countStill = document.getElementById("countStill");
  const backLink = document.getElementById("flashcardsBackLink");

  let currentIndex = 0;
  let isFlipped = false;
  const ratings = { got: 0, almost: 0, still: 0 };

  function render() {
    const active = cards[currentIndex];
    if (!active) {
      questionDisplayFront.textContent = "No flashcards yet.";
      answerDisplay.textContent = "Generate notes first to build your deck.";
      return;
    }
    questionDisplayFront.textContent = active.question;
    questionDisplayBack.textContent = active.question;
    answerDisplay.textContent = active.answer;
    cardProgressText.textContent = "Card " + (currentIndex + 1) + " of " + cards.length;
    progressBar.style.width = ((currentIndex + 1) / Math.max(cards.length, 1)) * 100 + "%";
    prevBtn.disabled = currentIndex === 0;
    nextBtn.disabled = currentIndex === cards.length - 1;
    flashcardInner.classList.toggle("is-flipped", isFlipped);
    countGot.textContent = ratings.got;
    countAlmost.textContent = ratings.almost;
    countStill.textContent = ratings.still;
  }

  async function review(rating) {
    const active = cards[currentIndex];
    await window.NudgeApp.jsonFetch("/api/flashcards/" + active.id + "/review", {
      method: "POST",
      body: JSON.stringify({ rating: rating }),
    });
    ratings[rating] += 1;
    if (currentIndex < cards.length - 1) {
      currentIndex += 1;
      isFlipped = false;
    }
    render();
  }

  window.revealAnswer = function () {
    isFlipped = true;
    render();
  };
  window.goPrev = function () {
    currentIndex = Math.max(0, currentIndex - 1);
    isFlipped = false;
    render();
  };
  window.goNext = function () {
    currentIndex = Math.min(cards.length - 1, currentIndex + 1);
    isFlipped = false;
    render();
  };
  window.handleRate = review;
  window.handleShuffle = function () {
    cards.sort(function () {
      return Math.random() - 0.5;
    });
    currentIndex = 0;
    isFlipped = false;
    render();
  };
  window.handleRestart = function () {
    currentIndex = 0;
    isFlipped = false;
    ratings.got = 0;
    ratings.almost = 0;
    ratings.still = 0;
    render();
  };

  if (backLink) {
    backLink.addEventListener("click", function (event) {
      event.preventDefault();
      window.NudgeApp.navigate(pageData.back_link || "/notes");
    });
  }

  render();
  window.NudgeApp.createIconSet();
})();
