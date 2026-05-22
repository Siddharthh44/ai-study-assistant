(function () {
  const pageData = window.PAGE_DATA;
  if (!pageData || !pageData.stats) return;

  window.NudgeApp.setText("quizzesTakenValue", String(pageData.stats.quizzes_taken || 0));
  window.NudgeApp.setText("averageScoreValue", (pageData.stats.average_score || 0) + "%");
  window.NudgeApp.setText("cardsReviewedValue", String(pageData.stats.cards_reviewed || 0));
  window.NudgeApp.setText("studyStreakProgressValue", (pageData.stats.study_streak || 0) + " days");

  const topicProgressWrapper = document.getElementById("topicProgressWrapper");
  const focusPillsContainer = document.getElementById("focusPillsContainer");
  const quizHistoryWrapper = document.getElementById("quizHistoryWrapper");

  topicProgressWrapper.innerHTML = (pageData.topic_performance || [])
    .map(function (topic) {
      return '<div class="flex items-center gap-4"><span class="w-20 text-[13px] font-medium text-[#1A1A1A] flex-shrink-0">' +
        window.NudgeApp.escapeHtml(topic.name) + '</span><div class="flex-1 h-2 bg-[#F0F0EE] rounded-full overflow-hidden"><div class="h-full rounded-full transition-all duration-700" style="width:' +
        topic.score + "%;background-color:" + topic.fill + ';"></div></div><span class="font-mono text-[11px] text-[#6B6B6B] w-10 text-right tracking-[0.03em]">' +
        topic.score + "%</span></div>";
    })
    .join("");

  focusPillsContainer.innerHTML = (pageData.focus_areas || [])
    .map(function (topic) {
      return '<span class="bg-[#F4F4F2] border border-[#E2E2E2] px-4 py-2 rounded-full text-[13px] text-[#1A1A1A]">' +
        window.NudgeApp.escapeHtml(topic.name) + " · " + topic.score + "%</span>";
    })
    .join("");

  quizHistoryWrapper.innerHTML = (pageData.recent_quiz_history || [])
    .map(function (attempt) {
      return '<div class="px-6 py-4 flex items-center justify-between"><div><p class="font-medium text-[#1A1A1A]">' +
        window.NudgeApp.escapeHtml(attempt.title) + '</p><p class="text-[13px] text-[#6B6B6B]">' +
        attempt.completion_seconds + 's</p></div><span class="font-mono text-[11px] font-medium bg-[#D8E8E0] text-[#2D6A4F] px-2 py-0.5 rounded-full">' +
        attempt.percent + "%</span></div>";
    })
    .join("");

  if (window.Chart) {
    const canvas = document.getElementById("scoreTrendChart");
    new window.Chart(canvas, {
      type: "line",
      data: {
        labels: (pageData.score_trend || []).map(function (item) { return item.name; }),
        datasets: [
          {
            label: "Score",
            data: (pageData.score_trend || []).map(function (item) { return item.score; }),
            borderColor: "#2D6A4F",
            tension: 0.35,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { suggestedMin: 0, suggestedMax: 100 },
        },
      },
    });
  }

  window.NudgeApp.createIconSet();
})();
