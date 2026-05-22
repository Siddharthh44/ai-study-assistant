(function () {
  const searchToggle = document.getElementById("topbarSearchToggle");
  const searchForm = document.getElementById("topbarSearchForm");
  const searchInput = document.getElementById("topbarSearchInput");
  const notificationToggle = document.getElementById("topbarNotificationToggle");
  const notificationPanel = document.getElementById("topbarNotificationPanel");
  const notificationClose = document.getElementById("topbarNotificationClose");

  function setSearchOpen(open) {
    if (!searchForm || !searchToggle) return;
    searchForm.classList.toggle("hidden", !open);
    searchForm.classList.toggle("flex", open);
    searchToggle.classList.toggle("hidden", open);
    if (open && searchInput) {
      window.setTimeout(function () {
        searchInput.focus();
      }, 0);
    }
  }

  function setNotificationsOpen(open) {
    if (!notificationPanel) return;
    notificationPanel.classList.toggle("hidden", !open);
  }

  if (searchToggle) {
    searchToggle.addEventListener("click", function () {
      setSearchOpen(true);
    });
  }

  if (searchInput) {
    searchInput.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        setSearchOpen(false);
      }
    });
    searchInput.addEventListener("blur", function () {
      window.setTimeout(function () {
        setSearchOpen(false);
      }, 120);
    });
  }

  if (notificationToggle) {
    notificationToggle.addEventListener("click", function (event) {
      event.stopPropagation();
      const isHidden = notificationPanel && notificationPanel.classList.contains("hidden");
      setNotificationsOpen(isHidden);
    });
  }

  if (notificationClose) {
    notificationClose.addEventListener("click", function () {
      setNotificationsOpen(false);
    });
  }

  document.addEventListener("click", function (event) {
    if (!notificationPanel || !notificationToggle) return;
    if (notificationPanel.contains(event.target) || notificationToggle.contains(event.target)) return;
    setNotificationsOpen(false);
  });

  window.NudgeApp.createIconSet();
})();
