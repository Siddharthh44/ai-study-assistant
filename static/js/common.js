(function () {
  function jsonFetch(url, options) {
    return fetch(url, {
      credentials: "same-origin",
      headers: {
        Accept: "application/json",
        ...(options && options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...(options && options.headers ? options.headers : {}),
      },
      ...options,
    }).then(async (response) => {
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json")
        ? await response.json()
        : await response.text();

      if (response.status === 401) {
        window.location.href = "/login";
        throw new Error("Your session expired. Please sign in again.");
      }

      if (!response.ok) {
        const detail = payload && payload.detail ? payload.detail : "Something went wrong.";
        throw new Error(detail);
      }

      return payload;
    });
  }

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function setText(id, value) {
    const node = document.getElementById(id);
    if (node) {
      node.textContent = value;
    }
  }

  function navigate(url) {
    window.location.href = url;
  }

  function download(url) {
    window.location.href = url;
  }

  function toggleButtonState(button, busy, busyLabel, defaultLabel) {
    if (!button) return;
    button.disabled = busy;
    if (busyLabel || defaultLabel) {
      button.innerHTML = busy ? busyLabel : defaultLabel;
    }
  }

  function createIconSet() {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  }

  window.NudgeApp = {
    jsonFetch,
    escapeHtml,
    setText,
    navigate,
    download,
    toggleButtonState,
    createIconSet,
  };
})();
