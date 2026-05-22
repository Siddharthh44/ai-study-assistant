(function () {
  const pageData = window.PAGE_DATA || { initial_mode: "login" };
  const authForm = document.getElementById("authForm");
  if (!authForm) return;

  const formTitle = document.getElementById("formTitle");
  const formSubtitle = document.getElementById("formSubtitle");
  const nameFieldContainer = document.getElementById("nameFieldContainer");
  const fullNameInput = document.getElementById("fullName");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");
  const confirmPasswordFieldContainer = document.getElementById("confirmPasswordFieldContainer");
  const confirmPasswordInput = document.getElementById("confirmPassword");
  const forgotPasswordContainer = document.getElementById("forgotPasswordContainer");
  const submitBtn = document.getElementById("submitBtn");
  const toggleAuthMode = document.getElementById("toggleAuthMode");
  const togglePromptText = document.getElementById("togglePromptText");
  const submitBtnText = document.getElementById("submitBtnText");
  const togglePasswordBtn = document.getElementById("togglePassword");
  const eyeIcon = document.getElementById("eyeIcon");

  let isLogin = pageData.initial_mode !== "signup";
  let showPassword = false;

  function renderMode() {
    formTitle.textContent = isLogin ? "Welcome back" : "Create your account";
    formSubtitle.textContent = isLogin
      ? "Let's pick up where you left off."
      : "Start your learning journey today.";
    nameFieldContainer.classList.toggle("hidden", isLogin);
    confirmPasswordFieldContainer.classList.toggle("hidden", isLogin);
    forgotPasswordContainer.classList.toggle("hidden", !isLogin);
    fullNameInput.required = !isLogin;
    confirmPasswordInput.required = !isLogin;
    submitBtnText.textContent = isLogin ? "Sign in →" : "Create account →";
    togglePromptText.textContent = isLogin ? "New here? " : "Already have an account? ";
    toggleAuthMode.textContent = isLogin ? "Create a free account →" : "Sign in →";
  }

  function setStatus(message, isError) {
    formSubtitle.textContent = message;
    formSubtitle.className = isError ? "text-[15px] text-[#C0392B]" : "text-[15px] text-[#6B6B6B]";
  }

  toggleAuthMode.addEventListener("click", function () {
    isLogin = !isLogin;
    renderMode();
  });

  togglePasswordBtn.addEventListener("click", function () {
    showPassword = !showPassword;
    passwordInput.type = showPassword ? "text" : "password";
    eyeIcon.setAttribute("data-lucide", showPassword ? "eye-off" : "eye");
    window.NudgeApp.createIconSet();
  });

  authForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    if (!isLogin && confirmPasswordInput.value !== passwordInput.value) {
      setStatus("Passwords need to match before we can continue.", true);
      return;
    }

    const payload = isLogin
      ? {
          email: emailInput.value.trim(),
          password: passwordInput.value,
        }
      : {
          full_name: fullNameInput.value.trim(),
          email: emailInput.value.trim(),
          password: passwordInput.value,
        };

    const defaultLabel = submitBtn.innerHTML;
    window.NudgeApp.toggleButtonState(
      submitBtn,
      true,
      '<span class="flex items-center justify-center gap-2"><svg class="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="white" stroke-width="3" stroke-opacity="0.3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="white" stroke-width="3" stroke-linecap="round"></path></svg>Working...</span>',
      defaultLabel
    );

    try {
      const response = await window.NudgeApp.jsonFetch(isLogin ? "/api/auth/login" : "/api/auth/signup", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setStatus("Authentication successful. Redirecting...", false);
      window.NudgeApp.navigate(response.redirect_url || "/dashboard");
    } catch (error) {
      setStatus(error.message, true);
      window.NudgeApp.toggleButtonState(submitBtn, false, null, defaultLabel);
    }
  });

  renderMode();
  window.NudgeApp.createIconSet();
})();
