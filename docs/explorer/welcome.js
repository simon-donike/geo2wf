const welcomeBanner = document.querySelector("#welcomeBanner");
const welcomeDismissButtons = document.querySelectorAll("#dismissWelcome, #closeWelcome");

if (welcomeBanner && !welcomeBanner.open) {
  if (typeof welcomeBanner.showModal === "function") {
    welcomeBanner.showModal();
  } else {
    welcomeBanner.setAttribute("open", "");
  }
}

welcomeDismissButtons.forEach((button) => button.addEventListener("click", () => {
  if (typeof welcomeBanner.close === "function") {
    welcomeBanner.close();
  } else {
    welcomeBanner.removeAttribute("open");
  }
}));
