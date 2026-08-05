const welcomeBanner = document.querySelector("#welcomeBanner");
const dismissWelcome = document.querySelector("#dismissWelcome");

if (welcomeBanner && !welcomeBanner.open) {
  if (typeof welcomeBanner.showModal === "function") {
    welcomeBanner.showModal();
  } else {
    welcomeBanner.setAttribute("open", "");
  }
}

dismissWelcome?.addEventListener("click", () => {
  if (typeof welcomeBanner.close === "function") {
    welcomeBanner.close();
  } else {
    welcomeBanner.removeAttribute("open");
  }
});
