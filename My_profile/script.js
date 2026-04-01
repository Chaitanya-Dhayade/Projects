// ─── FEATURE 1: Time-based greeting ───────────────────────────

// Get the current hour (0-23) from the user's computer clock
const hour = new Date().getHours();

// Decide which greeting to use based on the hour
let greeting;
if (hour < 12) {
  greeting = "Good morning";
} else if (hour < 18) {
  greeting = "Good afternoon";
} else {
  greeting = "Good evening";
}

// Find the element with id="greeting-text" and change its text
document.getElementById("greeting-text").textContent =
  greeting + " — welcome to my profile!";


// ─── FEATURE 2: Skills progress tracker ───────────────────────

// This object stores the current value of each skill
const skills = {
  html: 40,
  css: 30,
  python: 20
};

// This function runs when you click "+ I practised today"
function increaseSkill(skillName) {
  // Increase the skill value by 5, but never go above 100
  if (skills[skillName] < 100) {
    skills[skillName] += 5;
  }

  // Get the new value
  const newValue = skills[skillName];

  // Update the percentage text on screen
  document.getElementById(skillName + "-val").textContent = newValue + "%";

  // Update the width of the coloured progress bar
  document.getElementById(skillName + "-fill").style.width = newValue + "%";
}


// ─── FEATURE 3: Dark / light mode toggle ──────────────────────

// Track whether dark mode is currently on
let darkMode = false;

function toggleTheme() {
  darkMode = !darkMode;

  if (darkMode) {
    // Add the "dark" class to <body> — CSS handles the rest
    document.body.classList.add("dark");
    document.getElementById("theme-btn").textContent = "Switch to light mode";
  } else {
    // Remove the "dark" class
    document.body.classList.remove("dark");
    document.getElementById("theme-btn").textContent = "Switch to dark mode";
  }
}