"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";

const subjects = [
  { name: "Physics", learners: "10M learners", icon: "🔬" },
  { name: "Mathematics", learners: "15M learners", icon: "📐" },
  { name: "Computer Science", learners: "12M learners", icon: "💻" },
  { name: "Biology", learners: "8M learners", icon: "🧬" },
  { name: "Chemistry", learners: "9M learners", icon: "⚗️" },
  { name: "Engineering", learners: "7M learners", icon: "🏗️" },
  { name: "Astronomy", learners: "5M learners", icon: "🌌" },
  { name: "Data Science", learners: "11M learners", icon: "📊" },
  { name: "Economics", learners: "6M learners", icon: "💰" },
];

export default function StudyApp() {
  const router = useRouter();
  const [selectedSubject, setSelectedSubject] = useState<string | null>(null);
  const [textbook, setTextbook] = useState("");

  // If a subject is selected, show the textbook input
  if (selectedSubject) {
    return (
      <div className="flex flex-col items-center justify-center h-screen py-16">
        <h1 className="text-4xl font-bold text-center mb-6">
          Please input {selectedSubject} textbook
        </h1>
        <input
          type="text"
          value={textbook}
          onChange={(e) => setTextbook(e.target.value)}
          placeholder="Enter textbook URL or title..."
          className="p-4 border rounded-md w-[32rem] text-center mb-6 text-2xl"
        />
        <button
          onClick={() => { // Removed async here, as we don't await fetch for navigation
            try {
              const formData = new FormData();
              // Ensure this script name matches what your API expects
              formData.append("scriptName", "your_python_script.py");
              // Send the textbook info
              formData.append("pdfFileName", textbook); // Or use a different key if your API expects it

              // --- Initiate the fetch request ---
              const fetchPromise = fetch("/api/runPython", {
                method: "POST",
                body: formData,
              });

              // --- Navigate immediately ---
              router.push("/welcome");

              // --- Optional: Handle the promise result later (for logging) ---
              fetchPromise.then(async (res) => {
                  if (res.ok) {
                    // The server acknowledged the request start.
                    // Note: This doesn't mean the Python script finished,
                    // only that the API endpoint received the request and *likely* started the script.
                    console.log("Python script execution initiated successfully.");
                    // You could potentially try to read the response if your API sends one back immediately
                    // const data = await res.json();
                    // console.log("API immediate response:", data);
                  } else {
                    // The API endpoint itself had an error *before* or *during* initiating the script
                    const errorText = await res.text();
                    console.error(`Error response from /api/runPython (${res.status}): ${errorText}`);
                    // Cannot easily alert the user here as they've navigated away.
                    // Consider a more robust notification system if feedback is critical.
                  }
                })
                .catch((err) => {
                  // Catch network errors or issues *during* the fetch itself (after navigation)
                  console.error("Error during fetch call to /api/runPython (after navigation):", err);
                   // Cannot easily alert the user here.
                });

            } catch (err) {
              // Catch errors *only* during FormData creation or the synchronous setup of fetch
              console.error("Error setting up API call:", err);
              alert("Failed to initiate the process. Please check your input or network connection.");
              // Don't navigate if we couldn't even start the request
            }
          }}
          className="px-6 py-4 bg-blue-500 text-white text-2xl rounded-md hover:bg-blue-600"
        >
          Submit
        </button>
      </div>
    );
  }

  // Build a 4×4 style layout
  const n = subjects.length;
  // Number of rows needed
  const rows = Math.ceil(n / 4);
  let used = 0;
  const gridItems = [];

  for (let r = 0; r < rows; r++) {
    gridItems.push(...subjects.slice(used, used + 4));
    used += 4;
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 py-16">
      <h1 className="text-5xl font-bold text-center mb-10">
        I want to learn...
      </h1>
      <div className="grid grid-cols-4 gap-4 place-items-center">
        {gridItems.map((item, index) => {
          if (!item) {
            return <div key={index} className="w-48 h-48" />;
          }
          return (
            <motion.div
              key={index}
              whileHover={{ scale: 1.05 }}
              onClick={() => setSelectedSubject(item.name)}
              className="bg-white border border-gray-300 rounded-xl
                         shadow-md w-48 h-50 p-4
                         flex flex-col items-center justify-center
                         cursor-pointer hover:shadow-lg transition-shadow"
            >
              <span className="text-6xl mb-3">{item.icon}</span>
              <h2 className="text-xl font-semibold text-center">{item.name}</h2>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
