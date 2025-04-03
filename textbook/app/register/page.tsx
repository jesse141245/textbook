"use client";
import { useState, useRef, DragEvent } from "react";
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
  const [phase, setPhase] = useState<"selectSubject" | "enterTextbook" | "questionnaire">("selectSubject");
  const [selectedSubject, setSelectedSubject] = useState<string | null>(null);

  // PDF Upload State
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Questionnaire States
  const [tocAnswer, setTocAnswer] = useState("");
  const [sourceAnswer, setSourceAnswer] = useState("");

  // ----- PHASE 1: Select Subject -----
  if (phase === "selectSubject") {
    const n = subjects.length;
    const rows = Math.ceil(n / 4);
    let used = 0;
    const gridItems = [];
    for (let r = 0; r < rows; r++) {
      gridItems.push(...subjects.slice(used, used + 4));
      used += 4;
    }
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 py-16">
        <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-4xl">
          <h1 className="text-3xl font-bold text-center mb-8">I want to learn...</h1>
          <div className="grid grid-cols-4 gap-6 place-items-center">
            {gridItems.map((item, index) => {
              if (!item) return <div key={index} className="w-48 h-48" />;
              return (
                <motion.div
                  key={index}
                  whileHover={{ scale: 1.05 }}
                  onClick={() => {
                    setSelectedSubject(item.name);
                    setPhase("enterTextbook");
                  }}
                  className="bg-gray-50 border border-gray-200 rounded-xl shadow-sm w-48 h-50 p-4 flex flex-col items-center justify-center cursor-pointer hover:shadow-md transition-shadow"
                >
                  <span className="text-6xl mb-3">{item.icon}</span>
                  <h2 className="text-lg font-semibold text-center">{item.name}</h2>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  // ----- PHASE 2: PDF Upload -----
  if (phase === "enterTextbook") {
    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
    };
    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (file.type !== "application/pdf") {
          alert("Please upload a PDF file.");
          return;
        }
        setPdfFile(file);
      }
    };
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (file.type !== "application/pdf") {
          alert("Please upload a PDF file.");
          return;
        }
        setPdfFile(file);
      }
    };

    const handleRemoveFile = (e: React.MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      setPdfFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    };

    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 py-16">
        <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-xl">
          <h1 className="text-3xl font-bold text-center mb-6">
            Please upload your {selectedSubject} textbook (PDF)
          </h1>
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="relative border-2 border-dashed border-blue-300 p-8 rounded-md text-center cursor-pointer hover:border-blue-500 transition-colors"
          >
            {pdfFile ? (
              <div className="flex items-center justify-center">
                <p className="text-xl font-medium">{pdfFile.name}</p>
                <button
                  onClick={handleRemoveFile}
                  className="ml-4 text-red-500 font-bold text-xl hover:text-red-700"
                  title="Remove file"
                >
                  ✕
                </button>
              </div>
            ) : (
              <p className="text-xl font-medium text-gray-600">
                Drag and drop a PDF file here, or click to browse
              </p>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            onClick={() => {
              if (pdfFile) {
                setPhase("questionnaire");
              } else {
                alert("Please upload a PDF file.");
              }
            }}
            className="block mx-auto mt-6 px-6 py-3 bg-blue-500 text-white text-xl rounded-md hover:bg-blue-600 transition-colors"
          >
            Submit
          </button>
        </div>
      </div>
    );
  }

  // ----- PHASE 3: Questionnaire -----
  if (phase === "questionnaire") {
    // Modified handleSubmit for fire-and-forget
    const handleSubmit = () => { // Removed async
      if (!tocAnswer || !sourceAnswer) {
        alert("Please answer both questions.");
        return;
      }
      if (!pdfFile) {
        alert("No PDF file found. Please go back and upload a file.");
        // Optionally reset phase if needed: setPhase("enterTextbook");
        return;
      }

      // Decide which script to call based on TOC answer
      const scriptName = tocAnswer === "Yes" ? "gemeni_json_toc.py" : "gemeni_jsonv2.py";

      // Create FormData with the script name, pdf file name, and the file itself
      const formData = new FormData();
      formData.append("scriptName", scriptName);
      formData.append("pdfFileName", pdfFile.name); // Send original filename
      formData.append("pdfFile", pdfFile);          // Send the actual file blob
      // Add questionnaire answers if needed by the backend script
      formData.append("hasToc", tocAnswer);
      formData.append("source", sourceAnswer);

      try {
        // --- Initiate the fetch request (don't await) ---
        const fetchPromise = fetch("/api/runPython", {
          method: "POST",
          body: formData,
        });

        // --- Navigate immediately ---
        console.log("Navigating to /welcome...");
        router.push("/welcome");

        // --- Optional: Handle the promise result later (for logging/background feedback) ---
        fetchPromise
          .then(async (res) => {
            // This code runs *after* navigation has likely occurred
            if (res.ok) {
              const data = await res.json(); // Try to parse immediate response
              console.log("Python script initiation successful (API Response):", data);
              // Server should ideally confirm process *started*, not finished
              if (!data.success) {
                 console.warn("API indicated potential issue starting script:", data.error || "Unknown issue");
              }
            } else {
              // Error from the API endpoint itself (e.g., 500 internal server error, bad request)
              const errorText = await res.text();
              console.error(`Error response from /api/runPython (${res.status}) after navigation: ${errorText}`);
              // Cannot alert user easily here. Log appropriately.
            }
          })
          .catch((err) => {
            // Catch network errors or issues *during* the fetch itself (after navigation)
            console.error("Network or fetch error calling /api/runPython (after navigation):", err);
             // Cannot alert user easily here.
          });

      } catch (error) {
        // Catch errors *during* FormData creation or synchronous fetch setup
        console.error("Error setting up the API call:", error);
        alert("Failed to start the processing. Please check console for details.");
        // Do not navigate if the request couldn't even be initiated
      }
    };

    // Rest of the questionnaire JSX remains the same...
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 py-16">
        <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-xl">
          <h1 className="text-3xl font-bold text-center mb-8">
            Welcome! Please answer the following:
          </h1>
          {/* Question 1: TOC */}
          <div className="mb-6">
            <div className="mb-2 text-xl font-semibold">
              Does the textbook have a Table of Contents (TOC)?
            </div>
            <div className="flex flex-col gap-2 ml-2">
              {["Yes", "No"].map((option, idx) => (
                <label key={idx} className="flex items-center space-x-2">
                  <input
                    type="radio"
                    name="toc"
                    value={option}
                    checked={tocAnswer === option}
                    onChange={(e) => setTocAnswer(e.target.value)}
                    className="w-5 h-5"
                  />
                  <span className="text-lg">{option}</span>
                </label>
              ))}
            </div>
          </div>
          {/* Question 2: Source */}
          <div className="mb-6">
            <div className="mb-2 text-xl font-semibold">
              Where is the textbook from?
            </div>
            <div className="flex flex-col gap-2 ml-2">
              {["LibTexts", "Other"].map((option, idx) => (
                <label key={idx} className="flex items-center space-x-2">
                  <input
                    type="radio"
                    name="source"
                    value={option}
                    checked={sourceAnswer === option}
                    onChange={(e) => setSourceAnswer(e.target.value)}
                    className="w-5 h-5"
                  />
                  <span className="text-lg">{option}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="flex justify-center">
            <button
              onClick={handleSubmit} // Use the modified handler
              className="px-6 py-3 bg-blue-500 text-white text-xl rounded-md hover:bg-blue-600 transition-colors"
            >
              Submit & Proceed
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
}