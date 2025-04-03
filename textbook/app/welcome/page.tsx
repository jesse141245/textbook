"use client";
import { useState, useEffect } from "react";

interface Progress {
  epoch: number;
  step: number;
  total_steps: number;
  percentage: number;
}

export default function TrainingProgress() {
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null); // Added state for better error display

  useEffect(() => {
    const fetchData = async () => {
      setError(null); // Clear previous errors on new fetch
      try {
        // --- CORRECTED URL HERE ---
        const res = await fetch(`/api/training-progress?t=${Date.now()}`);
        // --- ---

        if (res.ok) {
          const data: Progress = await res.json();
          setProgress(data);
        } else {
          const errorText = await res.text();
          console.error("Error fetching progress data:", res.status, errorText);
          setError(`Failed to load progress: ${res.status} ${errorText.substring(0, 100)}`); // Store error message
          if (res.status === 404) 
            {
            clearInterval(interval);
          }
        }
      } catch (err) {
        console.error("Network or parsing error fetching progress:", err);
        setError(`Network error: ${err instanceof Error ? err.message : String(err)}`);
        // Consider stopping polling on network errors too
        // clearInterval(interval);
      }
    };

    // Fetch immediately on mount
    fetchData();

    // Set up polling interval
    const interval = setInterval(fetchData, 2000); // poll every 2 seconds

    // Cleanup interval on component unmount
    return () => clearInterval(interval);
  }, []); // Empty dependency array ensures this runs only once on mount to set up interval

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 py-16">
      <h1 className="text-5xl font-bold text-center mb-10">Training Progress</h1>
      {error && ( // Display error message if present
        <p className="text-red-500 text-xl mb-4 bg-red-100 p-3 rounded border border-red-300">
          Error: {error}
        </p>
      )}
      {progress ? (
        <div className="w-[32rem] text-center">
          <div className="w-full bg-gray-200 rounded-full h-6 mb-4">
            <div
              className="bg-blue-500 h-6 rounded-full transition-all duration-500 ease-out" // Added ease-out for smoother transition
              style={{ width: `${progress.percentage}%` }}
            />
          </div>
          <p className="text-2xl font-semibold">{progress.percentage.toFixed(1)}% complete</p> {/* Added toFixed for better percentage display */}
        </div>
      ) : !error ? ( // Only show "No data" if there isn't an error
        <p className="text-2xl">Loading progress data or none available yet...</p>
      ) : null /* Don't show "No data" if an error is already displayed */
      }
    </div>
  );
}