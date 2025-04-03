// app/api/training-progress/route.ts
import { NextResponse } from 'next/server'; // Import NextResponse
import fs from 'fs';
import path from 'path';

// Use GET, POST, etc. exported functions
export async function GET(request: Request) {
  const progressFile = path.join(process.cwd(), 'progress.json');
  console.log("Looking for progress file at:", progressFile);

  if (fs.existsSync(progressFile)) {
    const data = fs.readFileSync(progressFile, 'utf8');
    console.log("Progress file content:", data);
    try {
      const progress = JSON.parse(data);
      console.log("Parsed progress:", progress);
      // Use NextResponse.json for responses
      return NextResponse.json(progress);
    } catch (e) {
      console.error("Error parsing progress file:", e);
      // Use NextResponse.json for errors, setting the status
      return NextResponse.json(
        { error: 'Error parsing progress file.' },
        { status: 500 }
      );
    }
  } else {
    console.log("Progress file not found.");
    // Use NextResponse.json for errors, setting the status
    return NextResponse.json(
      { error: 'Progress file not found.' },
      { status: 404 } // Send a 404 if the *data* file isn't found
                     // Note: This is different from the router 404 you were getting before
    );
  }
}