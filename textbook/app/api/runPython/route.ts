import { NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs/promises";
import path from "path";

const execAsync = promisify(exec);

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const scriptName = formData.get("scriptName") as string;
    // This field can be used as a display name; otherwise we'll use the uploaded file's name.
    const pdfFileNameField = formData.get("pdfFileName") as string;
    const pdfFile = formData.get("pdfFile") as File;
    if (!pdfFile) {
      throw new Error("No PDF file uploaded");
    }

    // Save the uploaded PDF file in an 'uploads' folder within your project.
    const uploadsDir = path.join(process.cwd(), "uploads");
    await fs.mkdir(uploadsDir, { recursive: true });
    // Save using the original file name.
    const pdfFilePath = path.join(uploadsDir, pdfFile.name);
    const arrayBuffer = await pdfFile.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    await fs.writeFile(pdfFilePath, buffer);

    // Build the absolute path to the Python script.
    // Adjust this relative path as needed (here, we assume your Python scripts folder is outside your Next.js folder).
    const pythonFolderPath = path.join(process.cwd(), "..", "textbook2json");
    const scriptPath = path.join(pythonFolderPath, scriptName);

    // Use the provided pdfFileName field or fallback to the file's name.
    const outputFileArg = pdfFileNameField || pdfFile.name;

    // Construct the command to run the Python script, passing the local file path.
    const command = `python "${scriptPath}" "${pdfFilePath}" --output-file "${outputFileArg}"`;

    const { stdout, stderr } = await execAsync(command);

    return NextResponse.json({ success: true, stdout, stderr });
  } catch (error) {
    return NextResponse.json({ success: false, error: String(error) }, { status: 500 });
  }
}
