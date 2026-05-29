import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
# Phase 3 will add .pdf, .docx, .pptx — for now, text files only
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".py", ".yaml", ".yml"}

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. THE TOOL ---
def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"ERROR: Could not read file — {e}"

# --- 2. DESCRIBE the tool to the model ---
read_file_declaration = {
    "name": "read_file",
    "description": "Reads a text file from local disk and returns its full contents. "
                   "Use this to see a file's contents before summarizing it.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file, e.g. 'input/notes.txt'"},
        },
        "required": ["path"],
    },
}

SYSTEM_PROMPT = """You are a meticulous technical summarizer. You act as an intelligent filter: \
you condense dense material into its absolute core essentials so that even a non-technical reader grasps the main \
takeaway at a glance — without losing crucial context. You operate as an \
agent with tools that let you read source documents. The summaries you produce are saved as \
study notes the reader will return to later, so they must be accurate and self-contained.

YOUR TASK
When given a file or document:
1. Use the read_file tool to read the FULL source before summarizing. Never summarize a file you haven't read.
2. Produce the summary as a markdown document with these sections:
   - **Bottom line**: one sentence — the single most important takeaway.
   - **Key points**: 3-5 bullets, one distinct idea each, in order of importance.
   - **Common misconceptions**: Identify highly prevalent misunderstandings that could potentially arise (omit this section if there are none).

RULES
- Be concise. Cut filler, keep substance. Use plain language, but keep the author's key terms where technical jargon matters.
- Preserve nuance. Never flatten a trade-off into a one-sided claim.
- Ground everything in the source. Do NOT add facts, examples, or opinions that aren't in the original.
- Mark uncertainty. If something is context-dependent or the author is speculating, signal it ("the author argues...").
- If you can't read the file or it's empty, say so plainly and stop. Do not guess at the contents."""

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=[read_file_declaration])], # define the tool in the config so the model "sees" it
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

available_tools = {"read_file": read_file}

# --- 3. SCAN input folder ---
files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if os.path.isfile(os.path.join(INPUT_FOLDER, f))
    and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
])

if not files:
    print(f"No supported files found in '{INPUT_FOLDER}/'. Drop a .txt or .md file in and try again.")
else:
    print(f"Found {len(files)} file(s): {files}\n")

    for filename in files:
        filepath = os.path.join(INPUT_FOLDER, filename)
        print(f"\n{'='*50}\nProcessing: {filename}\n{'='*50}")

        # Reset conversation for each new file
        contents = [
            types.Content(role="user", parts=[types.Part(text=f"Summarize the file at '{filepath}' for me.")])
        ]

        # --- 4. THE LOOP (unchanged) ---
        while True:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=contents,
                config=config,
            )

            if response.function_calls: # the model decided that it needs the tool to answer
                fc = response.function_calls[0]
                print(f"[1] Model REQUESTED: {fc.name}(path={fc.args.get('path')})")

                result = available_tools[fc.name](**fc.args)
                print(f"[2] Tool RAN, returned {len(result)} chars. Feeding back...")

                contents.append(response.candidates[0].content)
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                        id=fc.id, 
                    )],
                ))
            else:
                # Save summary as a .md file in output/
                output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0] + ".md")
                with open(output_path, "w", encoding="utf-8") as out:
                    out.write(response.text)
                print(f"[3] Summary saved → {output_path}")
                break