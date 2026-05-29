    import os
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv

    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # --- 1. THE TOOL: Python function ---
    def read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # --- 2. DESCRIBE the tool to the model (this is what it "sees") ---
    read_file_declaration = {
        "name": "read_file",
        "description": "Reads a text file from local disk and returns its full contents. "
                    "Use this to see a file's contents before summarizing it.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file, e.g. '.txt'"},
            },
            "required": ["path"],
        },
    }

    #################################################################

    # setting the stage, task, rules
    SYSTEM_PROMPT = """You are a meticulous technical summarizer. You act as an intelligent filter: \
    you condense dense material into its absolute core essentials so thay even a non-technical reader grasps the main \
    takeaway at a glance — without losing crucial context. You operate as an \
    agent with tools that let you read source documents. The summaries you produce are saved as \
    study notes the reader will return to later, so they must be accurate and self-contained.

    YOUR TASK
    When given a file or document:
    1. Use the read_file tool to read the FULL source before summarizing. Never summarize a file you haven't read.
    2. Produce the summary in the output folder as a markdown file with these sections:
    - **Bottom line**: one sentence — the single most important takeaway.
    - **Key points**: 3-5 bullets, one distinct idea each, in order of importance.
    - **Common misconceptions**: Identify highly prevalent misunderstandings that could potentially arise (omit this section if there are none).

    RULES
    - Be concise. Cut filler, keep substance. Use plain language, but keep the author's key terms where technical jargons matters.
    - Preserve nuance. Never flatten a trade-off into a one-sided claim.
    - Ground everything in the source. Do NOT add facts, examples, or opinions that aren't in the original.
    - Mark uncertainty. If something is context-dependent or the author is speculating, signal it ("the author argues...").
    - If you can't read the file or it's empty, say so plainly and stop. Do not guess at the contents."""

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=[read_file_declaration])],
        # Disable auto-calling so WE drive the loop
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Map the tool name the model uses -> the real Python function
    available_tools = {"read_file": read_file}

    # --- 3. THE LOOP ---
    contents = [
        types.Content(role="user", parts=[types.Part(text="Summarize notes.txt for me.")])
    ]

    while True:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )

        if response.function_calls:
            fc = response.function_calls[0]
            print(f"[1] Model REQUESTED: {fc.name}(path={fc.args.get('path')})")

            # YOUR code runs the function. The model never touched your disk.
            result = available_tools[fc.name](**fc.args)
            print(f"[2] Tool RAN, returned {len(result)} chars. Feeding back to model...")

            # Append the model's request AND your result to the conversation
            contents.append(response.candidates[0].content)
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result},
                    id=fc.id,   # may be None on some models — that's fine
                )],
            ))
            # loop again — now the model has the file contents in context
        else:
            print("\n[3] Model gave FINAL answer:\n")
            print(response.text)
            break