Instruction
You will be given a CADQuery code snippet <cadquery> and a rendered image <image>. The CAD model has a length (X-axis) of <length> units, width (Y-axis) of <width> units, height (Z-axis) of <height> units, and includes <through-holes> through holes.

Role
You are a senior CAD engineer.

Task
Generate a concise geometric description of the CAD model. The description must follow these rules:
1. Output must be a single JSON object with one field named "description".
2. The value of "description" must be a single paragraph of plain English text without newline characters, special symbols, or markdown formatting.
3. Only describe geometry and dimensions. Do not mention color or material.
4. Combine information inferred from both the CADQuery code and the rendered image.
5. Be written in clear, fluent, human-readable language.
6. Keep the description as concise as possible and ensure it does not exceed 300 tokens.
7. Use "unit" for units.

Output format
{"description": "<your one-paragraph description here>"}