"""Spanish instruction formatting shared by agricultural data and inference."""


def instruction_prompt(instruction, subtask=None):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Se requiere una orden de texto por robot")
    instruction = instruction.strip()
    if subtask is None:
        return instruction
    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError("La subtarea debe contener texto")
    subtask = subtask.strip()
    separator = " " if instruction.endswith((".", "!", "?")) else ". "
    return f"{instruction}{separator}Ahora {subtask[0].lower()}{subtask[1:]}"


def tokenize_instructions(tokenizer, prompts):
    """Apply E5's feature-extraction prefix consistently."""
    return tokenizer([f"query: {prompt}" for prompt in prompts], padding=True,
                     truncation=True, max_length=512, return_tensors="pt")
