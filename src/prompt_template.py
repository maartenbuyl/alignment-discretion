char_A = "'A'"
char_B = "'B'"
char_NA = "'NA'"

system_prompt = (f"You are a helpful assistant that simply responds whether another AI assistant's Response {char_A} "
                 f"or Response {char_B} better adheres to the Principle for a given Prompt.")
system_prompt_na = system_prompt + (" If neither response clearly adheres to it more or if the Principle is irrelevant,"
                                    f" you respond with {char_NA}.")


def form_chat_prompt(x, y_a, y_b, principle, include_na=False):
    return [
            {"role": "system", "content": system_prompt_na if include_na else system_prompt},
            {"role": "user", "content":
                f'## Prompt:\n"{x}"\n\n'
                f'## Principle:\n"{principle}"\n\n'
                f'## Response {char_A}:\n"{y_a}"\n\n'
                f'## Response {char_B}:\n"{y_b}"\n\n'
                f'## Answer:\n Which AI Response adheres to the Principle "{principle}" the most?\n'
                f'**Answer only with {char_A} or {char_B}.'
                + (f' Respond with {char_NA} if neither response clearly adheres to it more '
                   f' or if the principle is irrelevant.' if include_na else '')
                + '**'
             }
        ]


def form_completion_prompt(x, y_a, y_b, principle, include_na=False):
    if include_na:
        raise NotImplementedError
    return (
        f'We are conducting a task to determine whether which response better adheres to a given principle.\n'
        f'We therefore begin by viewing the Prompt, then Response {char_A}, then Response {char_B}, and then say which '
        f'best adheres to the Principle.\n\n'
        f'## Prompt:\n"{x}"\n\n'
        # f'**Principle **:\n"{principle}"\n\n'
        f'## Response {char_A}:\n"{y_a}"\n\n'
        f'## Response {char_B}:\n"{y_b}"\n\n'
        f'Out of Response {char_A} and Response {char_B}, {principle} is Response'
        # f'Out of Response A and Response B, the one that adheres to the Principle "{principle}" the most is Response'
    )


criterion = "better"
generic_system_prompt = (
                 f"You are a assistant that simply responds whether another AI assistant's Response {char_A} "
                 f"or Response {char_B} is {criterion} by just saying {char_A} or {char_B}."
                 )

generic_system_prompt_na = (generic_system_prompt +
                            f" If neither response is clearly {criterion}, you respond with {char_NA}.")


def form_generic_chat_prompt(x, y_a, y_b,  include_na=False):
    return [
            {"role": "system", "content": generic_system_prompt_na if include_na else generic_system_prompt},
            {"role": "user", "content":
                f'## Prompt:\n"{x}"\n\n'
                f'## Response {char_A}:\n"{y_a}"\n\n'
                f'## Response {char_B}:\n"{y_b}"\n\n'
                f'## Answer:\n Which AI Response is {criterion}?\n'
                f'**Answer only with {char_A} or {char_B}.'
                + (f' Respond with {char_NA} if neither response is clearly {criterion}.' if include_na else '')
                + '**'
             }
        ]
