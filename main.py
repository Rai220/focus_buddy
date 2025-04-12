import pyautogui
from langchain_core.messages import HumanMessage
from langchain_gigachat import GigaChat
from dotenv import load_dotenv, find_dotenv
from rich import print as rprint
import argparse
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())


class ScreenshotAnalysis(BaseModel):
    """Результат анализа скриншота."""

    thoughts: str = Field(..., description="Мысли по поводу скриншота")
    moving_to_gool: bool = Field(
        ...,
        description="Соответствует ли то что происходит на скриншоте движению к цели",
    )
    tips: str = Field(
        ..., description="Мотивирующая критика человека, если он не движется к целе"
    )


llm = GigaChat(
    model="GigaChat-2-Max",
    verify_ssl_certs=False,
    profanity_check=False,
    base_url="https://gigachat.ift.sberdevices.ru/v1",
    top_p=0,
    streaming=True,
    max_tokens=16000,
)


def take_screenshot_every_15_seconds(goal):
    print(f"Goal: {goal}")

    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)

    struct_llm = llm.with_structured_output(
        ScreenshotAnalysis, method="format_instructions"
    )

    while True:
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = os.path.join(
                screenshot_dir, f"screenshot_{timestamp}.png"
            )
            screen = pyautogui.screenshot()
            screen.save(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
            file = llm.upload_file(open(screenshot_path, "rb"))

            messages = [
                HumanMessage(
                    content=f"Проанализируй чем занимается человек на скриншоте и оцени, соответствует ли это достижению цели: {goal}",
                    additional_kwargs={"attachments": [file.id_]},
                )
            ]

            resp = struct_llm.invoke(messages)
            rprint(resp)
            if not resp.moving_to_gool:
                pyautogui.alert(
                    text=f"Ты хотел заняться {goal}! Please refocus!\n\n{resp.tips}",
                    title="Ты занимаешься не тем, чем надо!",
                    button="OK",
                )
            time.sleep(15)
        except Exception as e:
            print(f"An error occurred: {e}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take screenshots every 15 seconds.")
    parser.add_argument(
        "--goal", required=True, help="Specify the goal for the session."
    )
    args = parser.parse_args()

    take_screenshot_every_15_seconds(args.goal)
