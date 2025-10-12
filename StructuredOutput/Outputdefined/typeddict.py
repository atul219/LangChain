from langchain_openai import ChatOpenAI
from typing import TypedDict
from typing import Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()


# schema
# simple typed dict

# class Review(TypedDict):
#     summary: str
#     sentiment: str

# annotated
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of review"]
    sentiment: Annotated[str, "Return sentiment of review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Ever wanted someone (or something) to judge your life choices right from your front porch? Welcome the Battery Video Doorbell — your new sentinel, watchdog, and gossip reporter in one.

What It Delivers:

Crisp video that catches your postman’s existential dread when he realizes he’s late again.

Motion alerts that inspire you to wave confidently at strangers, just in case they’re watching.

Easy install (no wires!), so now you can smugly boast, “I’m tech-savvy and lazy.”

Minor Quibbles:

It’s so vigilant, I’m half-convinced it’s already ghosted me for not posting enough deliveries.

Battery life is decent, but don’t expect it to last until next Blue Moon if you activate live view constantly.

In short: this doorbell doesn’t just ring — it judges, records, and silently plots. In the best way possible.""")


print(result)
