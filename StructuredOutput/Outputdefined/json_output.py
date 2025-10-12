
from langchain_openai import ChatOpenAI
from typing import TypedDict
from typing import Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()


json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of review either negative, positive or neutral"
    },
    "pros": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Write down all the pros inside a list",
      "default": None
    },
    "cons": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Write down all the cons inside a list",
      "default": None
    },
    "name": {
      "type": "string",
      "description": "Write the name of the reviewer",
      "default": None
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}



structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""Ever wanted someone (or something) to judge your life choices right from your front porch? Welcome the Battery Video Doorbell — your new sentinel, watchdog, and gossip reporter in one.

What It Delivers:

Crisp video that catches your postman’s existential dread when he realizes he’s late again.

Motion alerts that inspire you to wave confidently at strangers, just in case they’re watching.

Easy install (no wires!), so now you can smugly boast, “I’m tech-savvy and lazy.”

Minor Quibbles:

It’s so vigilant, I’m half-convinced it’s already ghosted me for not posting enough deliveries.

Battery life is decent, but don’t expect it to last until next Blue Moon if you activate live view constantly.

In short: this doorbell doesn’t just ring — it judges, records, and silently plots. In the best way possible.
                                 
Review By Atul""")


print(result)
