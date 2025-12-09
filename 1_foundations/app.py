from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Yanki Markovich"
        self.max_evaluator_retries = 2  # Max times to regenerate if evaluation fails
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def evaluator_system_prompt(self):
        """System prompt for the evaluator LLM that checks response quality."""
        return f"""You are an evaluator for a career chatbot that represents {self.name}.
Your job is to evaluate responses and ensure they meet quality standards.

You will receive:
1. The user's question
2. The chatbot's response
3. Context about {self.name} (summary and LinkedIn profile)

Evaluate the response against these criteria:
1. **ACCURACY**: Does the response only contain information that can be verified from the provided context? No made-up facts.
2. **IN_CHARACTER**: Does it sound like {self.name} speaking, not a generic AI? Uses first person appropriately.
3. **PROFESSIONALISM**: Is the tone appropriate for a professional context (potential employers/clients)?
4. **RELEVANCE**: Does it actually address the user's question?
5. **NO_HALLUCINATIONS**: Does NOT invent specific details (dates, numbers, company names) not in the context.

Respond ONLY with valid JSON in this exact format:
{{"passed": true, "score": 8, "issues": [], "feedback": ""}}

Or if there are problems:
{{"passed": false, "score": 4, "issues": ["issue 1", "issue 2"], "feedback": "specific guidance to fix the response"}}

Be strict but fair. Minor issues don't require a fail. Only fail if there are clear problems.
If the response admits to not knowing something, that's GOOD, not a failure.

## Context about {self.name}:

### Summary:
{self.summary}

### LinkedIn Profile:
{self.linkedin}"""

    def evaluate_response(self, user_question, bot_response):
        """
        Evaluates a chatbot response using a second LLM call.
        Returns (passed: bool, feedback: str, score: int)
        """
        evaluation_prompt = f"""Please evaluate this chatbot response:

**User Question:** {user_question}

**Chatbot Response:** {bot_response}

Evaluate and respond with JSON only."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.evaluator_system_prompt()},
                    {"role": "user", "content": evaluation_prompt}
                ],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            passed = result.get("passed", True)
            feedback = result.get("feedback", "")
            score = result.get("score", 5)
            issues = result.get("issues", [])

            # Log evaluation results
            print(f"[Evaluator] Score: {score}/10, Passed: {passed}", flush=True)
            if issues:
                print(f"[Evaluator] Issues: {issues}", flush=True)

            return passed, feedback, score

        except Exception as e:
            print(f"[Evaluator] Error during evaluation: {e}", flush=True)
            # If evaluation fails, pass the response through
            return True, "", 5

    def generate_response(self, messages):
        """Generate a response from the LLM, handling any tool calls."""
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason == "tool_calls":
                assistant_message = response.choices[0].message
                tool_calls = assistant_message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(assistant_message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content, messages

    def chat(self, message, history):
        """Main chat function with evaluator-optimizer pattern."""
        user_question = message  # Save original question for evaluator
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]

        # Generate initial response
        bot_response, messages = self.generate_response(messages)

        # Evaluator-optimizer loop
        for attempt in range(self.max_evaluator_retries):
            passed, feedback, score = self.evaluate_response(user_question, bot_response)

            if passed:
                print(f"[Evaluator] Response passed on attempt {attempt + 1}", flush=True)
                return bot_response

            # Response didn't pass - regenerate with feedback
            print(f"[Evaluator] Response failed (attempt {attempt + 1}), regenerating...", flush=True)

            # Add the failed response and feedback to messages for context
            messages.append({"role": "assistant", "content": bot_response})
            messages.append({
                "role": "user",
                "content": f"[INTERNAL FEEDBACK - Please revise your response]\n"
                           f"Your previous response had issues: {feedback}\n"
                           f"Please provide an improved response to the original question: {user_question}"
            })

            # Generate improved response
            bot_response, messages = self.generate_response(messages)

        # If we've exhausted retries, return the last response anyway
        print(f"[Evaluator] Max retries reached, returning last response", flush=True)
        return bot_response
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    