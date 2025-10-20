import sys
import os
import argparse
from google import genai
from google.genai import types
from dotenv import load_dotenv


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    parser = argparse.ArgumentParser()
    parser.add_argument("user_prompt")                    
    parser.add_argument("--verbose", action="store_true")  # boolean flag
    args = parser.parse_args()

    messages = [
        types.Content(role="user", parts=[types.Part(text=args.user_prompt)]),
    ]

    generate_content(client, messages, args)


def generate_content(client, messages, args):
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=messages,
    )
    if args.verbose:
        print("User prompt:", args.user_prompt)
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Response tokens:", response.usage_metadata.candidates_token_count)
    
    print("Response:")
    print(response.text)


if __name__ == "__main__":
    main()