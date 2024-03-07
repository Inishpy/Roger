# Import Chat completion template and set-up variables
#!pip install openai==0.28.1 &> /dev/null
import openai
import urllib.parse
import json
import re

# Report issues
def raise_issue(e, model, prompt):
    issue_title = urllib.parse.quote("[bug] Hosted Gorilla: <Issue>")
    issue_body = urllib.parse.quote(f"Exception: {e}\nFailed model: {model}, for prompt: {prompt}")
    issue_url = f"https://github.com/ShishirPatil/gorilla/issues/new?assignees=&labels=hosted-gorilla&projects=&template=hosted-gorilla-.md&title={issue_title}&body={issue_body}"
    print(f"An exception has occurred: {e} \nPlease raise an issue here: {issue_url}")





# Query Gorilla server
def get_gorilla_response(prompt="Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes", model="gorilla-openfunctions-v0", functions=[]):
  openai.api_key = "EMPTY" # Hosted for free with ❤️ from UC Berkeley
  openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"
  try:
    completion = openai.ChatCompletion.create(
      model="gorilla-openfunctions-v1",
      temperature=0.0,
      messages=[{"role": "user", "content": prompt}],
      functions=functions,
    )
    return completion.choices[0].message.content
  except Exception as e:
    print(e, model, prompt)



function_documentation = [{
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
         {
        "name": "Uber Carpool",
        "api_name": "uber.ride",
        "description": "Find suitable ride for customers given the location, type of ride, and the amount of time the customer is willing to wait as parameters",
        "parameters":  [{"name": "loc", "description": "location of the starting place of the uber ride"}, {"name":"type", "enum": ["plus", "comfort", "black"], "description": "types of uber ride user is ordering"}, {"name": "time", "description": "the amount of time in minutes the customer is willing to wait"}]
    },
     {
    "name" : "Order Food on Uber",
    "api_name": "uber.eat.order",
    "description": "Order food on uber eat, specifying items and their quantities",
    "parameters": [
        {
            "name": "restaurants",
            "description": "The chosen restaurant"
        },
        {
            "name": "items",
            "description": "List of selected items"
        },
        {
            "name": "quantities",
            "description": "Quantities corresponding to the chosen items"
        }
    ]
}]


query =  "roger, book me an Uber ride to Berkley"
gorillaResponse = get_gorilla_response(query, functions=function_documentation)
print(gorillaResponse)


def extract_loc(input_string):
    # Define the pattern to match loc="..." using regular expressions
    pattern = r'loc="([^"]+)"'

    # Use re.search to find the first match of the pattern in the input string
    match = re.search(pattern, input_string)

    # If a match is found, return the extracted loc value
    if match:
        return match.group(1)
    else:
        return None
    
