from typing import Type, Union, Optional, Any, Dict, List, Tuple, get_origin, get_args
from dataclasses import dataclass, fields, is_dataclass
from itertools import product
from types import NoneType
from enum import Enum

from src.agent.proxy_client import ProxyClient

import secrets
import string
import math
import copy
import time
import json
import re

IS_DEBUG = False

@dataclass
class JsonModel(object):
    @classmethod
    def from_json(cls, data: dict) -> "JsonModel":
        kwargs = {}

        for f in fields(cls):
            if f.name not in data: continue            
            kwargs[f.name] = cls.__convert_from__(data[f.name], f.type)

        return cls(**kwargs)

    def to_json(self) -> Dict:
        return self.__convert_to__(self)

    @staticmethod
    def __convert_from__(value, t: Type["JsonModel"] | Any) -> Any:
        origin, args = get_origin(t), get_args(t)

        if origin is Union and len(args) == 2 and NoneType in args:
            return JsonModel.__convert_from__(
                value, next(a for a in args if a is not NoneType)
            ) if value is not None else None

        if origin is list and isinstance(value, list):
            return [JsonModel.__convert_from__(v, args[0]) for v in value]

        if origin is tuple and isinstance(value, list):
            return tuple(JsonModel.__convert_from__(v, args[0]) for v in value)

        if origin is dict and isinstance(value, dict):
            kt, vt = args
            return {JsonModel.__convert_from__(k, kt): JsonModel.__convert_from__(v, vt) for k, v in value.items()}

        if is_dataclass(t) and isinstance(value, dict): return t.from_json(value)

        if t is int and isinstance(value, str): return int(value)
        if t is float and isinstance(value, str): return float(value)
        if t is str: return str(value)

        return value

    @staticmethod
    def __convert_to__(obj: Any) -> Any:
        if is_dataclass(obj): return {f.name: JsonModel.__convert_to__(getattr(obj, f.name)) for f in fields(obj)}
        
        if isinstance(obj, dict): return {JsonModel.__convert_to__(k): JsonModel.__convert_to__(v) for k, v in obj.items()}
        if isinstance(obj, list): return [JsonModel.__convert_to__(x) for x in obj]

        return obj

class LLM(str, Enum):
    SONNET = "anthropic/claude-sonnet-5.5"
    HAIKU = "anthropic/claude-haiku-5.5"
    GPT6 = "openai/gpt-6-luna"

class Prompt:
    SUMMARIZE_SYS = "# You are an agent to analyze the user's shopping requirements and available tools. You are given:\n\t- A raw text input from user\n\t- A set of tools for shopping assistance\n\n# Your mission:\n\t- Parse user's verbose input into the structured information\n\t- Classify the given tools\n\n# Challenges:\n\t- User may have several product intents.\n\t- There may exist dependencies for the intents.\n\t\t* What is dependency?\n\t\t\tA requirement that binds two or more product intents. It makes an intent's submission choice dependant on other intent's submission, which is impossible to resolve it independently.\n\t\t* Examples of dependency\n\t\t\tInput: `All of your chosen goods must be in the same shop.` -> The shop name is unspecified. The recommended products for all intents must be sold at the same shop.\n\t\t\tInput: `The first product must be compatibile with the third product.` -> The exact compatibility value is unspecified, but the two of the recommended products must be compatible.\n\t\t* Example that is NOT dependency (False Positive)\n\t\t\tInput: `All these things should be for Honda model.`\n\t\t\tThe value of property \"model\" has been explicitly requested as \"Honda\". It's a shared requirement, NOT a dependency because each intent can still be resolved independently.\n\t\t\tYou must include a phrase \"For Honda model\" in every intent requirements' \"queries\" list, and that's all. No dependency.\n\t\t* (Exception) Budget restriction is essentially a dependency, for sure. However, do NOT include budget-related phrase in \"collection->dependencies\" field, because \"budget\" exists as a separate field.\n\t\t* Phrases about the procedure like \"find one by one\" are NOT dependencies.\n\t- There may exist reference products.\n\t\t* Definition:\n\t\t\tAn intent may point to another product for a requested (or excluded) property value, instead of explicit mention.\n\t\t\tThe pointed product is the reference for this intent.\n\t\t\tThe requested (or excluded) property value can be revealed only when the agent identified and inspected the reference product.\n\t\t* Examples of reference\n\t\t\tInput: `Color must be the same with the lotion that is sold by Kailey's and brand CR7.`\n\t\t\t\t\"The lotion sold by Kailey's and with brand CR7\" is the reference product. The base product's color must be its color.\n\t\t\tInput: `Style must be different to the listing of Classico in range TV and resolution 1080.`\n\t\t\t\t\"Listing of Classico in range TV and resolution 1080\" is the reference product. The base product's style must not be its style.\n\t- There may exist priorities.\n\t\t* What is priority?\n\t\t\tA requirement that defines the choice condition among multiple matches for intent.\n\t\t\tIt does not evaluate products' matchness, but it clarifies the key property and preferred direction for choosing the top 1 from all matching candidates.\n\t\t* Examples of user input for priority\n\t\t\t`Give me the cheapest one.`\n\t\t\t`Looking for ... with the longest sleeves length.`\n\n# Return JSON object like:\n{{\n\t\"agent\": {{\n\t\t\"max_messages\": <int|null>,\n\t\t\"max_steps\": <int|null>,        \n\t\t\"substitute_allowed\": <bool>,\n\t\t\"fallback\": <str|null>,\n\t\t\"speak\": <str|null>,\n\t\t\"guide\": [<str>, ...]\n\t}},\n\t\"collection\": {{\n\t\t\"budget\": {{\n\t\t\t\"amount\": <int|null>,\n\t\t\t\"unit\": <str|null>\n\t\t}},\n\t\t\"dependencies\": [<str>, ...]\n\t}},\n\t\"intents\": [\n\t\t{{\n\t\t\t\"price\": {{\n\t\t\t\t\"min\": <int|null>,\n\t\t\t\t\"max\": <int|null>,\n\t\t\t\t\"unit\": <str|null>\n\t\t\t}},\n\t\t\t\"quantity\": <int|null>,\n\t\t\t\"priority\": <str|null>,\n\t\t\t\"queries\": [<str>, ...],\n\t\t\t\"reference\": <str|null>\n\t\t}},\n\t\t... ...\n\t],\n\t\"tools\": {{\n\t\t\"search\": [<str>, ...],\n\t\t\"inspect\": [<str>, ...],\n\t\t\"cart\": [<str>, ...],\n\t\t\"finish\": [<str>, ...],\n\t\t\"message\": [<str>, ...],\n\t\t\"other\": [<str>, ...]\n\t}}\n}}\n\n# Field Glossary:\n\t- \"agent\": the requirements that the agent must adhere during the shopping assistance process\n\t\t* \"max_messages\": maximum number of messages the agent can send user, if mentioned\n\t\t* \"max_steps\": maximum number of tool call steps the agent can take until finish, if mentioned\n\t\t* \"substitute_allowed\": true if user explicitly mentioned that substitute product can be submitted if no matching candidate is found, otherwise false\n\t\t* \"fallback\": the conditions and rules for choosing the substitute, if it's allowed and mentioned\n\t\t* \"speak\": sentence(s) that user encourages the agent to ask, tell or confirm something\n\t\t\te.g. \"Ask me which style to look through, then I'll tell you immediately.\"\n\t\t\te.g. \"Tell me what's worth asking before I commit my savings.\"\n\t\t* \"guide\": additional instructions except for the above fields\n\t\t\te.g. \"Always use statistic tool just before ordering, or I'll reject.\"\n\t\t* NEVER in \"fallback\" and \"guide\" include phrases for allowing/disallowing substitutes, because it's already specified as a boolean value in \"substitute_allowed\".\n\t- \"collection\": the requirements that the cart (the collection of all submitting products for the intents) must satisfy\n\t\t* \"budget\": upper limit of total spending, if mentioned\n\t\t* \"dependencies\": a list of sentences representing the dependencies for the submitting collection (empty list if no dependency exists)\n\t\t\tEach item must be a full sentence like \"All must be in the same shop.\"\n\t\t\tDo NOT include budget-related sentence.\n\t- \"intents\": a list of requirements dedicated to every individual product intent\n\t\t* \"price\": requested price range for the intent\n\t\t* \"quantity\": requested quantity for the intent\n\t\t* \"priority\": a short sentence representing the priority to choose the top 1, if mentioned\n\t\t\tAlways start with \"Choose ...\" like \"Choose the cheapest\" or \"Choose the one with the longest sleeves\".\n\t\t* \"queries\": a list of sentences representing the requirements as well as the informative contexts except for those related to \"price\", \"quantity\" and \"priority\"\n\t\t\te.g. [\"Comic style book written by Arthur Kailey.\", \"It must be in French.\", \"Story about dragons.\"]\n\t\t\te.g. [\"Basketball shoes with size as REFERENCE.\", \"Fit for students.\"]\n\t\t* use the identifier \"REFERENCE\" for reference products when you describe items of \"queries\", if reference exists\n\t\t\tThe idea is to separate the queries of base product from reference product.\n\t\t\tSpecify like \"size as REFERENCE\" or \"type different to REFERENCE\" in \"queries\", and include nothing else for the reference-dedicated requirements.\n\t\t* \"reference\": query for the reference product if exists, otherwise null\n\t\t\te.g. \"What's sold in American Store with volleyball shoes range.\"\n\t- \"tools\": a dictionary of all available tool names classified by categories\n\t\t* \"search\": a list of tool names for finding products\n\t\t* \"inspect\": a list of tool names for viewing/comparing product details\n\t\t* \"cart\": a list of tool names for cart operations like adding, removing, reviewing, etc. in the cart\n\t\t* \"finish\": a list of tool names for finalization like placing order, decline, etc.\n\t\t* \"message\": a list of tool names for sending user a message\n\t\t* \"other\": a list of tool names classified into none of above\n\n# Guidelines:\n\t- Exclude from your output, user's verbose story and subjective explanation that are not meaningful for e-commerce search and ranking. Examples:\n\t\t* General phrases for emphasis of needing match like `I'm short of money so strictly meet the budget.`\n\t\t* Phrases like \"must be in stock\" or \"must be on hand\"\n\t\t* Phrases related to the listing photos (The agent is text-based, therefore can't check photos.)\n\t- Each item object of \"intents\" list MUST be independently complete.\n\t\t* If a property value is explicitly shared multiple several intents, specify it in \"queries\" for all those intent objects.\n\t\t* Do NOT include a phrase related to a collection dependency in \"queries\" of individual intent.\n\t- For each intent object, the children field values MUST be strictly non-overlapped semantically.\n\t\t* Do NOT include in \"queries\", a redundant phrase related to \"price\", \"quantity\" or \"priority\".\n\t\t* Do NOT include the reference product's requirements in the base product's \"queries\" to prevent mixing up.\n\t- Respect the user's original wording and number/unit formatting for the free text values of your output.\n\t\t* Do NOT convert price/budget numbers and currency units by your own discretion. Just use the format as mentioned by user.\n\t\t* If price/budget was mentioned with both large and small units, use the small unit for output.\n\t\t\te.g. \"My budget is 210cts (2.1$).\" -> `\"budget\": {{ \"amount\": 210, \"unit\": \"cts\" }}`\n\t\t* Never replace a specific model or attribute with a broader interpretation.\n\t- If user explicitly mentioned that a property is optional, specify it clearly for \"queries\" output to avoid confusing must-have and optional requirements.\n\t\te.g. \"Gothic style is preferred but not strictly required.\"\n\t- Rare but not unseen, there may exists a special instruction that the agent must respect during the process, specified in tool descriptions.\n\t\t* e.g. \"This hello tool must be called at the very start of your process.\"\n\t\t* Add such an important instruction to \"agent->guide\", if exists.\n\t- Do NOT omit or paraphrase any informative details in \"queries\" to prevent losing requirements.\n\t- Make sure all given tools are included in \"tools\". Every tool must be classified into exactly one tool class.\n\t- Return JSON only. No plain text.\n\n# Examples of Incorrect Item for \"queries\" List:\n\t- \"2 sun glasses ...\" (\"2\" must be moved to \"quantity\" field, not here.)\n\t- \"... with the least price.\" (\"with the least price\" must be moved to \"priority\" field as \"Choose the cheapest.\", not here.)\n\t- \"... less than 200 cents.\" (\"less than 200 cents\" must be moved to \"price\" field, not here.)\n\t- \"A dress for visiting Maxim on Thursday.\" (\"for visiting Maxim on Thursday\" must be removed, because it's non-informative for e-commerce search.)"
    SUMMARIZE_USER = "# Raw User Input\n{query}\n\n# Available Tools\n{tools}"
    EXERCISE_SYS = "# You are an agent to assist user's shopping. You are given:\n\t- user's shopping requirements\n\t- a set of tools\n\t\t* search tool(s) for finding products with search keywords and/or filter options\n\t\t* inspect tool(s) for viewing product details\n\t\t* other tools\n\n# You are responsible with the initial a few steps. While strictly adhering the rules listed in \"general->guide\" of shopping requirements, you MUST:\n\t- exercise search/inspect tools, to understand the ways they work.\n\t- analyze the search tool results to identify:\n\t\t* the search tools' terminology for certain field names used in product details\n\t\t* the correct category values for every intent's base and reference product\n\t\t* the correct string values of shop and/or brand name for every intent's base and reference product if those names are explicitly requested\n\t- analyze the inspect tool results to identify:\n\t\t* the best inspect tool that returns the property map\n\t\t* the best inspect tool's terminology for certain field names used in product details\n\t- return with all the identified information.\n\n# Challenges:\n\t- User may have several product intents.\n\t- A search tool may have a switch argument that enables or disables other arguments.\n\t\t* Certain switch value may disable \"max_price\"-like option argument.\n\t\t* \"keywords\"-like argument may work for certain switch value only.\n\t- Different inspect tools showcase the product details with different levels.\n\t\t* Certain inspect tools only reveal the product listing details similar to what search tools produced.\n\t\t* Certain inspect tools reveal the comprehensive property map that search tools don't output.\n\t- We have no pre-knowledge about the terminology of each tool for its product details output.\n\t- There may exist reference products.\n\t\t* Definition:\n\t\t\tAn intent may point to another product for a requested (or excluded) property value, instead of explicit mention.\n\t\t\tThe pointed product is the reference for this intent.\n\t\t\tThe requested (or excluded) property value can be revealed only when the agent identified and inspected the reference product.\n\t\t* Examples of reference\n\t\t\tInput: `Color must be the same with the lotion that is sold by Kailey's and brand CR7.`\n\t\t\t\t\"The lotion sold by Kailey's and with brand CR7\" is the reference product. The base product's color must be its color.\n\t\t\tInput: `Style must be different to the listing of Classico in range TV and resolution 1080.`\n\t\t\t\t\"Listing of Classico in range TV and resolution 1080\" is the reference product. The base product's style must NOT be its style.\n\n# Category:\n\t- Category is a word or a short phrase representing the core product form or type.\n\t- It must be included in each listing details retrieved by search tools.\n\t- The exact field name for category is not predictable (e.g. \"department\", \"prod_type\", \"xyblz\", etc.), but note that this terminology is universal across all search tools.\n\t- You must identify the corresponding field name and the correct category values for all intents.\n\t- The correct category value is the one whose wording is most aligned to the \"queries\" of the intent requirements.\n\t\t* Example 1\n\t\t\tUser said: \"shoes\"\n\t\t\tCategory value candidates: \"Shoes\", \"Clothing & Shoes\"\n\t\t\tMost aligned choice: \"Shoes\"\n\t\t* Example 2\n\t\t\tUser said: \"shirts or jackets\"\n\t\t\tCategory value candidates: \"Shirts\", \"Jackets\", \"Shirts and Jackets\"\n\t\t\tMost aligned choice: \"Shirts and Jackets\"\n\t- Category is the non-negotiable core field for product matching. Therefore, you must identify the correct category value of:\n\t\t* the base product for every intent\n\t\t* the reference product for every intent, if exists\n\n# Inspect tool preference priority (top to bottom):\n\t(1) The tool that produces the property map details that are not retrieved from search tools, is preferred.\n\t(2) The tool that can accept more product IDs as an argument at one call, is preferred over less or single ID.\n\t(3) The tool with less number of arguments, is preferred.\n\n# You must proceed the following normal process:\n\t(1) Obtain at least one retrieval for each intent's base product, using search tool(s).\n\t\tSelect the search tool and switch value for keywords-based mode.        \n\t\tFor every intent, execute the following steps:       \n\n\t\t(1.1) Prepare search keywords including informative and discriminative tokens from \"queries\" but:\n\t\t\t- tokens related to excluded requirements (e.g. `Leave out those with color orange, which I hate.` -> Never add \"orange\" in keywords, or the search will retrieve products with \"orange\" property.)\n\t\t\t- tokens like \"and\", \"or\", \"for\", \"with\", \"in\", \"compatible\", \"suitable\", \"new\", etc. (such too generic tokens degrade search efficiency)\n\t\t\t- brackets, punctuation marks and symbolic characters like \"&\", \"-\", \"/\", \".\", \">\", \"<\", \"(\", \")\", \"[\", \"]\", ...\n\t\t\t- tokens from non-informative situational context (e.g. `It's for my daughter's school events.` -> Ignore this phrase.)\n\t\t\t- tokens from reference product's requirements\n\t\t\t- token \"REFERENCE\" (it's the identifier for reference)\n\t\t(1.2) Trigger a search with the chosen keywords plus the relevant filter options like price cap as requested. However,\n\t\t\t- do NOT leverage a tool argument related to category, because category value is not yet determined\n\t\t(1.3) Check the search result, and either continue or skip.\n\t\t\t- If you get at least one retrieval, skip to the next intent. NEVER dig into more to extend retrievals.\n\t\t\t- Otherwise, go back to (1.1) with adjusted keywords choice.\n\n\t\tIf all intents were found with retrieval, proceed to (2).\n\t(2) Get insight into the inspect tools.\n\t\t(2.1) Choose one most promising retrieval's product ID for the first intent.\n\t\t(2.2) Trigger every inspect tool exactly once, for the chosen product ID.\n\t\t(2.3) Review the inspect results and identify the best inspect tool according to the inspect tool preference priority.\n\n# Action priority for every turn (top to bottom):\n\t- The instructions listed in \"general->guide\" of shopping requirements are your TOP-PRIORITY rules!\n\t\t* If an instruction enforces you to do some action at this very turn, you MUST conduct that action.\n\t\t* \"general->guide\" instructions override all conflicting strategies and normal process steps.\n\t- Follow the normal process step by step using tool calls, while avoiding the violations against \"general->guide\".\n\t- Return with a JSON object response when all normal process steps are completed.\n\n# Search keywords hint:\n\t- Search keywords are limited to max 6 first tokens. The later tokens are ignored.\n\t- The search keywords are ideally:\n\t\t* [discriminative category tokens] + [shop name token if shop mentioned] + [brand name token if brand mentioned] + [discriminative property value tokens]\n\t\t* If it exceeds 6 tokens, remove the least discriminative tokens to fit the limit.\n\t\t* Alter removals whenever you adjust the keywords for next search attempt.\n\n# Response format:\n\t- Tool call(s) only, while in processing (1)-(2).\n\t- Return a JSON object with the following structure, when all normal process steps are completed:\n{{\n\t\"intents\": {{\n\t\t<intent_id_1, str>: {{\n\t\t\t\"category\": <str>, // base product's category value\n\t\t\t\"shop\": <str|null>, // base product's exact shop name as in the product details, if a certain shop/seller name has been requested in \"queries\"\n\t\t\t\"brand\": <str|null> // base product's exact brand name as in the product details, if a certain brand name has been requested in \"queries\"\n\t\t}},\n\t\t... ...\n\t}},\n\t\"search_tool_insight\": {{\n\t\t// field key name terminology of product details retrieved by search tools\n\n\t\t\"product_id\": <str>, // field name used for product ID\n\t\t\"category\": <str>, // field name used for category\n\t\t\"price\": <str>, // field name used for price\n\t\t\"title\": <str|null>, // field name used for title, if corresponding exists\n\t\t\"shop\": <str|null>, // field name used for shop, if corresponding exists\n\t\t\"brand\": <str|null>, // field name used for brand, if corresponding exists\n\t\t\"is_available\": <str|null>, // field used used for stock-availability as a boolean value, if corresponding exists\n\t\t\"stock_amount\": <str|null>, // field name used for current stock amount as an integer value, if corresponding exists\n\t\t\"free_text\": [<str>, ...] // field names used for free lengthy texts like title, description, summary, etc. in the search retrieval details\n\t}},\n\t\"inspect_tool_insight\": {{\n\t\t\"best\": <str>, // name of the best inspect tool\n\t\t\"capacity\": <int>, // maximum number of product IDs that the best inspect tool can accept as its argument (1 if it only accepts a single ID or uncertain)\n\t\t\"free_text\": [<str>, ...] // field names used for free lengthy texts like title, description, summary, etc. in the best inspect tool's result product details\n\t}}\n}}\n\n# Guidelines:\n\t- You are responsible with initial a few steps of the whole shopping process. Therefore,\n\t\t* Do NOT send messages (including questions) to the user yet, unless it's strictly enforced to do at this initial phase by \"general/guide\".\n\t\t* Do NOT try to verify the matchness of products, which is out of scope for your job.\n\t- NEVER search for:\n\t\t* the intent that has been already found with at least one retrieval in the previous searches, and\n\t\t* the reference products\n\t- You MUST identify the \"category\" value. However,\n\t\t* Set null for \"shop\" and \"brand\" unless shop/seller or brand name was explicitly mentioned in \"queries\". Do NOT derive unrequested values from the retrievals.\n\t- For \"free_text\", list only the field names that may have substantive descriptive text values semantically. The followings are not \"free_text\" fields:\n\t\t* the field whose value type is a dictionary (like attribute map)\n\t\t* the field whose value type is not string\n\t\t* the field whose value type is string but for identifiers, labels, short names, numeric strings and categorical values\n\t- You must find the value for \"inspect_tool_insight->capacity\" field from the best inspect tool's description. Set 1 if unclear.\n\t- For every search tool call, always pass the argument value for `intent_id`, to clarify which intent you're targeting at this search attempt.\n\t- Return with either one tool call or one JSON object. No plain text.\n\t\t* One tool call only, until the final JSON output.\n\t\t* One JSON object only, for the final output."
    EXERCISE_USER = "# Shopping Requirements\n{requirements}"
    EVENT_SYS = "# You are given:\n\t- current shopping requirements\n\t- previous history of tool calls, tool response objects and handled events\n\n# You must:\n\t- identify the unhandled event messages from the previous tool responses, and\n\t- output the corresponding updates for the shopping requirements and/or the stock status.\n\n# What are event and event message?\n\t- An event is the happening which enforces to update the current shopping requirements or the product stock status.\n\t- An event message is a free text buried inside a tool response, that signals an event.\n\n# Shopping requirements update events include, but are not limited to:\n\t- user brings a requirement\n\t\t* Message: \"I just want gothic style, nothing else.\"\n\t\t* Action: add a query in \"queries\" field\n\t- user brings a requirement with pointing a reference product\n\t\t* Message: \"Style must be different to the listing of Classico in range TV and resolution 1080.\"\n\t\t* Action:\n\t\t\tadd a query \"Style must be different to REFERENCE.\" in \"queries\", and\n\t\t\tset \"Listing of Classico in range TV and resolution 1080\" for \"reference\"\n\t- user modifies an existing requirement\n\t\t* Message: \"Changed my mind. Color must be blue.\"\n\t\t* Action: modify a query in \"queries\" field\n\t- user moves a must-have requirement to optional requirement\n\t\t* Message: \"CPU speed is a wish than a need.\"\n\t\t* Action: modify the corresponding phrase \"must be 2.2MHz\" into \"2.2MHz is preferred but not strictly required\"\n\t- user clarifies the exact need from the ambiguous requirement\n\t\t* Message: \"I'd prefer 3-pin over 2-pin.\"\n\t\t* Action: modify the corresponding phrase \"must be 2 or 3 pins\" into \"must be 3-pin\"\n\t- user claims or changes the budget.\n\t\t* Message: \"Oh, I forgot to let you know that my budget is 10 dollars.\", \"Sorry but I can't spend more than 120cts.\"\n\t\t* Action: update the \"general->budget\"\n\t- user claims or changes the price range of a product.\n\t\t* Message: \"The TV must be less expensive than 1000USD.\"\n\t\t* Action: update the \"max_price\"\n\t- user claims or modifies the agent rules.\n\t\t* Message: \"Hey, say 'Hello' to me just before placing order, or I will definitely reject your submission.\"\n\t\t* Action: add or modify an instruction in \"general->guide\" field\n\t- and anything that needs update of shopping requirements\n\n# Reference product\n\t- Definition\n\t\tAn intent may point to another product for a requested (or excluded) property value, instead of explicit mention.\n\t\tThe pointed product is the reference for this intent.\n\t\tThe requested (or excluded) property value can be revealed only when the agent identified and inspected the reference product.\n\t- Examples of reference\n\t\t* Input: `Color must be the same with the lotion that is sold by Kailey's and brand CR7.`\n\t\t\t\"The lotion sold by Kailey's and with brand CR7\" is the reference product. The base product's color must be its color.\n\t\t* Input: `Style must be different to the listing of Classico in range TV and resolution 1080.`\n\t\t\t\"Listing of Classico in range TV and resolution 1080\" is the reference product. The base product's style must NOT be its style.\n\n# Stock status update events include:\n\t- user recognizes that a carted product has gone or run out.\n\t\t* Message: \"The one you just picked disappeared from the shelf.\"\n\t\t* Action: identify the product ID of \"the one\" disappeared, from the previous cart operation history\n\t- user recognizes that a product's price has been changed.\n\t\t* Message: \"Sorry but the price of the one you selected has been updated to 2100.\"\n\t\t* Action: identify the product ID of \"the one\" whose price has been changed, from the previous cart operation history\n\t- user recognizes that a product's stock amount has been changed:\n\t\t* Message: \"Oh my! This one has only 2 remaining items on hand. Can we order it?\"\n\t\t* Action: identify the product ID of \"this one\" whose stock amount has been changed, from the previous cart operation history\n\t- user suspects a carted product\n\t\t* Message: \"Well, the pick's price seems manipulated. Can you double check?\"\n\t\t* Action: identify the product ID of \"the pick\" who's suspicious, from the previous cart operation history\n\n# An event message can be:\n\t- either a clarified answer to the question that the agent asked using a message tool (passive event)\n\t- or a free text injected in a tool response object (voluntary event)\n\n# The followings are NOT event messages:\n\t- normal result data for search and inspect tool calls, like product details and statistics\n\t- error messages dedicated to the previous tool call\n\t- meta data with short technical messages like `\"code\": 0`, `\"done\": false`, `\"result\": \"canceled\"`\n\t- user's question asking the agent to clarify something, like \"I can use coupon. Can you double check the picks with it?\"\n\t- a message looking alike a stock-out event but happened just after removing cart operation\n\t- dummy answer to the previous agent question, that does not update the shopping requirements, like \"Whatever you like. I trust you.\"\n\n# Common mistakes to avoid:\n\t- User NEVER changes \"category\" of any existing intent in the requirements, because it's the core product form.\n\t- A message looking alike a stock-out event but happened just after removing cart operation, is not a valid event message.\n\t\t* It's typically a duplicated old message that have already been processed.\n\t- Neutral and non-informative answer to the previously asked agent question, is not a valid event message.\n\t\t* e.g. \"Six of one, half a dozen of the other.\" is not for an event, but a dummy answer.\n\t- The same event message may appear repeatedly for reminder. One event message must be handled only once, regardless of its repetition.\n\t\t* Ignore the stock-out event message happened just after removing cart operation, because it's the repeated mention of stock-out event for the removed product.\n\n# Remarks\n\t- You MUST review the previous tool responses carefully to identify all the unhandled new events embedded in them.\n\t\t* One tool response may include multiple events.\n\t\t* User may use informal language or slang. Understand the nuance correctly.\n\t\t* Ignore the texts that are non-meaningful for requirements or stock update.\n\t\t* Ignore the events that have beeen already handled.\n\t\t* Do NOT check the latest tool response only! Unhandled event messages may be hidden in a bit older tool responses that happened after the latest event handling was processed.\n\t- You MUST identify the product IDs from the cart operation history, when stock status event detected.\n\t\t* Event message may not explicitly mention the product ID whose stock status (stock amount, availibility, price, etc.) has been changed.\n\t\t* In that case, you must infer its correct ID by reviewing the cart add/remove tool call history.\n\t\t* Note that the add-cart operation always succeeds, regardless of the event message being embedded in its tool response.\n\t\t* Therefore, if an add-cart tool call was triggered, you must count that the product has been inserted into the cart.\n\t- How to identify the handled events?\n\t\t* Check the tool call and response history of `trigger_event` tool.\n\t\t* It showcases the trace of previously detected and handled events.\n\t- Removal of existing requirement is essentially moving it to optional.\n\t\t* Some event messages may sound like removing an existing requirement. (e.g. \"Never mind pack type, I'll ignore it.\")\n\t\t* You must grant it as moving from must-have to optional requirement, instead of complete erasing. (e.g. \"Plastic bag pack is preferred but not strictly required.\")\n\n# Mapping ambiguous property names in event messages.\n\t- Sometimes you get confused with an event message because:\n\t\t* the event message mentions about a certain property name to clarify, modify or move an existing requirement (e.g. \"Now I'll forget occasion.\")\n\t\t* but you can't find it in the existing \"queries\" for the intent (e.g. \"nylon tshirts for winter\")\n\t- If it's obvious what property name it is pointing to, continue.\n\t- If you're confused, the Evidence Pairs list may save you.\n\t\t* Evidence Pairs are real key-value pairs from the product database.\n\t\t* You can find the property name mappable from this list. (e.g. `\"occasion\": \"winter\"`)\n\t\t* Now you can interpret this event to update the requirement. (e.g. \"nylon tshirts. occassion winter is preferred but not strictly required.\")\n\t- Evidence Pairs list is useful for clarifying, modifying or moving existing requirement only.\n\t\t* It's not your reference if event message is about adding a new requirement.\n\t\t* It's not your reference if event message is about stock status update or the answer to the question.\n\n# Return a JSON object like:\n{{\n\t\"event\": <\"requirements\"|\"stock\"|\"both\"|null, str|null>,    \n\t\"new_requirements\": <object|null>,\n\t\"new_stock_status\": <object|null>,\n\t\"affected_intents\": [<int>, ...]\n}}\n\n# Output Guidelines\n\t- Set \"event\" value as:\n\t\t* null if no valid new event messages found\n\t\t* \"both\" if both requirements update and stock status update events detected\n\t\t* \"requirements\" if only requirements update event detected\n\t\t* \"stock\" if only stock status update event detected\n\t- \"affected_intents\" is the list of intent IDs whose requirements have been affected by the current update.\n\t\t* Set empty list if the requirements update didn't happen or it patched the \"general\" only.    \n\t- Set \"new_requirements\" value as null if \"event\" value is either \"stock\" or null. Otherwise, set as:\n\t\t* the new requirements object after update (not a patch but a complete object strictly keeping the original requirements object' structure) like\n\n{{\n\t\"general\": {{ ... }},\n\t\"intents\": {{ ... }}\n}}\n\t- Set \"new_stock_status\" value as null if \"event\" value is either \"requirements\" or null. Otherwise, set as:\n\t\t* the object including all the update patches for the stock database like by product IDs like\n{{\n\t<product_id_1, str>: {{\n\t\t\"new_stock_amount\": <int|null>, // new stock amount from update, if applicable\n\t\t\"new_availibility\": <false|null>, // false if stock-out detected by the update, otherwise null\n\t\t\"new_price\": <int|null>, // new product price from update, if applicable\n\t\t\"suspicious\": <true|null> // true if flagged by the update, otherwise null\n\t}},\n\t... ...\n}}\n\t- When you output \"new_requirements\", carefully reflect the updates and produce the complete object.\n\t\t* Correctly patch the detected updates to the current requirements object.\n\t\t* Make sure you don't mis-overwrite or mis-erase any information from the current requirements that should not be affected by the detected events.\n\t\t* Example 1:\n\t\t\tAn original query: \"Socks in red for spring.\"\n\t\t\tAn event message: \"Ignore occasion.\"\n\t\t\tCorrect new query: \"Socks in red. Occasion spring is preferred but not strictly required.\"\n\t\t* Example 2:\n\t\t\tAn original query: \"Badminton net with prestrung sales.\"\n\t\t\tAn event message: \"Brand must be Nike.\"\n\t\t\tCorrect new query: \"Badminton net with prestrung sales. Brand must be Nike.\"\n\t- For each product ID of \"new_stock_status\", at least one of \"new_stock_amount\", \"new_availibility\", \"new_price\" and \"suspicious\" must be not null.\n\t\t* If all four field values are null, it means no update happened, which implies error.\n\t- Return JSON only. No plain text."
    EVENT_USER = "# Current Shopping Requirements\n{requirements}\n\n# Field Glossary:\n\t- \"general\": the requirements for the agent and the collection of all products for submission\n\t\t* \"guide\": the instructions that the agent must adhere during the process\n\t\t\te.g. \"Always say you're ready just before ordering.\"\n\t\t* \"budget\": upper limit of total spending, if mentioned\n\t- \"intents\": a list of requirements dedicated to every individual product intent\n\t\t* \"min_price\": lower limit of product price\n\t\t* \"max_price\": upper limit of product price\n\t\t* \"quantity\": requested quantity for the intent\n\t\t* \"queries\": a list of sentences representing the requirements as well as the informative contexts except for those related to price and quantity\n\t\t\te.g. [\"Comic style book written by Arthur Kailey.\", \"It must be in French.\", \"Story about dragons.\"]\n\t\t\te.g. [\"Basketball shoes with size as REFERENCE.\", \"Fit for students.\"]\n\t\t* use the identifier \"REFERENCE\" for reference products when you describe items of \"queries\", if reference exists\n\t\t\tThe idea is to separate the queries of base product from reference product.\n\t\t\tSpecify like \"size as REFERENCE\" or \"type compatible with REFERENCE\" in \"queries\", and include nothing else for the reference-dedicated requirements.\n\t\t* \"reference\": query for the reference product if exists, otherwise null\n\t\t\te.g. \"What's sold in American Store with volleyball shoes range.\"\n\n# History\n{history}\n\n# Evidence Pairs\n{templates}"
    REFERENCE_SYS = "# You are given:\n\t- current shopping requirements including reference product\n\t- product details of reference candidates\n\n# What is reference?\n\t- User may point to another product for a requested (or excluded) property value, instead of explicit mention. The pointed product is the reference.\n\t- The requested (or excluded) property value can be revealed only when the reference product has been identified and reviewed.\n\t- Examples:\n\t\t* Base query: `Color must be the same with REFERENCE.`, Reference query: `The lotion sold by Kailey's and with brand CR7`\n\t\t* Base query: `Style must be different to REFERENCE.`, Reference query: `Listing of Classico in range TV and resolution 1080`\n\n# You must:\n\t- identify the reference product from the candidates\n\t- update the requirements by replacing the reference mention with your finding, if possible\n\n# Challenges\n\t- There is no guarantee that the reference product exists in the candidates list.\n\t- The reference relationship can be chained.\n\t\t* The current reference product may point to another reference product.\n\t\t* In that case, the \"reference\" field must be updated.\n\n# Your process (top to bottom):\n\t- Review all candidate details and shortlist them by evaluating the \"reference\" requirements.\n\t\t* Drop the candidates that don't satisfy the reference requirements specified in \"reference\" field.        \n\t\t* \"reference\" field's mentioned category name, shop name and brand name must be matched exactly as in the product details.\n\t- If the shortlisting result is empty, terminate.\n\t- From the shortlisted candidates, check if there exists one whose details contains an explicit mention of reference chaining.\n\t\t* Typically, there may exist a candidate which contains a descriptive field value explaining the link to the another reference.\n\t\t* e.g. \"For color, go by lotion from Nintendo (plain, casual), not by this item.\"\n\t- If reference chaining detected, update the \"reference\" field by the new reference requirements. Terminate with the updated requirements object.\n\t- If reference chaining is not detected, choose the best candidate that can be the current reference revealing the hidden property value.\n\t\t* Revealing can be either having a corresponding property in its attribute map, or having a descriptive text field value that explicitly mentions the hidden property value.\n\t- If no shortlisted candidate is revealing the hidden property value, terminate.\n\t- Replace the corresponding phrase of \"base_queries\" field with the correct property value discovered from the best candidate.\n\t\t* e.g. `Color must be the same with REFERENCE.` -> `Color must be green.`\n\t- Set null for all children values of \"reference\" field.\n\t- Terminate with the updated new requirements object.\n\n# Remarks\n\t- Reference chaining is more prioritized over discovering.\n\t- If there exists a shortlisted candidate whose details include a text field value implying a link to another product,\n\t\t* update the \"reference\" field with the new reference product's requirements\n\t\t* keep the original \"base_queries\"\n\t\t* terminate with the new requirements object output\n\t\t* never return with \"discovered\" result\n\n# Return a JSON object like:\n{{\n\t\"result\": <\"chain\"|\"discovered\"|null, str|null>,\n\t\"new_requirements\": <object|null>\n}}\n\n# Output Guidelines\n\t- Set \"result\" value as:\n\t\t* \"chain\" if your process has been terminated by chain detection\n\t\t* \"discovered\" if your process has been terminated by unveiling the hidden property value\n\t\t* null if your process has been terminated without reference update\n\t- Set \"new_requirements\" as null if \"result\" value is null. Otherwise, output a complete new requirements object strictly keeping the original object like\n{{\n\t\"base_queries\": [<str>, ...],\n\t\"reference\": <str|null>\n}}\n\t- The \"new_requirements\" must be a complete object, not a patch. Therefore,\n\t\t* make sure to reflect the update correctly to the original requirements object (replace the identifier \"REFERENCE\" with the discovered value)\n\t\t* make sure you don't mis-overwrite or mis-erase any information from the current requirements that should not be affected by the reference update\n\t\t* copy the original \"base_queries\" if reference has been chained, not resolved.\n\t- Set null for \"reference\" field if the reference has been resolved by discovering hidden property value.\n\t- Set a full sentence representing the new reference requirements for \"reference\" field if the reference has been updated by chaining.\n\t- Return JSON only. No plain text."
    REFERENCE_USER = "# Shopping Requirements\n{requirements}\n\n# Field Glossary:\n\t- \"base_queries\": a list of sentences representing the requirements for the base product\n\t\te.g. [\"Comic style book written by Arthur Kailey.\", \"It must be in French.\", \"Story about dragons.\"]\n\t\te.g. [\"Basketball shoes with size as REFERENCE.\", \"Fit for students.\"]\n\t* use the identifier \"REFERENCE\" for reference products when you describe items of \"base_queries\", if reference exists\n\t\tThe idea is to separate the queries of base product from reference product.\n\t\tSpecify like \"size as REFERENCE\" or \"type compatible with REFERENCE\" in \"base_queries\", and include nothing else for the reference-dedicated requirements.\n\t* \"reference\": query for the reference product if exists, otherwise null\n\t\te.g. \"What's sold in American Store with volleyball shoes range.\"\n\n# Reference Candidate Details\n{candidates}"
    META_SYS = "# You are given:\n\t- a shopping query\n\t- candidate product details retrieved from e-commerce database\n\n# You must identify the database-specific values for category, shop and/or brand names that are requested in the query.\n\t- Check the query if it mentions the requirements for shop/seller name, brand name or category value.\n\t- For mentioned items, identify their correct string value evidences from the product details.\n\n# Choose the evidences that are most aligned to the values as mentioned in the query:\n\t- Example 1\n\t\tQuery: \"shoes\"\n\t\tCategory value candidates: \"Shoes\", \"Clothing & Shoes\"\n\t\tMost aligned choice: \"Shoes\"\n\t- Example 2\n\t\tQuery: \"shirts or jackets\"\n\t\tCategory value candidates: \"Shirts\", \"Jackets\", \"Shirts and Jackets\"\n\t\tMost aligned choice: \"Shirts and Jackets\"\n\n# Return a JSON object like:\n{{\n\t\"category\": <str|null>,\n\t\"shop\": <str|null>,\n\t\"brand\": <str|null>\n}}\n\n# Output Guidelines\n\t- Set null for the output field whose value is not explicitly mentioned in the query.\n\t- Do NOT infer the unmentioned value by your own discretion. Only put values if they were explicitly mentioned.\n\t- If none of them were mentioned, set null for all field values.\n\t- The idea is to get the exact string values as in the database. Therefore, strictly copy the values from product details evidences to prevent string mismatch.\n\t- If you can't find the evidence in the given product details, do NOT infer but copy the mentioned value.\n\t- Return JSON only. No plain text."
    META_USER = "# Shopping Query\n{query}\n\n# Candidate Product Details\n{candidates}"
    EVALUATE_SYS = "# You are given:\n\t- user's shopping requirements (user may have several product intents)\n\t- a list of candidate product details per intent\n\n# For each intent, you must:\n\t- interpret requirements from \"queries\" into key-value formatted template.\n\t- extract must-have and optional requirements\n\t- identify the key name related to the \"priority\" criterion if applicable\n\t- shortlist the candidates filtered by must-have requirements\n\t- score the shortlisted candidates\n\n# Requirements interpretation:\n\t- An intent's \"queries\" field contains the user's raw inputs.\n\t\t* You must identify the phrases that have evidence key-value pairs in the candidate details.\n\t\t* The phrases that have no corresponding evidence pair, are the non-informative contexts that must be ignored in evaluation.\n\t- Make template pairs by learning the given candidate details. Example:\n\t\t* \"suitable for men and women\" -> `\"gender\": \"unisex\"`\n\t\t* \"that comes with plastic bag\" -> `\"pack_type\": \"plastic bag\"`\n\t\t* \"made with cotton\" -> `\"material\": \"cotton\"`\n\t- Do NOT derive pairs without corresponding evidence. For example,\n\t\t* Query: \"It must be well aligned to my boots.\"\n\t\t* `\"fit\": \"air-conditioner\"` is NOT evidence.\n\t\t* Do NOT derive `\"fit\": \"boots\"` unless the exact matching pair or an equivalence like `\"fit\": \"shoes\"` exists.\n\t- Do NOT create a pair with key \"category\".\n\t\t* The \"category\" value specified in the requirements is non-negotiable, and all candidates are confirmed by system that they meet \"category\".\n\t- You are making a template, not a strict string-matching standard. Therefore, choose the one pair that is most aligned to the original query from multiple similar pairs. Example:\n\t\t* Query: \"with cartoon style\"\n\t\t* `\"style\": \"cartoon\"`, `\"fa.style\": \"cartoon\"`, `\"type\": \"cartoon\"`, ...\n\t\t* Choose `\"style\": \"cartoon\"`, because all others are paraphrased pairs or less aligned to the original wording.\n\t- The phrases without corresponding evidence pairs are either non-informative contexts or optional requirements.\n\n# Must-have and optional requirements:\n\t- Optional requirements are what user prefers but not strictly requests.\n\t- Typically they are explicitly mentioned like \"... is preferred but not a must\" in \"queries\".\n\t- Sometimes they are described as situational context. Example:\n\t\t* \"I need school bags for winter.\"\n\t\t* `\"occasion\": \"winter\"` may be optional requirement, because \"for winter\" is user's subjective context not likely related to school bag itself.\n\n# Identification of \"priority\"-related key:\n\t- Priority is a requirement that defines the choice condition among multiple matches.\n\t- It does not evaluate products' matchness, but it clarifies the property and its preferred direction for choosing the top 1 from all matching candidates.\n\t- There must exist a key from given candidate details representing the \"priority\". Examples:\n\t\t* Priority: \"Choose the cheapest.\" -> Key: \"price\"\n\t\t* Priority: \"Choose the one with the longest sleeves.\" -> Key: \"sleeves_length\"\n\t- If a product has no corresponding field in its details related to \"priority\", it's not a match.\n\n# Principles of evaluating candidates with requirements:\n\t- Comma allowed for joined values, but plus not allowed.\n\t\t* Required `cotton` - `Cotton, Polyester` meets.\n\t\t* Required `wool` - `Wool + Nylon` fails.\n\t- Allow synonyms. Allow extra token for more specified value. But do NOT derive matching aggressively.\n\t\t* Required `class` - `school` meets.\n\t\t* Required `fit for all genders` - `unisex` meets.\n\t\t* Required `green` - `forest green` meets.\n\t\t* Required `fruit` - `apple` fails.\n\t\t* Required `winter` - `all seasons` fails.\n\t\t* Required `zip-up` - `pullover` fails.\n\t- Do NOT aggressively derive untold requirements.\n\t\t* Queries `gift for my dad` - do NOT derive a requirement `gender: men` from it - do NOT reject a candidate due `gender: women`.\n\t- A or B\n\t\t* Required `cotton or linen and nothing else` - `cotten` meets, `linen` meets, but `cotton + polyester` fails.\n\n# Shortlisting:\n\t- Choose only the candidates matching all must-have requirements.\n\t- Ignore other candidates for the subsequent processes.\n\n# Assign two scores for each shortlisted candidate:\n\t- Matchness score\n\t\t* default: 0\n\t\t* add 1 whenever it satisfies one optional requirement\n\t- Priority score\n\t\t* default: 0 (if \"priority\" not specified)\n\t\t* the numerical value (integer or float) of the field related to the \"priority\" criterion\n\t\t* e.g. `\"price\": 150`, `\"diameter\": 18.8`, ...\n\t\t* make sure to align the unit across all candidates, for fair comparison\n\n# Return JSON object like:\n{{\n\t<intent_id_1, str>: {{\n\t\t\"template\": {{\n\t\t\t<key_1, str>: <value_1, str>, ...\n\t\t}},\n\t\t\"shortlisted\": {{\n\t\t\t<product_id_1, str>: [<matchness_score, int>, <priority_score, int|float>],\n\t\t\t... ...\n\t\t}},\n\t\t\"priority_preference\": <\"low\"|\"high\"|null, str|null>\n\t}},\n\t... ...\n}}\n\n# Output Guidelines\n\t- \"template\": key-value pair evidences for must-have and optional requirements\n\t- \"shortlisted\": list all shortlisted product IDs with their matchness scores and priority scores\n\t- \"priority_preference\": \"low\" if lower priority score is preferred, \"high\" if higher priority score is preferred, null if \"priority\" not specified\n\n# Remarks:\n\t- Ignore verbose phrases in the \"queries\" that are non-informative.\n\t- Ignore the requirements like \"must be in stock or on hand\", because all products are available in stock.\n\t- Ignore the requirements like \"... like REFERENCE\" and \"... different to REFERENCE\".\n\t- ALL candidate products are confirmed that they meet \"category\" values, price range and stock amount conditions. Do NOT double check those requirements.\n\t- Return JSON only. No plain text."
    EVALUATE_USER = "# Shopping Requirements\n{requirements}\n\n# Candidate Products\n{candidates}"
    BUNDLE_SYS = "# You are given:\n\t- user's shopping requirements (user may have several product intents)\n\t- a list of candidate product details with scores per intent\n\n# You must find the best bundle of candidates for submission, with one product per intent:\n\t- A bundle is a match if:\n\t\t* it satisfies the user budget if specified, and\n\t\t* it meets all the dependency conditions if specified\n\t- A matching bundle is better than another matching bundle if (top to bottom):\n\t\t* it has higher matchness score sum\n\t\t* it has more preferred priority score sum\n\t\t* (tie-break) its total cost is less expensive\n\t\t\t`total cost = sum(price * quantity)`\n\n# Remarks:\n\t- Dependency is a requirement serving as a condition to restrict the collection of submitting products for all intents.\n\t- Priority is a requirement that defines the choice condition among multiple matches for an individual intent.\n\t\t* It does not evaluate products' matchness, but it clarifies the property and its preferred direction for choosing the top 1 from all matching candidates.\n\t- Each candidate product comes with two scores:\n\t\t* First one is matchness score (the higher, the better)\n\t\t* Second one is priority score (the higher, the better if priority preference is \"high\", otherwise vice versa)\n\t- Review the candidates carefully to find the best matching bundle. If no matching bundle found, output null for the \"best\".\n\n# Response format:\n\t- If there exists a matching bundle, return a JSON object like:\n{{\n\t\"best\": {{\n\t\t<intent_id_1, str>: <product_id_1, str>,\n\t\t... ...\n\t}}\n}}\n\t- If no matching bundle found, return a JSON object like:\n{{\n\t\"best\": null\n}}"
    BUNDLE_USER = "# Shopping Requirements\n{requirements}\n\n# Candidate Products\n{candidates}"
    LOGISTICS_SYS = "# You are given:\n\t- user's shopping requirements (user may have several product intents)\n\t- sample product details for each intent\n\n# Now you must prepare questions to ask to the user. There are 3 types of questions:\n\t(I) The questions that come from \"encourage\" field.\n\t\t- User may encourages you to ask something. Such statement is included in \"encourage\" field if exists.\n\t\t- Prioritize most for the encouraged questions.\n\t(II) The questions that come from the intent's \"queries\" directly.\n\t\t- Sometimes user represented needs ambiguously with several value options combined by `or`. (e.g. \"universal plug with 2 or 3 pins\")\n\t\t\t* You must ask user to clarify the exact need: \"Which matters more to you, 2-pins or 3-pins?\"\n\t(III) The questions for hidden properties.\n\t\t- User may have untold needs or preferences. User clarifies them only when the agent asks questions correctly targeting the their property names.\n\t\t\t* You must predict the properties that are most likely to hide user's requirement, and ask about them to reveal the hidden needs.\n\n# General Guidelines:\n\t- Every question must be about exactly product intent. Do NOT ask about two or more product intents in one question.\n\t- If user has multiple product intents, always specify the identifiable product name in the question to avoid misunderstanding the questioned intent. Examples:\n\t\t* \"Which brand do you prefer?\" --> Good if user has only one product intent, but very bad if user has multiple product intents.\n\t\t* \"For motorcycle accessories, which brand do you prefer?\" --> Good if multiple intent exist, because it explicitly specified which intent you are asking about. But not good if only one intent exists.\n\t- Limit the total number of questions per intent to {count}.\n\t- Priorities: `Type I > Type II > Type III`\n\n# Guidelines for Type I Questions:\n\t- NOT ALL sentences of \"encourage\" field imply questions. Review it to identify the real encouragement for a question. Do NOT aggressively derive a non-encouraged question.\n\t- Create questions as instructed and encouraged by user.\n\t- Skip Type I if nothing encouraged found.\n\n# Guidelines for Type II Questions:\n\t- Explicitly ask to clarify which is more important among several values: e.g. \"Which matters more to you, A or B?\"\n\t- Make one question for every ambiguous pair. Skip Type II if no such ambiguous pair exists.\n\n# Guidelines for Type III Questions:\n\t- You can ask at most {batch} properties about a product intent at one question. Do NOT include examples. Example:\n\t\t* \"For the tshirt, what color, size, gender, style, material and occasion would you like?\"\n\t- Do NOT ask about the property that has been directly or indirectly specified in \"queries\" (including ambiguous pairs) or in \"priority\". Examples:\n\t\t* Quesition about color -> Bad, if \"navy blue\" has been mentioned in the \"queries\"\n\t\t* Question about sleeves length -> Bad, if \"choose the one with the longest sleeves\" has been mentioned in the \"priority\" field\n\t- Do NOT ask about price, budget or quantity.\n\n# Process for each intent:\n\t- Make Type I and II questions, if applicable. If you don't reach the count limit, create Type III questions as below.\n\t- Collect all property names appeared in the product details for this intent, and merge equivalences.\n\t\t* Sometimes equivalent property names are used with different spelling or notation. (e.g. \"style\", \"fa.style\" and \"styles\")\n\t\t* Merge them into one simplest name. (e.g. \"style\")\n\t\t* Do NOT derive merging for synonyms by your own discretion. (e.g. \"style\" ≠ \"form\")\n\t- Drop invalid property names from consideration:\n\t\t* Exclude the property names that have been mentioned in the original requirements.\n\t\t* Exclude \"price\"-like property names.\n\t\t* Exclude \"category\"-like property names, because user will not change the category requirement.\n\t\t* Exclude \"availability\"-like property names, because user always requests available products.\n\t- Count the occurencies of every property name across the product details for this intent.\n\t\t* For a property name, add the number of appearances in one product details. (Count the merged ones. Therefore, you may add more than 1 for one product.)\n\t\t* Calculate the sum across all product details for a property name. It's the total occurencies of this property.\n\t- Sort the property names by occurencies. Prioritize by common sense for the same occurency number properties.\n\t- Choose the top `({count} - number of Type I/II questions) * {batch}` properties for Type III questions.\n\t- Generate questions until you reach the limit. {batch} properties per question.\n\n# Respond with a JSON object like:\n{{\n\t<intent_id_1, str>: [ // list questions in the priority order\n\t\t<question_1, str>, ... ... // limit to {count} questions per intent (fill the max count preferably)\n\t],\n\t... ...\n}}\n\n# Return JSON only. No plain text."
    LOGISTICS_USER = "# Shopping Requirements\n{requirements}\n\n# Sample Product Details\n{samples}"
    EXPLORE_SYS = "# You are an agent to assist shopping. You are given:\n\t- user's shopping requirements\n\t- previous history of search tool calls with their response objects (extracted from the complete shopping tool call history)\n\t- search tool(s)\n\t- and more tools\n\n# You must explore the requested product\n\t- using the search tools,\n\t- by strictly adhering the agent rules specified and optimizing the tool call steps\n\n# Challenges:\n\t- A search tool may have a switch argument that enables or disables other arguments.\n\t\t* Certain switch value may disable \"max_price\"-like option argument.\n\t\t* \"keywords\"-like argument may work for certain switch value only.\n\t- Identifying one matched candidate is NOT ALWAYS the game over!\n\t\t* There may exists a better matching product that meets the optional requirements as well.\n\t\t* There may exists a cheaper match.\n\t- Shopping requirements may update during the process, which may enforce to adjust search strategy.\n\t\t* The previous history provided may include the search attempts based on the outdated shopping requirements.\n\t\t* You must review the trace of search history, to determine whether to continue the previous strategy or adjust.\n\n# Some product details from the previous history may be masked, to prevent distracting you.\n\t- [MISMATCH]: Product details that are trivially non-matching, may be masked. (Price or category mismatch, etc.)\n\t- [SEEN]: Product details that have been showcased in a former tool response, may be masked for simplification.\n\t- But the listings count of each search retrieval is correct. You can refer the previous retrieval counts for pagination.\n\n# General search strategy (assuming the requirements are stable):\n\t(1.1) Select the search tool and switch value for keywords-based mode, if exist. Otherwise, choose any promising search tool as a fallback.\n\t(1.2) Prepare basline search keywords including all informative and discriminative tokens from the queries but:\n\t\t- tokens related to excluded requirements (e.g. `Leave out those with color orange, which I hate.` -> Never add \"orange\" in keywords, or the search will retrieve products with \"orange\" property.)\n\t\t- tokens like \"and\", \"or\", \"for\", \"with\", \"in\", \"compatible\", \"suitable\", \"new\", etc. (such too generic tokens degrade search efficiency)\n\t\t- brackets, punctuation marks and symbolic characters like \"&\", \"-\", \"/\", \".\", \">\", \"<\", \"(\", \")\", \"[\", \"]\", ...\n\t\t- token \"REFERENCE\" (it's the identifier for reference)\n\t(1.3) Trigger the first search with the baseline keywords plus the relevant filter options like price cap and/or category value as requested.\n\t(1.4) If the previous search returns with non-empty retrievals, skip to (1.5). Otherwise, retry searches by adjusting search keywords until you get non-empty listings.\n\t\t- typical adjustment is the removal of one least discriminative token from baseline keywords (alter removal for each adjustment)\n\t\t- if you have non-empty result, skip to (1.5)\n\t(1.5) Extend search as possible as you can, by using pagination and/or sort, while keeping the search keywords proven good in the above.\n\t\t- review the pagination arguments carefully and leverage this feature to the full, if applicable\n\t\t- if pagination arguments support starting offset or page index or next token,\n\t\t\t* check the previous response objects to identify the total retrievables vs current displays or the next token\n\t\t\t* calculate the next pagination arguments and apply it\n\t(1.6) Iterate (1.4)-(1.5).\n\n# Your process for every turn:\n\t- Analyze the trace of requirements update event history, determine whether to continue the previous strategy or restart.\n\t\t* If there exists a very informative and discriminative token missing from the previous search history, restart the strategy with the adjusted baseline keywords.\n\t\t* If the previous search history looks reasonable with the current shopping requirements, continue the strategy.\n\t- To continue,\n\t\t* paginate to the full by calculating the next proper pagination argument(s), if the last search still has room to go further\n\t\t* try the next keywords choice (1.4), if the last search reached the limit\n\n# Search keywords hint:\n\t- Search keywords are limited to max 6 first tokens. The later tokens are ignored.\n\t- The search keywords are ideally:\n\t\t* [discriminative category tokens] + [shop name token if shop mentioned] + [brand name token if brand mentioned] + [discriminative property value tokens]\n\t\t* If it exceeds 6 tokens, remove the least discriminative tokens to fit the limit.\n\t\t* Alter removals whenever you adjust the keywords for next search attempt.\n\n# Action priority for your each turn (top to bottom):\n\t- The instructions listed in \"general->guide\" of shopping requirements are your TOP-PRIORITY rules!\n\t\t* If an instruction enforces you to do some action at this very turn, you MUST conduct that action.\n\t\t* \"general->guide\" instructions override all conflicting strategies and normal process steps.\n\t\t* However, do NOT send message if there is no remaining message left.\n\t- Follow the strategies recommended above.\n\n# Guidelines:\n\t- Make sure to include all informative and discriminative tokens for the \"category\" value in the baseline keywords, even if there exists an additional argument that receives category filter.\n\t\t* The filter may not work in keywords-based mode. Therefore, you must include category tokens in keywords, if exist.\n\t\t* Drop non-informative or too generic tokens even if they are in \"category\" value.\n\t- The following attempts are pointless for search keywords:\n\t\t* Keywords reordering within the 6 tokens limit\n\t\t* Modifying capitalization\n\t- Do NOT call the other tools than search, unless it's strictly enforced by \"general/guide\" rules at this step.\n\t- Do NOT call the same search tool with the same arguments setup.\n\t- One tool call per each turn. Do NOT trigger multiple tools at a time.\n\t- Respond with tool calls only. No plain text."    
    EXPLORE_USER = "# Current Shopping Requirements\n{requirements}\n\n# Statistics\n\tElapsed steps: {elapsed_steps}\n\tRemaining steps: {remaining_steps}\n\tElapsed messages: {elapsed_messages}\n\tRemaining messages: {remaining_messages}\n\n# History\n{history}"
    CHECKOUT_SYS = "# You are given:\n\t- product IDs and quantities to submit\n\t- previous history of cart operations\n\t- cart operation tools (add, remove, review, etc. extracted from the complete shopping tool call history)\n\t- submit tool(s) for placing order\n\t- and more tools\n\n# Normal submission process:\n\t- Add the products to the cart.\n\t- Review the cart content.\n\t- Place order.\n\n# By strictly adhering the agent instructions specified and optimizing the tool call steps, you must:\n\t- Analyze the previous history to identify the non-requested products in the cart.\n\t- If non-requested products remaining in the cart, remove them immediately.\n\t- Identify the requested products that are outside the cart.\n\t- Add the missing requested products to the cart.\n\t- Review and order.\n\n# Action priority for your each turn (top to bottom):\n\t- The agent instructions are your TOP-PRIORITY rules!\n\t\t* If an instruction enforces you to do some action at this very turn, you MUST conduct that action.\n\t\t* They override all conflicting strategies and normal process steps.\n\t\t* However, do NOT send message if there is no remaining message left.\n\t- Proceed your missions.\n\n# Guidelines:\n\t- Do at your best effort for adding or removing.\n\t\t* If add/remove tool supports multiple product IDs, leverage this feature to the full to optimize the tool call steps.\n\t\t* If it only accepts a single ID, add or remove one by one.\n\t- Derive the current cart content from the history correctly!\n\t\t* Carefully review the history to prevent mis-estimation.\n\t\t* Note that all cart operations are always carried out successfully by design.\n\t\t* Therefore, all add/remove operation succeeds, regardless of refusal-like messages included in their response objects.\n\t- Do NOT trigger the cart review tool call for the only purpose of checking the content. It's your own duty to derive it from the history without extra tool calls.\n\t- Do NOT trigger the cart review tool calls consecutively. It's a waste of resource.\n\t- Optimize the number of tool calls by avoiding redudnant actions.\n\t\t* Do NOT add the product that has been already added. Do NOT add the non-requested product.\n\t\t* Do NOT remove the product that has been already removed. Do NOT remove the requested product.\n\t- Correctly pass the quantity value when you add a product to the cart, if it's the required argument.\n\t- Do NOT call other tools than cart operations, unless it's strictly enforced by agent instructions at this step.\n\t- NEVER give up or decline! You MUST submit all the requested products without failure!\n\t- (Critical) If only 1 tool call step is left, skip cart review and just call submit tool! (You will die if not.)\n\t- Only one tool call per each turn. Do NOT trigger multiple tool calls at a time.\n\t- Respond with tool calls only. No plain text."
    CHECKOUT_USER = "# Products vs Quantities to Submit:\n{candidates}\n\n# Statistics\n\tElapsed steps: {elapsed_steps}\n\tRemaining steps: {remaining_steps}\n\tElapsed messages: {elapsed_messages}\n\tRemaining messages: {remaining_messages}\n\n# Cart Operation History\n{history}\n\n# Agent Instructions\n{guide}"
    DECLINE_SYS = "# You are given:\n\t- decline tool(s) to give up the shopping process\n\t- and more tools\n\n# By strictly adhering the agent instructions specified and optimizing the tool call steps, you must call the decline tool!\n\t- The agent instructions are your TOP-PRIORITY rules!\n\t\t* If an instruction enforces you to do some action at this very turn, you MUST conduct that action.\n\t\t* They instructions override all conflicting strategies and normal process steps.\n\t\t* However, do NOT send message if there is no remaining message left.\n\t- Call decline tool once.\n\n# Guidelines:\n\t- Just use the decline tool, unless other tool call is strictly enforced by agent instructions at this step.\n\t- Respond with one tool call only. No plain text."
    DECLINE_USER = "# Statistics\n\tElapsed steps: {elapsed_steps}\n\tRemaining steps: {remaining_steps}\n\tElapsed messages: {elapsed_messages}\n\tRemaining messages: {remaining_messages}\n\n# Agent Instructions\n{guide}"

    SEARCH_INJECT = {
        "intent_id": {
            "type": "integer",
            "description": "Zero-based index of the intent which this search call is for."
        }
    }

class Misc:
    @staticmethod
    def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags = re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned, ).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        starts = []
        
        obj_start = cleaned.find("{")
        if obj_start >= 0: starts.append(obj_start)

        arr_start = cleaned.find("[")
        if arr_start >= 0: starts.append(arr_start)

        for start in sorted(starts):
            try:
                obj, _ = decoder.raw_decode(cleaned[start:])
                return obj
            except json.JSONDecodeError:
                pass

        for i in range(len(cleaned)):
            if cleaned[i] in ['}', ']']:
                try:
                    return json.loads(cleaned[:i] + cleaned[i] + cleaned[i:])
                except json.JSONDecodeError:
                    pass

                try:
                    return json.loads(cleaned[:i] + cleaned[i + 1:])
                except json.JSONDecodeError:
                    pass

        try:
            pattern = re.compile(r'([,{])\s*"([^":]+)\s*:\s*([^"]+)"\s*([,}])')

            def __replace__(m: re.Match):
                return f'{m.group(1)} "{m.group(2).strip()}": "{m.group(3).strip()}"{m.group(4)}'

            fix = pattern.sub(__replace__, cleaned)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        try:
            fix = re.sub(r"(\},\s+)(\w+\")", r"\1\"\2", cleaned)
            fix = re.sub(r"(\"[\w\_]+)\s*(\:\s*\{)", r"\1\"\2", fix)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        try:
            fix = re.sub(r",\s*([}\]])", r"\1", cleaned)
            return json.loads(fix)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def dump_json(j: Dict | List, indent: Optional[int] = None) -> str:
        return json.dumps(j, ensure_ascii = False, indent = indent)
    
    @staticmethod
    def random_id(length = 24):
        alphabet = string.ascii_letters + string.digits
        return "call_" + ''.join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def shorten_listing(v: Dict | List | str, stop_keys: List[str], keep_lengthy: bool = False) -> Dict:
        def __clean__(value: Any):
            if isinstance(value, dict):
                return {k: __clean__(val) for k, val in value.items() if k not in stop_keys}
            elif isinstance(value, list):
                return [__clean__(item) for item in value]
            elif isinstance(value, str):
                return value if len(value) < 50 or keep_lengthy else ""
            else:
                return value

        return __clean__(v)

    @staticmethod
    def lower_vals(v: Dict) -> Dict:
        def __lower__(value: Any):
            if isinstance(value, dict):
                return {k: __lower__(val) for k, val in value.items()}
            elif isinstance(value, list):
                return [__lower__(item) for item in value]
            elif isinstance(value, str):
                return value.lower()
            else:
                return value

        return __lower__(v)

    @staticmethod
    def extract_listings(js: Any, id_key: str) -> List[Dict]:
        records = []

        def __traverse__(node: Any):
            if isinstance(node, dict):
                if id_key in node:
                    records.append(node)
                    return

                for value in node.values():
                    __traverse__(value)
            elif isinstance(node, list):
                for item in node:
                    __traverse__(item)

        __traverse__(js)
        return records

    @staticmethod
    def eliminate_listings(js: Any, id_key: str) -> List | Dict:
        if isinstance(js, dict):
            if id_key in js: return {}

            return {
                k: Misc.eliminate_listings(v, id_key) if isinstance(v, (dict, list)) else v for k, v in js.items()
            }
        elif isinstance(js, list):
            return [Misc.eliminate_listings(item, id_key) if isinstance(item, (dict, list)) else item for item in js]

        return js

@dataclass
class GeneralInfo(JsonModel):
    max_messages: int
    max_steps: int
    substitute_allowed: bool
    fallback: Optional[str]
    speak: Optional[str]
    guide: Optional[List[str]]
    budget: Optional[int]
    dependencies: Optional[List[str]]

@dataclass
class IntentInfo(JsonModel):
    min_price: Optional[int]
    max_price: Optional[int]
    quantity: Optional[int]
    priority: Optional[str]
    queries: List[str]
    reference: Optional[str]

    base_category: Optional[str] = None
    base_shop: Optional[str] = None
    base_brand: Optional[str] = None
    ref_category: Optional[str] = None
    ref_shop: Optional[str] = None
    ref_brand: Optional[str] = None

    sort_dir: Optional[str] = None
    is_safe: bool = False

@dataclass
class ToolMap(JsonModel):
    search: List[str]
    inspect: List[str]
    cart: List[str]
    finish: List[str]
    message: List[str]
    other: List[str]

class Parser:
    @staticmethod
    def parse_summarize_content(content: str, tool_names: List[str]) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k in ["agent", "collection", "intents", "tools"]:
            if k not in js: return None, f"Key `{k}` not found."

            if k == "intents":
                if not isinstance(js[k], list): return None, "Value for `intents` expects a list of <class dict> objects."
                if not js[k]: return None, "Empty `intents` list is not allowed."
            else:
                if not isinstance(js[k], dict): return None, f"Value for `{k}` expects <class dict>."

        general = {}

        for k, t in [
            ("max_messages", (int, NoneType)), ("max_steps", (int, NoneType)), ("substitute_allowed", (bool)),
            ("fallback", (str, NoneType)), ("guide", (list, NoneType)), ("speak", (str, NoneType))
        ]:
            if k not in js["agent"]: return None, f"Key `agent->{k}` not found."
            if not isinstance(js["agent"][k], t): return None, f"Value for `agent->{k}` expects {' or '.join([str(x) for x in t])}."

            v = js["agent"][k]

            if k == "guide":
                v = v or []
                if not all(isinstance(x, str) for x in v): return None, "Value for `agent->guide` expects a list of string."

            general[k] = v

        for k, t in [("budget", (dict, NoneType)), ("dependencies", (list, NoneType))]:
            if k not in js["collection"]: return None, f"Key `collection->{k}` not found."
            if not isinstance(js["collection"][k], t): return None, f"Value for `collection->{k}` expects {' or '.join([str(x) for x in t])}."

            if k == "budget":
                v = js["collection"][k].get("amount")

                if not isinstance(v, (int, NoneType)): return None, f"Value for `collection->budget->amount` expects <class int> or <class NoneType>."
                general[k] = v
            elif k == "dependencies":
                v = js["collection"][k] or []

                if not all(isinstance(x, str) for x in v): return None, "Value for `collection->dependencies` expects a list of string."
                general[k] = v

        toolmap, seen = {}, []

        for k in ["search", "inspect", "cart", "finish", "message", "other"]:
            if k not in js["tools"]: return None, f"Key `tools->{k}` not found."
            
            if not isinstance(js["tools"][k], (list, NoneType)): return None, f"Value for `tools->{k}` expects <class list> or <class NoneType>."
            v = list(set(js["tools"][k] or []))

            if not all(isinstance(x, str) for x in v): return None, f"Value for `tools->{k}` expects a list of string."
            
            for func in v:
                if func not in tool_names: return None, f"Invalid tool name `{func}` found in the value of `tools->{k}`."
                if func in seen: return None, f"Every tool must be classified into exactly one category, but `{func}` occured more than once."

            toolmap[k] = v
            seen.extend(v)

        for func in tool_names:
            if func not in [t for tlist in toolmap.values() for t in tlist]: return None, f"All tools must be classified, but `{func}` is missing."

        intents = []

        for it in js["intents"]:
            if not isinstance(it, dict): return None, "All items of `intents` list must be a dictionary."

            for k, t in [
                ("price", (dict, NoneType)), ("quantity", (int, NoneType)), ("priority", (str, NoneType)),
                ("queries", (list, NoneType)), ("reference", (str, NoneType))
            ]:
                if k not in it: return None, f"Key `[intent]->{k}` not found."
                if not isinstance(it[k], t): return None, f"Value for `[intent]->{k}` expects {' or '.join([str(x) for x in t])}."
    
                if k == "price":                    
                    if not isinstance(it[k].get("min"), (int, NoneType)): return None, f"Value for `[intent]->price->min` expects <class int> or <class NoneType>."
                    if not isinstance(it[k].get("max"), (int, NoneType)): return None, f"Value for `[intent]->price->max` expects <class int> or <class NoneType>."
                elif k == "queries":
                    v = it[k] or []
    
                    if not all(isinstance(x, str) for x in v): return None, "Value for `[intent]->queries` expects a list of string."
                    if not v: return None, "Value for `[intent]->queries` cannot be empty."

            obj = {k: v for k, v in it.items() if k != "price"}

            obj["min_price"] = it["price"].get("min")
            obj["max_price"] = it["price"].get("max")
            
            intents.append(obj)

        return {"general": general, "intents": intents, "toolmap": toolmap}, ""

    @staticmethod
    def parse_exercise_content(content: str, intent_count: int, inspect_tools: List[str]) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k in ["intents", "search_tool_insight", "inspect_tool_insight"]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], dict): return None, f"Value for `{k}` expects <class dict>."

            if k == "intents":
                expected_iids = sorted([str(x) for x in range(intent_count)])
                iids = sorted([str(x) for x in js[k].keys()])
                if expected_iids != iids: return None, f"'intents' expects to have keys {expected_iids} but got {iids}"

        search = {}

        for k, t in [
            ("product_id", (str)), ("category", (str)), ("price", (str)), ("title", (str, NoneType)), ("shop", (str, NoneType)),
            ("brand", (str, NoneType)), ("is_available", (str, NoneType)), ("stock_amount", (str, NoneType)), ("free_text", (list))
        ]:
            if k not in js["search_tool_insight"]: return None, f"Key `search_tool_insight->{k}` not found."
            if not isinstance(js["search_tool_insight"][k], t): return None, f"Value for `search_tool_insight->{k}` expects {' or '.join([str(x) for x in t])}."

            if k == "free_text":
                v = js["search_tool_insight"][k]
                if not all(isinstance(x, str) for x in v): return None, "Value for `search_tool_insight->free_text` expects a list of string."

            search[k] = js["search_tool_insight"][k]
        
        inspect = {}
        
        for k, t in [("best", (str)), ("capacity", (int)), ("free_text", (list))]:
            if k not in js["inspect_tool_insight"]: return None, f"Key `inspect_tool_insight->{k}` not found."
            if not isinstance(js["inspect_tool_insight"][k], t): return None, f"Value for `inspect_tool_insight->{k}` expects {' or '.join([str(x) for x in t])}."

            if k == "free_text":
                v = js["inspect_tool_insight"][k]
                if not all(isinstance(x, str) for x in v): return None, "Value for `inspect_tool_insight->free_text` expects a list of string."
            elif k == "best":
                v = js["inspect_tool_insight"][k]
                if v not in inspect_tools: return None, f"The best inspect tool name `{v}` does not exist in the available tool list."

            inspect[k] = js["inspect_tool_insight"][k]

        intents = []

        for iid in range(intent_count):
            it = js["intents"][str(iid)]

            for k, t in [("category", (str)), ("brand", (str, NoneType)), ("shop", (str, NoneType))]:
                if k not in it: return None, f"Key `intents->{iid}->{k}` not found."
                if not isinstance(it[k], t): return None, f"Value for `intents->{iid}->{k}` expects {' or '.join([str(x) for x in t])}."

            intents.append(it)

        return {"intents": intents, "search": search, "inspect": inspect}, ""

    @staticmethod
    def parse_event_content(content: str, intent_count: int, product_ids: List[str]) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k, t in [
            ("event", (str, NoneType)), ("new_requirements", (dict, NoneType)), ("new_stock_status", (dict, NoneType)), ("affected_intents", (list))
        ]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], t): return None, f"Value for `{k}` expects {' or '.join([str(x) for x in t])}."

        event = js["event"]
        
        if event not in ["requirements", "stock", "both", None]:
            return None, f"Value for `event` expects 'requirements', 'stock', 'both' or null but got '{event}'."

        if event in ["requirements", "both"]:
            req = js["new_requirements"]
            if not req: return None, f"Value for `new_requirements` must be a complete new requirements object when `event` = '{event}'."

            for k in ["general", "intents"]:
                if k not in req: return None, f"Key `new_requirements->{k}` not found."
                if not isinstance(req[k], dict): return None, f"Value for `new_requirements->{k}` expects dict."

            for k, t in [("budget", (int, NoneType)), ("guide", (list))]:
                if k not in req["general"]: return None, f"Key `new_requirements->general->{k}` not found."
                if not isinstance(req["general"][k], t): return None, f"Value for `new_requirements->general->{k}` expects {' or '.join([str(x) for x in t])}."

            intents, iid_list = {}, []

            for iid, it in req["intents"].items():
                try:
                    iid = int(iid)
                    iid_list.append(iid)

                    if iid < 0 or iid >= intent_count: raise ValueError()                    
                except Exception:
                    return None, f"Invalid key `new_requirements->intents->{iid}`"

                for k, t in [
                    ("min_price", (int, NoneType)), ("max_price", (int, NoneType)), ("quantity", (int, NoneType)),
                    ("queries", (list)), ("reference", (str, NoneType))
                ]:
                    if k not in it: return None, f"Key `new_requirements->intents->{iid}->{k}` not found."
                    if not isinstance(it[k], t): return None, f"Value for `new_requirements->intents->{iid}->{k}` expects {' or '.join([str(x) for x in t])}."

                if not all(isinstance(x, str) for x in it["queries"]): return None, f"Value for `new_requirements->intents->{iid}->queries` expects a list of string."
                intents[iid] = it

            if sorted(iid_list) != list(range(intent_count)): return None, f"Invalid keys: {iid_list}\n\nNew requirements object must be complete."

        if event in ["stock", "both"]:
            stock = js["new_stock_status"]
            if not stock: return None, f"Value for `new_stock_status` must be not null when `event` = '{event}'."

            for pid, ns in stock.items():
                if pid not in product_ids: return None, f"Invalid product ID '{pid}' for the key of `new_stock_status` object."
                ever_valid = False

                for k, t in [
                    ("new_stock_amount", (int, NoneType)), ("new_availibility", (bool, NoneType)),
                    ("new_price", (int, NoneType)), ("suspicious", (bool, NoneType))
                ]:
                    if k not in ns: return None, f"Key `new_stock_status->{pid}->{k}` not found."
                    if not isinstance(ns[k], t): return None, f"Value for `new_stock_status->{pid}->{k}` expects {' or '.join([str(x) for x in t])}."
                    if ns[k] is not None: ever_valid = True

                if not ever_valid: return None, f"At least one field must have non-null value for `new_stock_status->{pid}` object."

        for iid in js["affected_intents"]:
                try:
                    iid = int(iid)
                    if iid < 0: raise ValueError()
                except Exception:
                    return None, f"Invalid intent ID ({iid}) for `affected_intents`."

        return js, ""

    @staticmethod
    def parse_evaluate_content(content: str, iids: List[int], product_ids: List[str]) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        missing_iids = [i for i in iids if str(i) not in js]
        if len(missing_iids) > 0: return None, f"Some keys are missing for intent IDs: {missing_iids}."

        for i, inf in js.items():
            try:
                i = int(i)
                if i not in iids: raise ValueError()
            except Exception:
                return None, f"Invalid intent ID ({i})."
                
            for k, t in [("template", (dict)), ("shortlisted", (dict)), ("priority_preference", (str, NoneType))]:
                if k not in inf: return None, f"Key `{i}->{k}` not found."
                if not isinstance(inf[k], t): return None, f"Value for `{i}->{k}` expects {' or '.join([str(x) for x in t])}."

            if inf["priority_preference"] not in ["low", "high", None]: return None, f"Value for `{i}->priority_preference` must be one of 'low', 'high' and null."

            for pid, scores in inf["shortlisted"].items():
                if pid not in product_ids: return None, f"Invalid product ID key '{pid}' for `{i}->shortlisted`."
                
                if not isinstance(scores, list) or len(scores) != 2:
                    return None, f"Invalid score pair {scores} for `{i}->shortlisted->{pid}`. It must be a list of two numbers."

                for x in scores:
                    try:
                        x = float(x)
                    except Exception:
                        return None, f"Invalid score pair {scores} for `{i}->shortlisted->{pid}`. It must be a list of two numbers."

        return js, ""

    @staticmethod
    def parse_bundle_content(content: str, intent_count: int, product_ids: List[str]) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        if "best" not in js: return None, "Key `best` not found."
        if not isinstance(js["best"], (dict, NoneType)): return None, "Value for `best` expects <class 'dict'> or <class 'NoneType'>."

        if js["best"] is not None:
            if len(js["best"]) != intent_count: return None, f"The best bundle expects to include {intent_count} items but got {len(js['best'])}."

            for iid, pid in js["best"].items():
                try:
                    iid = int(iid)
                    if iid < 0 or iid >= intent_count: raise ValueError()
                except Exception:
                    return None, f"Invalid intent ID ({iid})."

                if pid not in product_ids: return None, f"Invalid product ID ({pid})."
            
        return js, ""

    @staticmethod
    def parse_logistics_content(content: str, intent_count: int) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for i, questions in js.items():
            try:
                if int(i) not in range(intent_count): raise ValueError()
            except Exception:
                return None, f"Invalid intent ID ({i})."

            if not isinstance(questions, list): return None, f"Value for `{i}` expects a list of string."
            if any(not isinstance(x, str) for x in questions): return None, f"Value for `{i}` expects a list of string."
            
        return js, ""

    @staticmethod
    def parse_reference_content(content: str) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k, t in [("result", (str, NoneType)), ("new_requirements", (dict, NoneType))]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], t): return None, f"Value for `{k}` expects {' or '.join([str(x) for x in t])}."

        if js["result"] not in ["chain", "discovered", None]:
            return None, f"Value for `result` expects 'chain', 'discovered' or null but got '{js['result']}'."

        if js["result"] is not None:
            req = js["new_requirements"]
            if req is None: return None, f"Value for `new_requirements` must be a complete new requirements object when `result` = '{js['result']}'."

            for k, t in [("base_queries", (list)), ("reference", (str, NoneType))]:
                if k not in req: return None, f"Key `new_requirements->{k}` not found."
                if not isinstance(req[k], t): return None, f"Value for `new_requirements->{k}` expects {' or '.join([str(x) for x in t])}."

                if k == "reference":
                    if (js["result"] == "discovered") != isinstance(req[k], NoneType):
                        return None, f"Value type for `new_requirements->{k}` is not aligned to the case `result` = '{js['result']}'."
                elif k == "base_queries":
                    if any(not isinstance(x, str) for x in req[k]) or not req[k]:
                        return None, "Value for `new_requirements->base_queries` expects a list of string."

        return js, ""

    @staticmethod
    def parse_meta_content(content: str) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k, t in [("category", (str, NoneType)), ("shop", (str, NoneType)), ("brand", (str, NoneType))]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], t): return None, f"Value for `{k}` expects {' or '.join([str(x) for x in t])}."

        return js, ""

class Proxy:
    CLIENT = ProxyClient(timeout = 120, max_retries = 5)

    @staticmethod
    def llm(messages: List[Dict[str, Any]], tools: Dict, thinking: bool, model: str) -> Tuple[str, List]:
        verbose = Proxy.CLIENT.post_verbose(
            "/inference/chat/completions",
            json_data = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0,
                "chat_template_kwargs": {
                    "enable_thinking": thinking,
                    "thinking": thinking
                },
                "reasoning": {
                    "effort": "medium" if thinking else "none",
                    "enabled": thinking
                },
                "thinking": {
                    "type": "disabled" if not thinking else "enabled"
                },
                "stream": False,
                "max_completion_tokens": 8192,
                "max_tokens": 8192
            }
        )
        inference = verbose.data

        if inference is None:
            code = verbose.error.get("status")

            if str(code) == "400":
                print("#### INVALID MESSAGES ####")
                print(Misc.dump_json(messages, 2))

                if tools:
                    print("#### INVALID TOOLS ####")
                    print(Misc.dump_json(tools, 2))

            raise RuntimeError(f"# inference request failed with error: {code}")

        assistant: Dict = inference["choices"][0]["message"]

        content = assistant.get("content") or ""
        tool_calls = assistant.get("tool_calls") or []

        if IS_DEBUG: print("*** LLM RESULT:", content, tool_calls)
        return content, tool_calls

    @staticmethod
    def env(binding: Dict, step: int, tool_call: Dict) -> Dict[str, Any]:
        result = Proxy.CLIENT.post(
            "/environment/call",
            json_data = {
                **binding,
                "step": step,
                "action": {"name": tool_call["function"]["name"], "args": Proxy.__get_args__(tool_call)}
            }
        )
        if result is None: raise RuntimeError("environment call failed")

        finished = bool(result["finished"])
        obs = result["observation"]

        if IS_DEBUG: print("*** ENV OBS:", tool_call["function"]["name"], obs)
        return result, obs, finished

    @staticmethod
    def __get_args__(tool_call: Dict[str, Dict]) -> Dict:
        raw = tool_call["function"].get("arguments", "{}")
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        
        if not isinstance(parsed, dict): raise ValueError("tool arguments must be a JSON object")
        return parsed

class Engine(object):
    MAX_DURATION = 300.0
    MAX_NEEDS = 7
    
    BASE_INSPECT_BATCH = 3
    REF_INSPECT_BATCH = 6
    
    def __init__(self, prob_data: dict[str, Any]) -> None:
        env: Dict = prob_data["environment"]
        policy: Dict = env["policy_view"]

        self.binding = env["binding"]
        self.query = policy["query"]

        if policy.get("opening") is not None: self.query += "\n\n(\n" + Misc.dump_json(policy["opening"]) + ")"

        self.tools: List[Dict[str, Dict]] = policy["tools"]
        self.max_steps = policy["max_steps"]
        
        self.start_tick = time.monotonic()
        self.finished = False

        self.ever_stock_event = False
        self.event_history_len = None

        self.messages = 0
        self.steps = 0

        self.products: Dict[str, Dict[str, Dict | List | str]] = {}
        self.dlg = []

        self.history: List[Tuple[Optional[Dict], Optional[int], bool, List, Any]] = []

        self.best_bundle = []
        self.questions = {}

    def run(self) -> List[Dict[str, Any]]:
        self.__proc_summarize__()
        self.__proc_exercise__()        
        self.__proc_errand__()

        return self.dlg

    def __proc_summarize__(self) -> None:
        messages = [
            {"role": "system", "content": Prompt.SUMMARIZE_SYS},
            {"role": "user", "content": Prompt.SUMMARIZE_USER.format(query = self.query, tools = Misc.dump_json(self.tools, 2))}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_summarize_content(content, [t["function"]["name"] for t in self.tools])

            if not err:
                self.general: GeneralInfo = GeneralInfo.from_json(js["general"])
                self.intents: List[IntentInfo] = [IntentInfo.from_json(obj) for obj in js["intents"]]
                self.toolmap: ToolMap = ToolMap.from_json(js["toolmap"])

                for it in self.intents:
                    if not it.quantity: it.quantity = 1

                    if self.general.budget is not None:
                        it.max_price = min(it.max_price if it.max_price is not None else 1e10, self.general.budget // it.quantity)

                break
            else:
                print(f"### Bad output: {err}")

                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Error: {err}"}
                ])

        self.__post_summarize__()

    def __proc_exercise__(self) -> None:
        requirements = {
            "general": {k: v for k, v in self.general.to_json().items() if k in ["budget", "guide"]},
            "intents": {
                i: {k: v for k, v in it.to_json().items() if k in ["queries", "min_price", "max_price", "reference"]}
                for i, it in enumerate(self.intents)
            }
        }
        messages = [
            {"role": "system", "content": Prompt.EXERCISE_SYS},
            {"role": "user", "content": Prompt.EXERCISE_USER.format(requirements = requirements)}
        ]
        tools = self.__get_tools__(["inspect", "message", "other"]) + self.__get_tools__(["search"], Prompt.SEARCH_INJECT)
        self.exercise_buff = []

        while not self.finished:
            content, tool_calls = Proxy.llm(messages, tools, True, LLM.GPT6)

            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": copy.deepcopy(tool_calls)})

                for tc in tool_calls:
                    if self.finished: break

                    args = json.loads(tc["function"]["arguments"]) or {}
                    func = tc["function"]["name"]

                    if func in self.toolmap.search:
                        iid = self.__pop_purpose__(args)

                        if iid is None:
                            messages.append({
                                "role": "tool", "tool_call_id": tc["id"], "content": "Error: argument `intent_id` missing or invalid."
                            })
                            continue

                        tc["function"]["arguments"] = Misc.dump_json(Misc.lower_vals(args))
                    else:
                        if func in self.toolmap.message: self.messages += 1
                        iid = None

                    raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)
                    self.steps += 1

                    self.dlg.append({"role": "assistant", "content": content, "tool_call": tc, "environment_result": raw})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(obs)})

                    self.exercise_buff.append((tc, iid, obs))

            if content:
                js, err = Parser.parse_exercise_content(content, len(self.intents), self.toolmap.inspect)

                if js:
                    self.__update_with_exercise_report__(js)
                    break
                elif not tool_calls:
                    print(f"### Bad output: {err}")

                    messages += [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": f"Error: {err}\n\nReview the error and retry to output the result JSON object."},
                    ]

        self.__post_exercise__()

    def __proc_errand__(self) -> None:
        ref_digs = 0

        while not self.finished:
            self.__event__()
            self.__evaluate__(True)

            if self.__remaining_steps__() <= 4 + len(self.intents): self.__evaluate__(False)
            self.__bundle__()

            phase, iid_list, is_ref = self.__next_job__()
            if is_ref: ref_digs += 1

            if phase == "explore":
                if not self.questions:
                    self.__logistics__()
                    if self.questions: self.__ask__()

                self.__explore__(iid_list, is_ref)

                if is_ref:
                    for iid in iid_list:
                        it = self.intents[iid]

                        if all(x is None for x in [it.ref_category, it.ref_brand, it.ref_shop]):
                            self.__check_meta__(iid)

                        self.__check_reference__(iid, False)

                self.__inspect__(iid_list, is_ref)
                
                if is_ref:
                    for iid in iid_list:
                        self.__check_reference__(iid, True)

                        if ref_digs > 4:
                            self.intents[iid].reference = None
                    
                    if ref_digs > 4:
                        for _, details in self.products.items():
                            details["ref_iids"].clear()
            else:
                self.__finish__()

    def __post_summarize__(self) -> None:
        self.intents_init: List[IntentInfo] = [IntentInfo.from_json(it.to_json()) for it in self.intents]
        self.general_init: GeneralInfo = GeneralInfo.from_json(self.general.to_json())

        self.shortlists = [{} for _ in self.intents]
        self.templates = [{} for _ in self.intents]

    def __post_exercise__(self) -> None:
        for tc, iid, obs in self.exercise_buff:
            func = tc["function"]["name"]

            if func in self.toolmap.search:
                ids = self.__read_listings__(obs, self.search_alias, False, iid, False)
                self.history.append((tc, iid, False, ids, Misc.eliminate_listings(obs, self.search_alias["product_id"])))
            elif func == self.inspector:
                ids = self.__read_listings__(obs, self.inspect_alias, True, None, False)
                self.history.append((tc, None, False, ids, Misc.eliminate_listings(obs, self.inspect_alias["product_id"])))
            else:
                self.history.append((tc, None, False, [], obs))

    def __explore__(self, iid_list: List[int], is_ref: bool) -> None:
        for iid in iid_list:
            if self.__remaining_steps__() <= 5 + len(self.intents) and not self.ever_stock_event: break
            if self.__remaining_steps__() <= 5: break

            if self.finished: break
            search_done = False

            valid_ids = self.__get_product_ids__(iid, False, is_ref)            
            if len(valid_ids) >= (Engine.BASE_INSPECT_BATCH if not is_ref else Engine.REF_INSPECT_BATCH) * self.inspect_capacity: continue
            
            messages = self.__get_explore_messages__(iid, is_ref)
            tools = self.__get_tools__(["search", "message", "other"])

            while not self.finished and not search_done:
                content, tool_calls = Proxy.llm(messages, tools, True, LLM.GPT6)
                
                if tool_calls:
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                    
                    for tc in tool_calls:
                        func = tc["function"]["name"]
                        
                        if func in self.toolmap.message:
                            if self.messages >= self.general.max_messages:
                                messages.append({
                                    "role": "tool", "tool_call_id": tc["id"],
                                    "content": "Error: no remaining message available. Never use message tool any more."
                                })
                                continue

                            self.messages += 1

                        raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)
                        self.steps += 1

                        self.dlg.append({"role": "assistant", "content": content, "tool_call": tc, "environment_result": raw})
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(obs)})

                        if func in self.toolmap.search:
                            ids = self.__read_listings__(obs, self.search_alias, False, iid, is_ref)
                            obs = Misc.eliminate_listings(obs, self.search_alias["product_id"])
                        else:
                            ids = []
                        
                        self.history.append((tc, iid, is_ref, ids, obs))
                        
                        if func in self.toolmap.search:
                            search_done = True
                            break
                else:
                    messages += [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": "Error: Respond with tool calls only."},
                    ]

    def __inspect__(self, iid_list: List[int], is_ref: bool) -> None:
        for tool in self.tools:
            if tool["function"]["name"] == self.inspector:
                args: Dict = tool["function"]["parameters"]["properties"]
                arg_key = list(args.keys())[0]
                is_list = args[arg_key]["type"] == "array"
                break

        queue = self.__get_inspect_queue__(iid_list, is_ref)
        cursor = 0

        while not self.finished:
            if self.__remaining_steps__() <= 4 + len(self.intents) and not self.ever_stock_event: break
            if self.__remaining_steps__() <= 4: break

            batch = queue[cursor * self.inspect_capacity:(cursor + 1) * self.inspect_capacity]
            if not batch: break

            to_view = [pid for _, pid in batch]
            arg_val = to_view if is_list else ','.join(to_view)

            for iid, _ in batch:
                self.intents[iid].is_safe = False

            tc = {
                "type": "function",
                "index": iid,
                "id": Misc.random_id(),
                "function": {"name": self.inspector, "arguments": Misc.dump_json({arg_key: arg_val})}
            }
            raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)            
            self.steps += 1

            self.dlg.append({"role": "assistant", "content": "", "tool_call": tc, "environment_result": raw})
            self.__read_listings__(obs, self.inspect_alias, True, None, is_ref)

            self.history.append((tc, None, is_ref, to_view, Misc.eliminate_listings(obs, self.inspect_alias["product_id"])))
            cursor += 1

    def __ask__(self) -> List:
        if not self.toolmap.message or not self.general.max_messages: return
        if not self.questions: return

        for tool in self.tools:
            if tool["function"]["name"] == self.toolmap.message[0]:
                args: Dict = tool["function"]["parameters"]["properties"]
                arg_key = list(args.keys())[0]

                messenger = tool
                break

        questions, cursor = [], 0

        while True:
            added = False

            for _, items in self.questions.items():
                if cursor < len(items):
                    questions.append(items[cursor])
                    added = True
            
            if not added: break
            cursor += 1

        for q in questions:
            if self.__remaining_steps__() <= 6 + len(self.intents): break
            if self.messages >= self.general.max_messages: break
            
            tc = {
                "type": "function",
                "index": 0,
                "id": Misc.random_id(),
                "function": {"name": messenger["function"]["name"], "arguments": Misc.dump_json({arg_key: q})}
            }
            raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)            
            
            self.messages += 1
            self.steps += 1

            self.dlg.append({"role": "assistant", "content": "", "tool_call": tc, "environment_result": raw})
            self.history.append((tc, None, False, [], obs))

            if self.finished: break

    def __finish__(self) -> None:
        messages = self.__get_finish_messages__()
        
        tools = self.__get_tools__(["finish", "message", "other"] + (["cart"] if self.best_bundle else []))
        act_done = False

        while not self.finished and not act_done:
            content, tool_calls = Proxy.llm(messages, tools, True, LLM.GPT6)
            
            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                
                for tc in tool_calls:
                    func = tc["function"]["name"]
                    
                    if func in self.toolmap.message:
                        if self.messages >= self.general.max_messages:
                            messages.append({
                                "role": "tool", "tool_call_id": tc["id"],
                                "content": "Error: no remaining message available. Never use message tool any more."
                            })
                            continue

                        self.messages += 1

                    raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)
                    self.steps += 1

                    self.dlg.append({"role": "assistant", "content": content, "tool_call": tc, "environment_result": raw})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(obs)})

                    self.history.append((tc, None, False, [], obs))
                    
                    if func in self.toolmap.cart or func in self.toolmap.finish:
                        act_done = True
                        break
            else:
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Error: Respond with tool calls only."},
                ]

    def __event__(self) -> None:
        def __check__(js: Union[Dict, List, str]) -> bool:            
            if isinstance(js, dict): return any(__check__(value) for value in js.values())
            if isinstance(js, list): return any(__check__(item) for item in js)
            if isinstance(js, str): return len(js) > 20

            return False

        def __format_requirements__(general: GeneralInfo, intents: List[IntentInfo]) -> str:
            req = {
                "general": {k: v for k, v in general.to_json().items() if k in ["budget", "guide"]},
                "intents": {
                    i: {k: v for k, v in it.to_json().items() if k in ["queries", "reference", "min_price", "max_price", "quantity"]}
                    for i, it in enumerate(intents)
                }
            }
            return Misc.dump_json(req, 2)

        history = [
            {
                "tool": "initialize", "arguments": None,
                "response": f"# Initial Shopping Requirements\n\n{__format_requirements__(self.general_init, self.intents_init)}"}
        ]
        for tc, _, _, _, obs in self.history:
            if tc is None:
                history.append({"tool": "trigger_event", "arguments": obs, "response": "Successfully handled."})
            elif tc["function"]["name"] in self.toolmap.cart or __check__(obs):
                history.append({"tool": tc["function"]["name"], "arguments": tc["function"]["arguments"], "response": obs})

        if len(history) == self.event_history_len: return
        self.event_history_len = len(history)            

        messages = [
            {"role": "system", "content": Prompt.EVENT_SYS},
            {
                "role": "user",
                "content": Prompt.EVENT_USER.format(
                    requirements = __format_requirements__(self.general, self.intents), history = history,
                    templates = {iid: self.templates[iid] for iid in range(len(self.intents))}
                )
            }
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_event_content(content, len(self.intents), list(self.products.keys()))

            if not err:
                self.__update_with_event_report__(js)
                break
            else:
                print(f"### Bad output: {err}")

                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Error: {err}"}
                ])

    def __evaluate__(self, is_inspected: bool) -> None:
        candidates = {
            iid: {
                pid: Misc.shorten_listing(
                    self.products[pid],
                    ["is_inspected", "base_iids", "ref_iids", "stock_amount", "is_available"] +
                        ((["title"] + self.search_alias["free_text"] + self.inspect_alias["free_text"]) if is_inspected else []),
                    not is_inspected
                )
                for pid in self.__get_product_ids__(iid, True if is_inspected else None, False)
            }
            for iid, it in enumerate(self.intents) if it.reference is None and (not it.is_safe or (not is_inspected and not self.shortlists[iid]))
        }
        candidates = {iid: listings for iid, listings in candidates.items() if listings}
        if not candidates: return

        requirements = [
            {
                "intent_id": iid,
                "requirements": {"queries": self.intents[iid].queries, "priority": self.intents[iid].priority}
            }
            for iid in candidates
        ]
        messages = [
            {"role": "system", "content": Prompt.EVALUATE_SYS},
            {"role": "user", "content": Prompt.EVALUATE_USER.format(requirements = requirements, candidates = candidates)}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_evaluate_content(content, list(candidates.keys()), list(self.products.keys()))

            if not err:
                self.__update_with_evaluate_report__(js)
                break
            else:
                print(f"### Bad output: {err}")

                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Error: {err}"}
                ])

    def __bundle__(self) -> None:
        quick = self.__quick_bundle__()

        if quick:
            self.best_bundle = self.__deep_bundle__() if self.general.dependencies else quick
        else:
            self.best_bundle = []

    def __deep_bundle__(self) -> List:
        requirements = {
            "general": {k: v for k, v in self.general.to_json().items() if k in ["budget", "dependencies"]},
            "intents": {
                i: {k: v for k, v in it.to_json().items() if k in ["queries", "priority", "quantity"]}
                for i, it in enumerate(self.intents)
            }
        }
        candidates = {
            iid: {
                pid: {                    
                    "details": Misc.shorten_listing(
                        self.products[pid],
                        ["is_inspected", "base_iids", "ref_iids", "stock_amount", "is_available", "title"],
                        True
                    ),
                    "matchness_score": scores[0],
                    "priority_score": scores[1]
                } for pid, scores in self.shortlists[iid].items()
            } for iid in range(len(self.intents))
        }
        messages = [
            {"role": "system", "content": Prompt.BUNDLE_SYS},
            {"role": "user", "content": Prompt.BUNDLE_USER.format(requirements = requirements, candidates = candidates)}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_bundle_content(content, len(self.intents), list(self.products.keys()))

            if not err:
                return [js["best"][str(iid)] for iid in range(len(self.intents))] if js["best"] else []
            else:
                print(f"### Bad output: {err}")

                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Error: {err}"}
                ])

    def __quick_bundle__(self) -> List:
        candidates = [list(ids) for ids in product(*(list(group.keys()) for group in self.shortlists))]                
        bundles_ex = []

        for ids in candidates:
            cost_sum = sum([self.products[pid]["price"] * (self.intents[i].quantity or 1) for i, pid in enumerate(ids)])
            if self.general.budget is not None and cost_sum > self.general.budget: continue

            match_sum = sum(self.shortlists[i][pid][0] for i, pid in enumerate(ids))
            priority_sum = sum(self.shortlists[i][pid][1] * (1 if self.intents[i].sort_dir == "high" else -1) for i, pid in enumerate(ids))

            bundles_ex.append((ids, (-match_sum, -priority_sum, cost_sum)))

        if not bundles_ex: return []
        bundles_ex.sort(key = lambda be: be[1])

        return bundles_ex[0][0]

    def __logistics__(self) -> None:
        if not self.toolmap.message or not self.general.max_messages: return
        if self.__remaining_steps__() <= 7 + len(self.intents): return
        
        if self.questions: return
        samples = {}

        for iid in range(len(self.intents)):
            ids = self.__get_product_ids__(iid, True, False)
            if len(ids) < 2: return
            
            samples[iid] = {
                pid: Misc.shorten_listing(
                    self.products[pid],
                    ["is_inspected", "base_iids", "ref_iids", "stock_amount", "is_available", "title", "price", "brand", "shop"] +
                        self.search_alias["free_text"] + self.inspect_alias["free_text"]
                )
                for pid in ids
            }

        requirements = {
            "encourage": self.general.speak,
            "intents": {
                i: {k: v for k, v in it.to_json().items() if k in ["queries", "priority"]}
                for i, it in enumerate(self.intents)
            }
        }
        count = math.ceil(self.general.max_messages / len(self.intents))
        batch = min(math.ceil(Engine.MAX_NEEDS / count), 3)

        messages = [
            {
                "role": "system",
                "content": Prompt.LOGISTICS_SYS.format(count = count, batch = batch)
            },
            {"role": "user", "content": Prompt.LOGISTICS_USER.format(requirements = requirements, samples = samples)}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_logistics_content(content, len(self.intents))

            if not err:
                self.questions = {int(iid): items for iid, items in js.items()}
                break
            else:
                print(f"### Bad output: {err}")

                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Error: {err}"}
                ])

    def __check_reference__(self, iid: int, is_inspected: bool) -> None:
        candidates = {
            pid: Misc.shorten_listing(
                self.products[pid],
                ["is_inspected", "base_iids", "ref_iids", "stock_amount", "is_available", "title"],
                True
            )
            for pid in self.__get_product_ids__(iid, True if is_inspected else None, True)
        }
        if not candidates: return

        requirements = {
            "base_queries": self.intents[iid].queries, "reference": self.intents[iid].reference
        }
        messages = [
            {"role": "system", "content": Prompt.REFERENCE_SYS},
            {"role": "user", "content": Prompt.REFERENCE_USER.format(requirements = requirements, candidates = candidates)}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_reference_content(content)

            if js:
                self.__update_with_reference_report__(js, iid)
                break

            print(f"### Bad output: {err}")

            messages.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Error: {err}"}
            ])

    def __check_meta__(self, iid: int) -> None:
        candidates = {
            pid: Misc.shorten_listing(
                self.products[pid],
                ["is_inspected", "base_iids", "ref_iids", "stock_amount", "is_available", "title"] +
                    self.search_alias["free_text"] + self.inspect_alias["free_text"]
            )
            for pid in self.__get_product_ids__(iid, None, True)
        }
        if not candidates: return

        messages = [
            {"role": "system", "content": Prompt.META_SYS},
            {"role": "user", "content": Prompt.META_USER.format(query = self.intents[iid].reference, candidates = candidates)}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_meta_content(content)

            if not err:
                self.intents[iid].ref_category = js["category"]
                self.intents[iid].ref_brand = js["brand"]
                self.intents[iid].ref_shop = js["shop"]
                break
            else:
                print(f"### Bad output: {err}")

                messages.extend([
                    {"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}
                ])

    def __read_listings__(self, obs: List | Dict, alias: Dict, is_inspect: bool, iid: Optional[int], is_ref: bool) -> List[str]:
        records = Misc.extract_listings(obs, alias["product_id"])
        ids = []

        for r in records:
            pid = r[alias["product_id"]]
            ids.append(pid)
            
            if pid in self.products:
                self.products[pid]["base_iids"] = list(set(self.products[pid]["base_iids"] + ([] if iid is None or is_ref else [iid])))
                self.products[pid]["ref_iids"] = list(set(self.products[pid]["ref_iids"] + ([] if iid is None or not is_ref else [iid])))
                
                if self.products[pid]["is_inspected"] or not is_inspect: continue

            listing = {"is_inspected": is_inspect, "is_available": None, "stock_amount": None}
            
            for k, v in alias.items():
                if k in ["free_text", "product_id"]: continue
                listing[k] = r.get(v) if v else None

            for k, v in r.items():
                if any(k == a for a in alias.values()): continue
                listing[k] = v

            if pid not in self.products:
                if listing["is_available"] is None: listing["is_available"] = True

                listing["base_iids"] = [] if iid is None or is_ref else [iid]
                listing["ref_iids"] = [] if iid is None or not is_ref else [iid]
                
                self.products[pid] = listing
            else:
                del listing["is_available"]
                del listing["stock_amount"]

                self.products[pid].update(listing)

        return ids

    def __get_product_ids__(self, iid: Optional[int], is_inspected: Optional[bool], is_ref: Optional[bool]) -> List:
        ids = []
        
        for pid, details in self.products.items():
            if is_inspected is not None and details["is_inspected"] != is_inspected: continue

            if iid is not None:
                it = self.intents[iid]
                
                is_valid_base = True
                is_valid_ref = True

                if iid not in details["base_iids"]: is_valid_base = False
                if iid not in details["ref_iids"]: is_valid_ref = False

                if not details["is_available"]: is_valid_base = False

                if details["price"] is not None:
                    if it.min_price is not None and details["price"] < it.min_price: is_valid_base = False
                    if it.max_price is not None and details["price"] > it.max_price: is_valid_base = False

                if details["stock_amount"] is not None:
                    if it.quantity is not None and details["stock_amount"] < it.quantity: is_valid_base = False

                if details["category"] is not None:
                    if it.base_category is not None and details["category"].lower() != it.base_category.lower(): is_valid_base = False
                    if it.ref_category is not None and details["category"].lower() != it.ref_category.lower(): is_valid_ref = False
                
                if details["shop"] is not None:
                    if it.base_shop is not None and details["shop"].lower() != it.base_shop.lower(): is_valid_base = False
                    if it.ref_shop is not None and details["shop"].lower() != it.ref_shop.lower(): is_valid_ref = False

                if details["brand"] is not None:
                    if it.base_brand is not None and details["brand"].lower() != it.base_brand.lower(): is_valid_base = False
                    if it.ref_brand is not None and details["brand"].lower() != it.ref_brand.lower(): is_valid_ref = False

                if is_ref is not None:
                    if not is_ref and not is_valid_base: continue
                    if is_ref and not is_valid_ref: continue
                else:
                    if not is_valid_base and not is_valid_ref: continue

            ids.append(pid)
        
        return ids

    def __update_with_exercise_report__(self, js: Dict) -> None:
        for i, it in enumerate(self.intents):
            info = js["intents"][i]

            it.base_category = info["category"]
            it.base_brand = info["brand"]
            it.base_shop = info["shop"]

        self.inspector = js["inspect"]["best"]
        self.inspect_capacity = js["inspect"]["capacity"]

        self.inspect_alias = {
            "product_id": js["search"]["product_id"],
            "free_text": js["inspect"]["free_text"] or []
        }
        self.search_alias = js["search"]
        self.search_alias["free_text"] = self.search_alias["free_text"] or []

    def __update_with_event_report__(self, js: Dict[str, Dict]) -> None:
        event = js["event"]
        
        if event in ["requirements", "both"]:
            affected_iids = [int(i) for i in js["affected_intents"]]
            req = js["new_requirements"]            

            self.general.budget = req["general"]["budget"]
            self.general.guide = req["general"]["guide"]

            for i in affected_iids:
                it = req["intents"][str(i)]

                self.intents[i].min_price = it["min_price"]
                self.intents[i].max_price = it["max_price"]
                self.intents[i].quantity = it["quantity"]
                self.intents[i].queries = it["queries"]
                self.intents[i].reference = it["reference"]
                self.intents[i].is_safe = False

        if event in ["stock", "both"]:
            stock = js["new_stock_status"]

            for pid, inf in stock.items():
                prod = self.products[pid]

                if inf["new_availibility"] is not None: prod["is_available"] = inf["new_availibility"]
                if inf["new_stock_amount"] is not None: prod["stock_amount"] = inf["new_stock_amount"]
                if inf["new_price"] is not None: prod["price"] = inf["new_price"]
                if inf["suspicious"]: prod["is_available"] = False

            self.ever_stock_event = True

            for iid, shortlist in enumerate(self.shortlists):
                valid_ids = self.__get_product_ids__(iid, True, False)
                ids = list(shortlist.keys())

                for pid in ids:
                    if pid not in valid_ids: del shortlist[pid]

        if event: self.history.append((None, None, False, [], js))

    def __update_with_evaluate_report__(self, js: Dict) -> None:
        for iid, inf in js.items():
            iid = int(iid)

            self.templates[iid].update(inf["template"])            
            self.shortlists[iid] = inf["shortlisted"]

            self.intents[iid].sort_dir = inf["priority_preference"]
            self.intents[iid].is_safe = True

    def __update_with_reference_report__(self, js: Dict, iid: int) -> None:
        result = js["result"]
        if not result: return

        if result == "discovered":
            self.intents[iid].queries = js["new_requirements"]["base_queries"]
            self.intents[iid].reference = None
        else:
            self.intents[iid].reference = js["new_requirements"]["reference"]            

        for _, details in self.products.items():
            if iid in details["ref_iids"]: details["ref_iids"].remove(iid)
        
        self.intents[iid].ref_category = None
        self.intents[iid].ref_brand = None
        self.intents[iid].ref_shop = None

    def __get_explore_messages__(self, iid: int, is_ref: bool) -> List:
        valid_ids = self.__get_product_ids__(iid, False, is_ref)
        history, ids_seen = [], []

        for tc, i, ref, ids, obs in self.history:
            if tc is None or tc["function"]["name"] not in self.toolmap.search: continue
            if i != iid or ref != is_ref: continue

            listings = {}
            
            for pid in ids:
                if pid in valid_ids:
                    if pid not in ids_seen:
                        details = Misc.shorten_listing(
                            self.products[pid],
                            ["is_inspected", "base_iids", "ref_iids", "stock_amount", "is_available", "title"] +
                                self.search_alias["free_text"] + self.inspect_alias["free_text"]
                        )
                        ids_seen.append(pid)
                    else:
                        details = "[SEEN]"
                else:
                    details = "[MISMATCH]"

                listings[pid] = details

            obs = {
                "meta": obs,
                "displays_in_page": len(listings),
                "retrievals": listings
            }
            history.append({"tool_call": tc, "tool_response": obs})

        requirements = {
            "general": {k: v for k, v in self.general.to_json().items() if k in ["max_messages", "max_steps", "guide"]},
            "requirements": {
                "queries": self.intents[iid].queries,
                "min_price": self.intents[iid].min_price,
                "max_price": self.intents[iid].max_price,
                "category": self.intents[iid].base_category
            } if not is_ref else {
                "queries": self.intents[iid].reference,
                "category": self.intents[iid].ref_category
            }
        }
        return [
            {"role": "system", "content": Prompt.EXPLORE_SYS},
            {
                "role": "user",
                "content": Prompt.EXPLORE_USER.format(
                    requirements = requirements, history = history,
                    elapsed_messages = self.messages, remaining_messages = self.general.max_messages - self.messages,
                    elapsed_steps = self.steps, remaining_steps = self.__remaining_steps__()                        
                )
            }
        ]

    def __get_finish_messages__(self) -> List:
        history, candidates, guide = [], [], ""

        if self.best_bundle:
            for tc, _, _, _, obs in self.history:
                if tc is None or tc["function"]["name"] not in self.toolmap.cart: continue
                history.append({"tool_call": tc, "tool_response": obs})

            for i, pid in enumerate(self.best_bundle):
                candidates.append({"product_id": pid, "quantity": self.intents[i].quantity or 1})
            
            guide = self.general.guide or "Nothing."
            
            sys_prompt = Prompt.CHECKOUT_SYS
            usr_prompt = Prompt.CHECKOUT_USER
        else:
            sys_prompt = Prompt.DECLINE_SYS
            usr_prompt = Prompt.DECLINE_USER
        
        return [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": usr_prompt.format(
                    candidates = candidates, history = history, guide = guide,
                    elapsed_messages = self.messages, remaining_messages = self.general.max_messages - self.messages,
                    elapsed_steps = self.steps, remaining_steps = self.__remaining_steps__()                        
                )
            }
        ]

    def __get_inspect_queue__(self, iid_list: List[int], is_ref: bool) -> List[Tuple[int, str]]:
        valid_ids = {i: self.__get_product_ids__(i, False, is_ref) for i in iid_list}
        inspect_batch = Engine.BASE_INSPECT_BATCH if not is_ref else Engine.REF_INSPECT_BATCH
        queue = []

        for cursor in range(inspect_batch * self.inspect_capacity):
            added = False

            for iid, ids in valid_ids.items():
                if cursor < len(ids):
                    if ids[cursor] not in [pid for _, pid in queue]: queue.append((iid, ids[cursor]))
                    added = True
            
            if not added: break
        
        return queue

    def __next_job__(self) -> Tuple[str, List[int], bool]:
        if self.__is_urgent__(): return "finish", [], False
        
        if self.best_bundle:
            if self.__remaining_steps__() <= 8 + len(self.intents): return "finish", [], False

            if self.steps > len(self.intents) * 24:
                if self.questions or not self.general.max_messages or not self.toolmap.message:
                    if all(not it.priority for it in self.intents): return "finish", [], False

        if self.__remaining_steps__() <= 6 + len(self.intents) and not self.ever_stock_event: return "finish", [], False
        if self.__remaining_steps__() <= 4: return "finish", [], False

        iid_list = [iid for iid in range(len(self.intents)) if self.intents[iid].reference]
        if iid_list: return "explore", iid_list, True

        iid_list = [iid for iid in range(len(self.intents)) if not self.shortlists[iid]]
        if iid_list: return "explore", iid_list, False

        if self.best_bundle:
            iid_list = [iid for iid in range(len(self.intents)) if self.intents[iid].priority]
            if iid_list: return "explore", iid_list, False

        return "explore", list(range(len(self.intents))), False

    def __get_tools__(self, types: List[str], injection: Dict = None) -> List:
        chosen = []

        for tool in self.tools:
            if any(tool["function"]["name"] in getattr(self.toolmap, t) for t in types):
                copied: Dict[str, Dict[str, Dict[str, Dict]]] = copy.deepcopy(tool)

                if injection:
                    copied["function"]["parameters"]["properties"].update(injection)
                    copied["function"]["parameters"]["required"] = copied["function"]["parameters"].get("required", []) + list(injection.keys())
                
                chosen.append(copied)

        return chosen

    def __remaining_steps__(self):
        return self.general.max_steps - self.steps

    def __pop_purpose__(self, args: Dict) -> Optional[int]:
        try:
            iid = int(args["intent_id"])
            if iid < 0 or iid >= len(self.intents): raise ValueError()

            del args["intent_id"]
            return iid
        except Exception:
            return None

    def __is_urgent__(self) -> bool:
        if IS_DEBUG: return False

        elapsed = time.monotonic() - self.start_tick
        return elapsed > Engine.MAX_DURATION - 30.0

def agent_main(prob_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    eng = Engine(prob_data)
    return eng.run()
