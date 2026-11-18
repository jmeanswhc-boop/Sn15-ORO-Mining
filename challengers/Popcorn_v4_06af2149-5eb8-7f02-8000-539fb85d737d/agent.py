from typing import Type, Union, Optional, Any, Dict, List, Tuple, get_origin, get_args
from dataclasses import dataclass, fields, is_dataclass
from collections import defaultdict
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
    SUMMARIZE_SYS = "# You are an agent to analyze the user's shopping requirements and available tools. You are given:\n\t- A raw text input from user\n\t- A set of tools for shopping assistance\n\n# Your mission:\n\t- Parse user's verbose input into the structured information\n\t- Classify the given tools\n\n# Challenges:\n\t- User may have several product intents.\n\t- There may exist dependencies for the intents.\n\t\t* What is dependency?\n\t\t\tA requirement that binds two or more product intents. It makes an intent's submission choice dependant on other intent's submission, which is impossible to resolve it independently.\n\t\t* Examples of dependency\n\t\t\tInput: `All of your chosen goods must be in the same shop.` -> The shop name is unspecified. The recommended products for all intents must be sold at the same shop.\n\t\t\tInput: `The first product must be compatibile with the third product.` -> The exact compatibility value is unspecified, but the two of the recommended products must be compatible.\n\t\t* Example that is NOT dependency (False Positive)\n\t\t\tInput: `All these things should be for Honda model.`\n\t\t\tThe value of property \"model\" has been explicitly requested as \"Honda\". It's a shared requirement, NOT a dependency because each intent can still be resolved independently.\n\t\t\tYou must include a phrase \"For Honda model\" in every intent requirements' \"queries\" list, and that's all. No dependency.\n\t\t* (Exception) Budget restriction is essentially a dependency, for sure. However, do NOT include budget-related phrase in \"collection->dependencies\" field, because \"budget\" exists as a separate field.\n\t\t* Phrases about the procedure like \"find one by one\" are NOT dependencies.\n\t- There may exist reference products.\n\t\t* Definition:\n\t\t\tAn intent may point to another product for a requested (or excluded) property value, instead of explicit mention.\n\t\t\tThe pointed product is the reference for this intent.\n\t\t\tThe requested (or excluded) property value can be revealed only when the agent identified and inspected the reference product.\n\t\t* Examples of reference\n\t\t\tInput: `Color must be the same with the lotion that is sold by Kailey's and brand CR7.`\n\t\t\t\t\"The lotion sold by Kailey's and with brand CR7\" is the reference product. The base product's color must be its color.\n\t\t\tInput: `Style must be different to the listing of Classico in range TV and resolution 1080.`\n\t\t\t\t\"Listing of Classico in range TV and resolution 1080\" is the reference product. The base product's style must not be its style.\n\t- There may exist priorities.\n\t\t* What is priority?\n\t\t\tA requirement that defines the choice condition among multiple matches for intent.\n\t\t\tIt does not evaluate products' matchness, but it clarifies the key property and preferred direction for choosing the top 1 from all matching candidates.\n\t\t* Examples of user input for priority\n\t\t\t`Give me the cheapest one.`\n\t\t\t`Looking for ... with the longest sleeves length.`\n\n# Return JSON object like:\n{{\n\t\"agent\": {{\n\t\t\"max_messages\": <int|null>,\n\t\t\"max_steps\": <int|null>,\n\t\t\"substitute_allowed\": <bool>,\n\t\t\"speak\": <str|null>\n\t}},\n\t\"collection\": {{\n\t\t\"budget\": {{\n\t\t\t\"amount\": <int|null>,\n\t\t\t\"unit\": <str|null>,\n\t\t\t\"optimize\": <bool>\n\t\t}},\n\t\t\"dependencies\": [<str>, ...]\n\t}},\n\t\"intents\": [\n\t\t{{\n\t\t\t\"price\": {{\n\t\t\t\t\"min\": <int|null>,\n\t\t\t\t\"max\": <int|null>,\n\t\t\t\t\"unit\": <str|null>\n\t\t\t}},\n\t\t\t\"quantity\": <int|null>,\n\t\t\t\"priority\": <str|null>,\n\t\t\t\"queries\": [<str>, ...],\n\t\t\t\"reference\": <str|null>\n\t\t}},\n\t\t... ...\n\t],\n\t\"tools\": {{\n\t\t\"search\": [<str>, ...],\n\t\t\"inspect\": [<str>, ...],\n\t\t\"cart\": [<str>, ...],\n\t\t\"finish\": [<str>, ...],\n\t\t\"message\": [<str>, ...],\n\t\t\"other\": [<str>, ...]\n\t}}\n}}\n\n# Field Glossary:\n\t- \"agent\": the requirements that the agent must adhere during the shopping assistance process\n\t\t* \"max_messages\": maximum number of messages the agent can send user, if mentioned\n\t\t* \"max_steps\": maximum number of tool call steps the agent can take until finish, if mentioned\n\t\t* \"substitute_allowed\": true if user explicitly mentioned that substitute product can be submitted if no matching candidate is found, otherwise false        \n\t\t* \"speak\": sentence(s) that user encourages the agent to ask, tell or confirm something\n\t\t\te.g. \"Ask me which style to look through, then I'll tell you immediately.\"\n\t\t\te.g. \"Tell me what's worth asking before I commit my savings.\"\n\t- \"collection\": the requirements that the cart (the collection of all submitting products for the intents) must satisfy\n\t\t* \"budget\": upper limit of total spending, if mentioned\n\t\t\tset \"optimize\" as true if user explicitly mentioned that you must choose the collection with the least total cost\n\t\t* \"dependencies\": a list of sentences representing the dependencies for the submitting collection (empty list if no dependency exists)\n\t\t\tEach item must be a full sentence, like \"All must be in the same shop.\" or \"TV table must be compatible with remote controller.\"\n\t\t\tDo NOT include budget-related sentence.\n\t- \"intents\": a list of requirements dedicated to every individual product intent\n\t\t* \"price\": requested price range for the intent\n\t\t* \"quantity\": requested quantity for the intent\n\t\t* \"priority\": a short sentence representing the priority to choose the top 1, if mentioned\n\t\t\tAlways start with \"Choose ...\", like \"Choose the cheapest\" or \"Choose the one with the longest sleeves\".\n\t\t* \"queries\": a list of sentences representing the product intent's requirements, except for those related to \"price\", \"quantity\" and \"priority\"\n\t\t\te.g. [\"Comic style book written by Arthur Kailey.\", \"It must be in French.\", \"Story about dragons.\"]\n\t\t\te.g. [\"Basketball shoes with size as REFERENCE.\", \"Fit for students.\"]\n\t\t* use the identifier \"REFERENCE\" for the reference product when you describe items of \"queries\", if reference exists\n\t\t\tThe idea is to separate the queries of base product from reference product.\n\t\t\tSpecify like \"size as REFERENCE\" or \"type different to REFERENCE\" in \"queries\", and include nothing else for the reference-dedicated requirements.\n\t\t* \"reference\": query for the reference product if exists, otherwise null\n\t\t\te.g. \"What's sold in American Store with volleyball shoes range.\"\n\t- \"tools\": a dictionary of all available tool names classified by categories\n\t\t* \"search\": a list of tool names for finding products\n\t\t* \"inspect\": a list of tool names for viewing/comparing product details\n\t\t* \"cart\": a list of tool names for cart operations like adding, removing, reviewing, etc. in the cart\n\t\t* \"finish\": a list of tool names for finalization like placing order, decline, etc.\n\t\t* \"message\": a list of tool names for sending user a message\n\t\t* \"other\": a list of tool names classified into none of above\n\n# Guidelines:\n\t- Exclude from your output, user's verbose story and subjective explanation that are not meaningful for e-commerce search and ranking. Examples:\n\t\t* General phrases for emphasis of needing match like `I'm short of money so strictly meet the budget.`\n\t\t* Phrases like \"must be in stock\" or \"must be on hand\"\n\t\t* Phrases related to the listing photos (The agent is text-based, therefore can't check photos.)\n\t\t* Phrases like \"if no fit, say so and don't pick alternative\"\n\t- Each item object of \"intents\" list MUST be independently complete.\n\t\t* If a property value is explicitly shared multiple several intents, specify it in \"queries\" for all those intent objects.\n\t\t* Do NOT include a phrase related to a collection dependency in \"queries\" of individual intent.\n\t- For each intent object, the children field values MUST be strictly non-overlapped semantically.\n\t\t* Do NOT include in \"queries\", a redundant phrase related to \"price\", \"quantity\" or \"priority\".\n\t\t* Do NOT include the reference product's requirements in the base product's \"queries\" to prevent mixing up.\n\t- Respect the user's original wording and number/unit formatting for the free text values of your output.\n\t\t* Do NOT convert price/budget numbers and currency units by your own discretion. Just use the format as mentioned by user.\n\t\t* If price/budget was mentioned with both large and small units, use the small unit for output.\n\t\t\te.g. \"My budget is 210cts (2.1$).\" -> `\"budget\": {{ \"amount\": 210, \"unit\": \"cts\" }}`\n\t\t* Never replace a specific model or attribute with a broader interpretation.\n\t- If user explicitly mentioned that a property is optional, specify it clearly for \"queries\" output to avoid confusing must-have and optional requirements.\n\t\te.g. \"Gothic style is preferred but not strictly required.\"\n\t- Do NOT omit or paraphrase any informative details in \"queries\" to prevent losing requirements.\n\t- Make sure all given tools are included in \"tools\". Every tool must be classified into exactly one tool class.\n\t- Return JSON only. No plain text.\n\n# Examples of Incorrect Item for \"queries\" List:\n\t- \"2 sun glasses ...\" (\"2\" must be moved to \"quantity\" field, not here.)\n\t- \"... with the least price.\" (\"with the least price\" must be moved to \"priority\" field as \"Choose the cheapest.\", not here.)\n\t- \"... less than 200 cents.\" (\"less than 200 cents\" must be moved to \"price\" field, not here.)\n\t- \"A dress for visiting Maxim on Thursday.\" (\"for visiting Maxim on Thursday\" must be removed, because it's non-informative for e-commerce search.)"
    SUMMARIZE_USER = "# Raw User Input\n{query}\n\n# Available Tools\n{tools}"
    EXPLORE_SYS = "# You are an agent to assist shopping. You must explore the requested product, given:\n\t- user's shopping requirements\n\t- previous history of search tool calls vs their response objects\n\t- search tool(s)\n\n# Previous history is not guaranteed:\n\t- User's requirements may update any time during the whole shopping process.\n\t- Therefore, some past tool calls may be irrelevant to the current requirements.\n\n# Some product details from the previous history may be masked, to prevent distracting you.\n\t- [MISMATCH]: Product details that are trivially non-matching. (Price or category mismatch, etc.)\n\t- [SEEN]: Product details that have been showcased in a former tool response.\n\n# Task:\n\tComplete one search session. A session is a full scan of the database to the end by pagination, with a fixed tool arguments setup.\n\n# Session process:\n\t(1) Set up search keywords and filter options like the following, and trigger a search tool. Strictly keep the search tool's supported arguments policy.\n\t\t- Search tool and/or search mode\n\t\t\t* prefer keywords-based mode over filters-based mode\n\t\t- Search keywords (if supported)\n\t\t\t* 2-5 most informative and discriminative tokens from the queries\n\t\t\t* review the previously used keywords to avoid re-usage of the same arguments setup and maximize the efficiency\n\t\t- Category filter (if supported)\n\t\t\t* put \"category\" value (nullable) from the requirements\n\t\t- Price range (if supported)\n\t\t\t* use the \"min_price\" and \"max_price\" values (nullable) from the requirements\n\t\t- Pagination (if supported)\n\t\t\t* set for the first page (e.g. page index as 1, or offset as 0, or ...)\n\t(2) If the search returned with non-empty retrieval,\n\t\t- check the result statistics (total founds, displays, page index, offset, next token, etc.) to identify the possibility of further pagination\n\t\t- explore to the full using the same argument setup, but with pagination, until you reach the end\n\t(3) If the first search returned with empty retrieval or you have completed with the max pagination, stop tool calling and terminate with JSON output.\n\t(4) (Early termination) Even though the search returned with non-empty retrieval and there is a room to paginate,\n\t\t- you must terminate if the first page's retrieval product ID list is EXACTLY SAME with a search result from the previous history.\n\t\t- typically it happens when your arguments setup looks new but in fact it's essentially same with the previous one (pointless attempt).\n\t\t- you should terminate early with JSON output instead of repeating the pagination.\n\n# Search keywords hints\n\t- Best template:\n\t\t[discriminative category tokens] + [shop name] + [brand name] + [discriminative property value tokens]\n\t- Max 5 tokens allowed. More tokens are rejected with error. If it exceeds 5 tokens, choose the most discriminative ones.\n\t\t* At least one category-related token must be included.\n\t\t* Do NOT drop shop and brand names if mentioned.\n\t- NEVER include:\n\t\t* tokens related to excluded requirements\n\t\t\te.g. `Leave out those with color orange.`\n\t\t\tNever add \"orange\" in keywords, or it will retrieve products with \"orange\" property.\n\t\t* non-trusted numbers serving as boundaries instead of deterministic value\n\t\t\te.g. `with pack size at most 20 pieces`\n\t\t\tNever add \"20\" in keywords, or it will retrieve only the products that have \"20\" as exact value.\n\t\t* \"and\", \"or\", \"for\", \"with\", \"in\", \"compatible\", \"suitable\", \"new\", ... (such too generic tokens degrade search efficiency)\n\t\t* brackets, punctuation marks and symbolic characters like \"&\", \"-\", \"/\", \".\", \">\", \"<\", \"(\", \")\", \"[\", \"]\", ...\n\t\t* tokens that were not mentioned in the original queries are STRICTLY forbidden!\n\t\t* tokens paraphrased from the one mentioned in the original queries are forbidden!\n\t\t* \"REFERENCE\" (it's a special identifier for reference)\n\t- Handling `A or B` patterns in the queries:\n\t\t* If the `A or B` is about the product type or category, you must include both tokens A and B, because it typically means the product category \"A & B\" or \"A and B\".\n\t\t\te.g. `Boots or shoes with ...`\n\t\t\tIt means the requested product category is \"Boots & Shoes\".\n\t\t* If the `A or B` is about a property value, you should include either A or B in the keywords, but not both.\n\t\t\te.g. `material must be cotton or nylon, nothing else`\n\t\t\tNever include both \"cotton\" and \"nylon\" in the keywords, or it will retrieve only the products that have both those property values.\n\t\t\tYou can alter A and B for different sessions.\n\t- The following attempts are pointless:\n\t\t* Tokens reordering\n\t\t* Modifying capitalization\n\t\t* Modifying disabled arguments\n\t\t\te.g. category and price cap filters are disabled for keywords-based mode\n\t\t\te.g. keywords argument is disabled for filters-based mode        \n\t- Token counting:\n\t\t* Replace punctuation marks and brackets with space, and split the text by space to count the number of tokens.\n\t\t* The split result must NOT exceed 5 tokens.\n\n# How to set up search arguments for each session:\n\t- Check the previous history and select the arguments setup that has never been used.\n\t- Priorities:\n\t\t* Keywords-based mode is preferred over filters-based mode.\n\t\t* If all good keywords choices were tried in the previous history, go with filters-based mode.\n\t- If no promising keywords choice you can choose for a new session, NEVER create one violating search keywords hints but use filters-based mode.\n\n# For final output, return a JSON object like:\n{{\n\t\"retrieved\": <bool> // true if the session was done with non-empty non-seen retrieval, otherwise false\n}}\n\n# Guidelines:\n\t- NEVER terminate with JSON at the very first step. You MUST search!\n\t\t* Analyze the current requirements and the previous history, think of the most promising new keywords and/or filter options to check next.\n\t\t* Only terminate if your search produces empty listing or ever-seen listing or it reaches the full end of pagination.\n\t- NEVER change arguments setup except for pagination, once it's determined.\n\t\t* Keep consistent with search keywords and options during your session, until it retrieves nothing or reaches the max page.\n\t- NEVER change arguments setup even if it returns with an error message, which is counted as an empty retrieval.\n\t\t* Just terminate with JSON output (`retrieved`=false) if an error message is returned.\n\t- NEVER copy the arguments setup that has been already processed in the previous history. Search engine is deterministic and reproducible.\n\t- NEVER continue from the previous history's arguments, because it's untrusted and deprecated. Only rely on the current requirements and start a new session with new arguments.\n\t- Carefully check the search tool's description to prevent mis-usage of the arguments.\n\t\t* If a search tool does not support certain argument, do NOT use it.\n\t\t* Do NOT confuse the pagination arguments with the price cap arguments.\n\t- Strictly count the keyword tokens before triggering the first search. Make sure it does not exceed 5 tokens.\n\t\t* If it exceeds 5 tokens, choose the 5 most discriminative ones.\n\t- Learn from the previous history the way how search tools work with different arguments setup.\n\t- Carefully check your first page result to make sure the product ID list is not exactly same with any previous history pages. If it's ever seen, early terminate.\n\t- Exactly one tool call output during the search process. Exactly one JSON object for termination.\n\t- Respond with one tool call or one JSON object only. No plain text."
    EXPLORE_USER = "# Shopping Requirements\n{requirements}\n\n# History\n{history}\n\n# Now, please initiate your own search!"
    MAPPING_SYS = "# Task:\n\tIdentify the db-specific key names mapping for some important product attribute fields, given a list of sample product details retrieved from database.\n\n# If there exist valid product details given, return a JSON object like:\n{{\n\t\"product_id\": <str>, // key name for product ID\n\t\"category\": <str>, // key name for category\n\t\"price\": <str>, // key name for price\n\t\"is_available\": <str|null>, // key name for boolean-based stock-availability field, if exists\n\t\"stock_amount\": <str|null>, // key name for integer-based stock balance field, if exists\n\t\"free_text\": [<str>, ...] // key names for free lengthy texts like title, description, summary, etc.\n}}\n\n# If there is no valid product details given, return a JSON object like:\n{{\n\t\"result\": null\n}}\n\n# Guidelines:\n\t- Category is a word or short phrase representing the core product form.\n\t- Free text fields are those that may have substantive descriptive text values semantically. The followings are NOT free text fields:\n\t\t* a field whose value type is a dictionary (like attribute map)\n\t\t* a field whose value type is not string\n\t\t* a field whose value type is string but for identifiers, labels, short names, numeric strings and categorical values\n\t- Respond with JSON only. No plain text."
    MAPPING_USER = "# Sample Product Details\n{samples}"
    META_SYS = "# You are given:\n\t- user's shopping requirements\n\t- candidate product details retrieved from e-commerce database\n\n# Task:\n\tIdentify the db-specific string value evidences for category, brand and shop requirements that user mentioned\n\n# Return a JSON object like:\n{{\n\t\"category\": <str>, // db-specific value for requested category value\n\t\"brand\": <str|null>, // db-specific value for brand name if requested in the requirements, otherwise null\n\t\"shop\": <str|null> // db-specific value for shop name if requested in the requirements, otherwise null\n}}\n\n# Guidelines:\n\t- Category is a word or short phrase representing the core product form. Category is always requested in the requirements.\n\t\t* Choose the evidence that is most aligned to the user queries.\n\t\t* Example 1\n\t\t\tUser said: \"shoes\"\n\t\t\tEvidence candidates: \"Shoes\", \"Clothing & Shoes\"\n\t\t\tCorrect choice: \"Shoes\"\n\t\t* Example 2\n\t\t\tUser said: \"shirts or jackets\"\n\t\t\tEvidence candidates: \"Shirts\", \"Jackets\", \"Shirts and Jackets\"\n\t\t\tCorrect choice: \"Shirts and Jackets\"       \n\t- Find evidences for brand and shop names, only if explicitly requested in the requirements. NEVER output evidences for unmentioned brand or shop.\n\t- Respond with JSON only. No plain text."
    META_USER = "# User's Shopping Requirements\n{requirements}\n\n# Candidate Product Details\n{candidates}"
    INSPECT_SYS = "# You are given:\n\t- a list of various inspect tools for viewing product details\n\t- sample tool results of inspect tools\n\t- sample tool results retrieved from search tools\n\n# Tasks:\n\t- identify the best inspect tool that returns the property map\n\t- identify the key names of free lengthy text fields in product details\n\n# Different inspect tools showcase the product details with different levels.\n\t- Certain inspect tools only reveal the product listing details similar to what search tools produced.\n\t- Certain inspect tools reveal the comprehensive property map that search tools don't output.\n\n# Inspect tool preference priority (top to bottom):\n\t(1) The tool that produces the property map details that are not retrieved from search tools, is preferred.\n\t(2) The tool that can accept more product IDs as an argument at one call, is preferred over less or single ID supports.\n\t(3) The tool with less number of arguments, is preferred.\n\n# Return a JSON object like:\n{{\n\t\"best\": <str>, // name of the best inspect tool\n\t\"capacity\": <int>, // maximum number of product IDs that the best inspect tool can accept as its argument (1 if it only accepts a single ID or uncertain)\n\t\"free_text\": [<str>, ...] // field names used for free lengthy texts like title, description, summary, etc. in the inspect tools' product details\n}}\n\n# Guidelines:\n\t- Free text fields are those that may have substantive descriptive text values semantically. The followings are NOT free text fields:\n\t\t* a field whose value type is a dictionary (like attribute map)\n\t\t* a field whose value type is not string\n\t\t* a field whose value type is string but for identifiers, labels, short names, numeric strings and categorical values\n\t- Respond with JSON only. No plain text."
    INSPECT_USER = "# Inspect Tools Information\n{info}\n\n# Sample Results of Inspect Tools\n{view_results}\n\n# Sample Results of Search Tools\n{search_results}"
    RESOLVE_SYS = "# You are given:\n\t- current shopping requirements including a reference\n\t- product details of reference candidates\n\n# What is reference?\n\t- User may point to another product for a requested (or excluded) property value, instead of explicit mention. The pointed product is the reference.\n\t- The base product's requested (or excluded) property value can be revealed only when the reference product has been identified and reviewed.\n\t- Examples:\n\t\t* Base query: `Color must be the same with REFERENCE.`, Reference query: `The lotion sold by Kailey's and with brand CR7`\n\t\t* Base query: `Style must be different to REFERENCE.`, Reference query: `Listing of Classico in range TV and resolution 1080`\n\n# You must:\n\t- identify the potential reference products from the candidates\n\t- identify the descriptive field revealing the reference, if exists\n\t- update the requirements by replacing the reference mention with your finding, if applicable\n\n# Challenges:\n\t- There is no guarantee that the reference product exists in the candidates list.\n\t- The reference relationship can be chained.\n\t\t* The current reference product may point to another reference product.\n\t\t* In that case, the reference query must be updated while keeping the base query.\n\n# Your process (top to bottom):\n\t- Review all candidate details and shortlist them by evaluating the \"reference\" requirements.\n\t\t* If the shortlisted result is empty, terminate.\n\t\t* Otherwise, continue.\n\t- From the shortlisted candidates, check if there exists one whose details contain an explicit mention of reference chaining.\n\t\t* There may exists a candidate which contains a descriptive field value explaining the link to the another reference.\n\t\t\te.g. \"For color, go by lotion from Nintendo (plain, casual), not by this item.\"\n\t\t* If reference chaining detected, terminate with updating the \"reference\" field by the new reference requirements and \"base_queries\" unchanged.\n\t\t* Otherwise, continue.\n\t- From the shortlisted candidates, check if there exists one whose details discover the hidden property value.\n\t\t* Discovery can be either:\n\t\t\thaving a descriptive text field value that explicitly mentions the hidden property value, or\n\t\t\thaving a corresponding property in its attribute map\n\t\t* If reference has been discovered, terminate with updating the \"base_queries\" field by replacing reference and \"reference\" as null.\n\t\t\te.g. Base query: `Color must be the same with REFERENCE.` -> `Color must be green.`\n\t\t* Otherwise, terminate with no update.\n\n# Remarks\n\t- Reference chaining is more prioritized over discovery.\n\t- Having a straightforward descriptive text field implying reference update (chaining or discovery) is more prioritized over attribute map.\n\n# Return a JSON object like:\n{{\n\t\"shortlisted\": [<str>, ...],\n\t\"result\": <\"chain\"|\"discovered\"|null, str|null>,    \n\t\"new_requirements\": <object|null>\n}}\n\n# Output Guidelines\n\t- Set \"shortlisted\" as a list of product IDs that satisfy the original reference query strictly.\n\t- Set \"result\" value as:\n\t\t* \"chain\" if your process has been terminated by chain detection\n\t\t* \"discovered\" if your process has been terminated by unveiling the hidden property value\n\t\t* null if your process has been terminated without reference update\n\t- Set \"new_requirements\" as null if \"result\" value is null. Otherwise, output a complete new requirements object strictly keeping the original object like\n{{\n\t\"base_queries\": [<str>, ...],\n\t\"reference\": <str|null>\n}}\n\t- The \"new_requirements\" must be a complete object, not a patch. Therefore,\n\t\t* make sure to reflect the update correctly to the original requirements object (replace the identifier \"REFERENCE\" with the discovered value)\n\t\t* make sure you don't mis-overwrite or mis-erase any information from the current requirements that should not be affected by the reference update\n\t\t* copy the original \"base_queries\" if reference has been chained, not resolved.\n\t- Set null for \"reference\" field if the reference has been resolved by discovering hidden property value.\n\t- Set a full sentence representing the new reference requirements for \"reference\" field if the reference has been updated by chaining.\n\t- Return JSON only. No plain text."
    RESOLVE_USER = "# Shopping Requirements\n{requirements}\n\n# Field Glossary:\n\t- \"base_queries\": a list of sentences representing the requirements for the base product\n\t\te.g. [\"Comic style book written by Arthur Kailey.\", \"It must be in French.\", \"Story about dragons.\"]\n\t\te.g. [\"Basketball shoes with size as REFERENCE.\", \"Fit for students.\"]\n\t* use the identifier \"REFERENCE\" for reference products when you describe items of \"base_queries\", if reference exists\n\t\tThe idea is to separate the queries of base product from reference product.\n\t\tSpecify like \"size as REFERENCE\" or \"type compatible with REFERENCE\" in \"base_queries\", and include nothing else for the reference-dedicated requirements.\n\t* \"reference\": query for the reference product if exists, otherwise null\n\t\te.g. \"What's sold in American Store with volleyball shoes range.\"\n\n# Reference Candidate Details\n{candidates}"
    EVALUATE_SYS = "# You are given:\n\t- user's shopping requirements\n\t- a list of candidate product details\n\n# You must:\n\t- identify must-have and optional requirements\n\t- identify the key name related to the \"priority\" criterion if applicable\n\t- shortlist the candidates filtered by must-have requirements\n\t- score the shortlisted candidates\n\n# Requirements interpretation:\n\t- \"queries\" field contains the user's raw inputs.\n\t- Optional requirements are what user prefers but not strictly requests.\n\t- You must identify must-have requirements, optional requirements and non-informative contexts from \"queries\".\n\n# Identification of \"priority\"-related key:\n\t- Priority is a requirement that defines the choice condition among multiple matches.\n\t- It does not evaluate products' matchness, but it clarifies the property and its preferred direction for choosing the top 1 from all matching candidates.\n\t- There must exist a key from given candidate details representing the \"priority\". Examples:\n\t\t* Priority: \"Choose the cheapest.\" -> Key: \"price\"\n\t\t* Priority: \"Choose the one with the longest sleeves.\" -> Key: \"sleeves_length\"\n\t- If a product has no corresponding field in its details related to \"priority\", it's not a match.\n\n# Principles of evaluating candidates with requirements:\n\t- Allow synonyms. Allow extra tokens for more specified values. But do NOT derive matching by an aggressive scope broadening.\n\t\t* Required `class` - `school` meets.\n\t\t* Required `fit for all genders` - `unisex` meets.\n\t\t* Required `green` - `forest green` meets.\n\t\t* Required `fruit` - `apple` fails.\n\t\t* Required `winter` - `all seasons` fails.\n\t\t* Required `zip-up` - `pullover` fails.\n\t- Do NOT aggressively derive untold requirements.\n\t\t* Queries `gift for my dad` - do NOT derive a requirement `gender: men` from it - do NOT reject a candidate due `gender: women`.\n\t- A or B\n\t\t* Required `cotton or linen and nothing else` - `cotten` meets, `linen` meets, but `cotton + polyester` fails.\n\n# Shortlisting:\n\t- Choose only the candidates matching all must-have requirements.\n\t- Ignore other candidates for the subsequent processes.\n\n# Assign two scores for each shortlisted candidate:\n\t- Matchness score\n\t\t* default: 0\n\t\t* add +1 whenever it satisfies one optional requirement\n\t- Priority score\n\t\t* default: 0 (if \"priority\" not specified)\n\t\t* the numerical value (integer or float) of the field related to the \"priority\" criterion\n\t\t* e.g. `\"price\": 150`, `\"diameter\": 18.8`, ...\n\t\t* make sure to align the unit across all candidates, for fair comparison\n\n# Return JSON object like:\n{{\n\t\"shortlisted\": {{\n\t\t<product_id_1, str>: [<matchness_score, int>, <priority_score, int|float>],\n\t\t... ...\n\t}},\n\t\"priority_preference\": <\"low\"|\"high\"|null, str|null>\n}}\n\n# Output Guidelines:\n\t- \"shortlisted\": list all shortlisted product IDs with their matchness scores and priority scores\n\t- \"priority_preference\": \"low\" if lower priority score is preferred, \"high\" if higher priority score is preferred, null if \"priority\" not specified\n\n# Remarks:\n\t- Ignore verbose phrases in the \"queries\" that are non-informative.\n\t- Ignore the requirements like \"must be in stock or on hand\", because all products are available in stock.\n\t- Ignore the requirements like \"... like REFERENCE\" and \"... different to REFERENCE\", because it's just a placeholder.\n\t- ALL candidate products are confirmed that they meet price range and stock amount conditions. Do NOT double check those requirements.\n\t- Return JSON only. No plain text."
    EVALUATE_USER = "# Shopping Requirements\n{requirements}\n\n# Candidate Products\n{candidates}"
    RANK_SYS = "# You are given:\n\t- user's raw input for shopping requirements (user may have several product intents)\n\t- a list of candidate product details per intent\n\n# You must rank the candidates for every intent:\n\t- All candidates are confirmed that they satisfy the must-have requirements for the corresponding intent.\n\t- You should check how well each candidate details align to the user's raw input.\n\t\t* Some candidates may satisfy optional requirements that other candidates are missing.\n\t\t* Some candidates may include an aligned evidence for situational context from user's raw input. Example:\n\t\t\tUser input: \"this laptop is for my coding\" --> not a must-have requirement\n\t\t\tProduct details: \"Best for Coding and Web Browsing!\" --> good alignment!\n\t\t* Some candidates may include more aligned evidence than others.\n\t\t\tUser input: \"fit for student girls\"\n\t\t\tCandidate 1: `user: student girls`\n\t\t\tCandidate 2: `gender: female`, `age: children`\n\t\t\tRanking: Candidate 1 is better than candidate 2\n\t- Ignore \"priority\" condition, because it will be handled by the system based on numerical sort.\n\n# \"priority\" condition:\n\t- Priority is a requirement that defines the choice condition among multiple top matches.\n\t\t* e.g. \"Choose the cheapest.\"\n\t\t* e.g. \"Choose the one with the longest sleeves.\"\n\t- It does not evaluate products' matchness or ranking, but it clarifies the property and its preferred direction for choosing the one from all top candidates.\n\t- Each candidate's \"priority\" values have been already identified. Therefore, \"priority\" value must not affect on your ranking process.\n\n# Return JSON object like:\n{{\n\t<intent_id_1, str>: [\n\t\t<product_id_1, str>, ...\n\t],\n\t... ...\n}}\n\n# Remarks:\n\t- Ignore the requirements like \"must be in stock or on hand\", because all products are available in stock.\n\t- ALL candidate products are confirmed that they meet price range and stock amount conditions. Do NOT double check those requirements.\n\t- Return JSON only. No plain text."
    RANK_USER = "# Shopping Requirements\n{requirements}\n\n# Candidate Products per Intent\n{candidates}"
    BUNDLE_SYS = "# You are given:\n\t- user's shopping requirements having several product intents\n\t- a list of candidate product details with scores per intent\n\n# You must find the best bundle of candidates for submission, with one product per intent:\n\t- A bundle is a match if:\n\t\t* it satisfies the user budget if specified, and\n\t\t* it meets all the dependency conditions if specified\n\t- A matching bundle is better than another matching bundle if (top to bottom):\n\t\t* it has higher matchness score sum\n\t\t* it has more preferred priority score sum\n\t\t* (tie-break) its total cost is less expensive\n\t\t\t`total cost = sum(price * quantity)`\n\n# Remarks:\n\t- Dependency is a requirement serving as a condition to restrict the collection of submitting products for all intents.\n\t- Priority is a requirement that defines the choice condition among multiple matches for an individual intent.\n\t\t* It does not evaluate products' matchness, but it clarifies the property and its preferred direction for choosing the top 1 from all matching candidates.\n\t- Each candidate product comes with two scores:\n\t\t* First one is matchness score (the higher, the better)\n\t\t* Second one is priority score (the higher, the better if priority preference is \"high\", otherwise vice versa)\n\t- Review the candidates carefully to find the best matching bundle. If no matching bundle found, output null for the \"best\".\n\t- Note that all candidates are confirmed that they meet the corresponding intent's requirements. Therefore, just focus on bundling.\n\n# Response format:\n\t- If there exists a matching bundle, return a JSON object like:\n{{\n\t\"best\": {{\n\t\t<intent_id_1, str>: <product_id_1, str>,\n\t\t... ...\n\t}}\n}}\n\t- If no matching bundle found, return a JSON object like:\n{{\n\t\"best\": null\n}}"
    BUNDLE_USER = "# Shopping Requirements\n{requirements}\n\n# Candidate Products\n{candidates}"
    EVENT_SYS = "# You are given:\n\t- current shopping requirements\n\t- previous history of tool calls, tool response objects and handled events\n\n# You must:\n\t- identify the unhandled event messages from the previous tool responses, and\n\t- output the corresponding updates for the shopping requirements and/or the stock status.\n\n# What are event and event message?\n\t- An event is the happening which enforces to update the current shopping requirements or the product stock status.\n\t- An event message is a free text buried inside a tool response, that signals an event.\n\n# Shopping requirements update events include, but are not limited to:\n\t- user brings a requirement\n\t\t* Message: \"I just want gothic style, nothing else.\"\n\t\t* Action: add a query in \"queries\" field\n\t- user brings a requirement with pointing a reference product\n\t\t* Message: \"Style must be different to the listing of Classico in range TV and resolution 1080.\"\n\t\t* Action:\n\t\t\tadd a query \"Style must be different to REFERENCE.\" in `queries`, and\n\t\t\tset \"Listing of Classico in range TV and resolution 1080\" for `reference` field\n\t- user modifies an existing requirement\n\t\t* Message: \"Changed my mind. Color must be blue.\"\n\t\t* Action: modify a query in \"queries\" field\n\t- user moves a must-have requirement to optional requirement\n\t\t* Message: \"CPU speed is a wish than a need.\"\n\t\t* Action: modify the corresponding phrase \"must be 2.2MHz\" into \"2.2MHz is preferred but not strictly required\"\n\t- user clarifies the exact need from the ambiguous requirement\n\t\t* Message: \"I'd prefer 3-pin over 2-pin.\"\n\t\t* Action: modify the corresponding phrase \"must be 2 or 3 pins\" into \"must be 3-pin\"\n\t- user claims or changes the budget.\n\t\t* Message: \"Oh, I forgot to let you know that my budget is 10 dollars.\", \"Sorry but I can't spend more than 120cts.\"\n\t\t* Action: update the \"general->budget\"\n\t- user claims or changes the price range requirement of a product.\n\t\t* Message: \"The TV must be less expensive than 1000USD.\"\n\t\t* Action: update the \"max_price\"\n\t- user claims or modifies the agent rules.\n\t\t* Message: \"Hey, I'm confused with your question. Can you ask one thing at a time?\"\n\t\t* Action: set `ask_one_by_one` as true\n\t- and anything that needs update of shopping requirements\n\n# Reference product:\n\t- Definition\n\t\tAn intent may point to another product for a requested (or excluded) property value, instead of explicit mention.\n\t\tThe pointed product is the reference for this intent.\n\t\tThe base product's requested (or excluded) property value can be revealed only when the agent identified and inspected the reference product.\n\t- Examples of reference\n\t\t* Input: `Color must be the same with the lotion that is sold by Kailey's and brand CR7.`\n\t\t\t\"The lotion sold by Kailey's and with brand CR7\" is the reference product. The base product's color must be its color.\n\t\t* Input: `Style must be different to the listing of Classico in range TV and resolution 1080.`\n\t\t\t\"Listing of Classico in range TV and resolution 1080\" is the reference product. The base product's style must NOT be its style.\n\n# Stock status update events include:\n\t- user recognizes that a carted product has gone or run out.\n\t\t* Message: \"The one you just picked disappeared from the shelf.\"\n\t\t* Action: identify the product ID of \"the one\" disappeared, from the previous cart operation history\n\t- user recognizes that a product's price has been changed.\n\t\t* Message: \"Sorry but the price of the one you selected has been updated to 2100.\"\n\t\t* Action: identify the product ID of \"the one\" whose price has been changed, from the previous cart operation history\n\t- user recognizes that a product's stock amount has been changed:\n\t\t* Message: \"Oh my! This one has only 2 remaining items on hand. Can we order it?\"\n\t\t* Action: identify the product ID of \"this one\" whose stock amount has been changed, from the previous cart operation history\n\t- user suspects a carted product\n\t\t* Message: \"Well, the pick's price seems manipulated. Can you double check?\"\n\t\t* Action: identify the product ID of \"the pick\" who's suspicious, from the previous cart operation history\n\n# An event message can be:\n\t- either a clarified answer to the question that the agent asked using a message tool (passive event)\n\t- or a free text injected in a tool response object (voluntary event)\n\n# The followings are NOT event messages:\n\t- normal result data for search and inspect tool calls, like product details and statistics\n\t- error messages dedicated to the previous tool call\n\t- meta data with short technical messages like `\"code\": 0`, `\"done\": false`, `\"result\": \"canceled\"`\n\t- user's question asking the agent to clarify something, like \"I can use coupon. Can you double check the picks with it?\"\n\t- a message looking alike a stock-out event but happened just after removing cart operation\n\t- dummy answer to the previous agent question, that does not update the shopping requirements, like \"Whatever you like. I trust you.\"\n\n# Common mistakes to avoid:\n\t- A message looking alike a stock-out event but happened just after removing cart operation, is not a valid event message.\n\t\t* It's typically a duplicated old message that have already been processed.\n\t- Neutral and non-informative answer to the previously asked agent question, is not a valid event message.\n\t\t* e.g. \"Six of one, half a dozen of the other.\" is not for an event, but a dummy answer.\n\t- The same event message may appear repeatedly for reminder. One event message must be handled only once, regardless of its repetition.\n\t\t* Ignore the stock-out event message happened just after removing cart operation, because it's the repeated mention of stock-out event for the removed product.\n\n# Remarks:\n\t- You MUST review the previous tool responses carefully to identify all the unhandled new events embedded in them.\n\t\t* One tool response may include multiple events.\n\t\t* User may use informal language or slang. Understand the nuance correctly.\n\t\t* Ignore the texts that are non-meaningful for requirements or stock update.\n\t\t* Ignore the events that have beeen already handled.\n\t\t* Do NOT check the latest tool response only! Unhandled event messages may be hidden in a bit older tool responses that happened after the latest event handling was processed.\n\t- You MUST identify the product IDs from the cart operation history, when stock status event detected.\n\t\t* Event message may not explicitly mention the product ID whose stock status (stock amount, availibility, price, etc.) has been changed.\n\t\t* In that case, you must infer its correct ID by reviewing the cart add/remove tool call history.\n\t\t* Note that the add-cart operation always succeeds, regardless of the event message being embedded in its tool response.\n\t\t* Therefore, if an add-cart tool call was triggered, you must count that the product has been inserted into the cart.\n\t- How to identify the handled events?\n\t\t* Check the tool call and response history of `trigger_event` tool.\n\t\t* It showcases the trace of previously detected and handled events.\n\t- Removal of existing requirement is essentially moving it to optional.\n\t\t* Some event messages may sound like removing an existing requirement. (e.g. \"Never mind pack type, I'll ignore it.\")\n\t\t* You must grant it as moving from must-have to optional requirement, instead of complete erasing. (e.g. \"Plastic bag pack is preferred but not strictly required.\")\n\n# Mapping implicit property names in event messages:\n\t- Sometimes you get confused with an event message because,\n\t\t* the event message mentions about a certain property name to clarify, modify or move an existing requirement (e.g. \"Now I'll forget occasion.\")\n\t\t* but you can't find it in the existing \"queries\" for the intent (e.g. \"nylon tshirts for winter\")\n\t- If it's obvious what property name it is pointing to, continue.\n\t- If you're confused, check the \"queries\" carefully to identify the property name the event implies. Examples:\n\t\t* \"for winter\" -> `occasion: winter`\n\t\t* \"that comes with can\" -> `pack_type: cans`\n\n# Return a JSON object like:\n{{\n\t\"event\": <\"requirements\"|\"stock\"|\"both\"|null, str|null>,    \n\t\"new_requirements\": <object|null>,\n\t\"new_stock_status\": <object|null>,\n\t\"affected_intents\": [<int>, ...]\n}}\n\n# Output Guidelines\n\t- Set \"event\" value as:\n\t\t* null if no valid new event messages found\n\t\t* \"both\" if both requirements update and stock status update events detected\n\t\t* \"requirements\" if only requirements update event detected\n\t\t* \"stock\" if only stock status update event detected\n\t- \"affected_intents\" is the list of intent IDs whose requirements have been affected by the current update.\n\t\t* Set empty list if the requirements update didn't happen or it patched the \"general\" only.    \n\t- Set \"new_requirements\" value as null if \"event\" value is either \"stock\" or null. Otherwise, set as:\n\t\t* the new requirements object after update (not a patch but a complete object strictly keeping the original requirements object' structure) like\n\n{{\n\t\"general\": {{ ... }},\n\t\"intents\": {{ ... }}\n}}\n\t- Set \"new_stock_status\" value as null if \"event\" value is either \"requirements\" or null. Otherwise, set as:\n\t\t* the object including all the update patches for the stock database like by product IDs like\n{{\n\t<product_id_1, str>: {{\n\t\t\"new_stock_amount\": <int|null>, // new stock amount from update, if applicable\n\t\t\"new_availibility\": <false|null>, // false if stock-out detected by the update, otherwise null\n\t\t\"new_price\": <int|null>, // new product price from update, if applicable\n\t\t\"suspicious\": <true|null> // true if flagged by the update, otherwise null\n\t}},\n\t... ...\n}}\n\t- When you output \"new_requirements\", carefully reflect the updates and produce the complete object.\n\t\t* Correctly patch the detected updates to the current requirements object.\n\t\t* Make sure you don't mis-overwrite or mis-erase any information from the current requirements that should not be affected by the detected events.\n\t\t* Example 1:\n\t\t\tAn original query: \"Socks in red for spring.\"\n\t\t\tAn event message: \"Ignore occasion.\"\n\t\t\tCorrect new query: \"Socks in red. Occasion spring is preferred but not strictly required.\"\n\t\t* Example 2:\n\t\t\tAn original query: \"Badminton net with prestrung sales.\"\n\t\t\tAn event message: \"Brand must be Nike.\"\n\t\t\tCorrect new query: \"Badminton net with prestrung sales. Brand must be Nike.\"\n\t- For each product ID of \"new_stock_status\", at least one of \"new_stock_amount\", \"new_availibility\", \"new_price\" and \"suspicious\" must be not null.\n\t\t* If all four field values are null, it means no update happened, which implies error.\n\t- Return JSON only. No plain text."
    EVENT_USER = "# Current Shopping Requirements\n{requirements}\n\n# Field Glossary:\n\t- \"general\": the requirements for the agent and the collection of all products for submission\n\t\t* \"budget\": (int|null) upper limit of total spending, if mentioned\n\t\t* \"ask_one_by_one\": true if agent can ask about several properties at one question, otherwise false\n\t- \"intents\": a list of requirements dedicated to every individual product intent\n\t\t* \"min_price\": (int|null) lower limit of product price\n\t\t* \"max_price\": (int|null) upper limit of product price\n\t\t* \"quantity\": (int|null) requested quantity for the intent\n\t\t* \"queries\": a list of sentences representing the requirements as well as the informative contexts except for those related to price and quantity\n\t\t\te.g. [\"Comic style book written by Arthur Kailey.\", \"It must be in French.\", \"Story about dragons.\"]\n\t\t\te.g. [\"Basketball shoes with size as REFERENCE.\", \"Fit for students.\"]\n\t\t* use the identifier \"REFERENCE\" for reference products when you describe items of \"queries\", if reference exists\n\t\t\tThe idea is to separate the queries of base product from reference product.\n\t\t\tSpecify like \"size as REFERENCE\" or \"type compatible with REFERENCE\" in \"queries\", and include nothing else for the reference-dedicated requirements.\n\t\t* \"reference\": query for the reference product if exists, otherwise null\n\t\t\te.g. \"What's sold in American Store with volleyball shoes range.\"\n\n# History\n{history}"
    LOGISTICS_SYS = "# You are given:\n\t- user's shopping requirements\n\t- sample product details\n\n# Now you must prepare questions to ask to the user. There are 3 types of questions:\n\t(I) The questions that come from \"encourage\" field.\n\t\t- User may encourages you to ask something about the requested product. Such statement is included in \"encourage\" field if exists.\n\t\t- Prioritize most for the encouraged questions.\n\t(II) The questions that come from the \"queries\" directly.\n\t\t- Sometimes user represented needs ambiguously with several value options combined by `or`.\n\t\t\t* e.g. \"universal plug with 2 or 3 pins\"\n\t\t\t* You must ask user to clarify the exact need.\n\t\t- The `or` used for representing category is not included in this case.\n\t\t\t* e.g. \"Pants or jeans with color blue for winter.\"\n\t\t\t* Do NOT ask user to clarify pants or jeans. \"Pants & Jeans\" is the category value.\n\t(III) The questions for hidden properties.\n\t\t- User may have untold needs or preferences.\n\t\t- User clarifies them only when the agent asks questions correctly targeting the their property names.\n\t\t- You must predict the properties that are most likely to hide user's requirement, and ask about them to reveal the hidden needs.\n\n# Step 1: Type I Questions\n\t- NOT ALL sentences of \"encourage\" field imply questions. Review it to identify the real encouragement for a question.\n\t\t* Do NOT aggressively derive a non-encouraged question.\n\t\t* Do NOT create a question that is encouraged but not related to the current shopping requirements.\n\t- Create questions as instructed and encouraged by user for the requesting product.\n\t- Skip Type I if nothing encouraged found for the requesting product.\n\n# Step 2: Type II Questions:\n\t- Explicitly ask to clarify which is more important among several values:\n\t\t* e.g. \"Which matters more to you, 2-pin or 3-pin?\"\n\t- Make one question for every ambiguous pair.\n\t- Skip Type II if no such ambiguous pair exists.\n\n# Step 3: Properties for Type III Questions:\n\t- Collect all property names appeared in the product details.\n\t- Ignore invalid property names from consideration:\n\t\t* Drop properties that have been mentioned in the original requirements directly or indirectly (including \"priority\").\n\t\t* Drop properties like price or budget.\n\t\t* Drop properties like product ID, shop/seller, brand or warranty period.\n\t- Merge valid property names by string similarity.\n\t\t* Merge properties with similar string names.\n\t\t\te.g. (\"style\", \"fa.style\", \"styles\") -> \"style\"\n\t\t\te.g. (\"color\", \"color_family\", \"color family\") -> \"color\"            \n\t\t* NEVER derive merging synonyms with completely different strings.\n\t\t\te.g. \"style\" ≠ \"form\"\n\t- Count the occurencies of every mergied name across the product details for this intent.\n\t\t* For a mergied name, add +N whenever it occured in one product details. N >= 1 because there may exist multiple occurencies in one product details due merging relationship.\n\t\t* Calculate the sum across all product details for each mergied name.\n\t- Sort the mergied property names by total occurencies in descending order.\n\t\t* (Tie-break) If two names have the same occurencies, prioritize the more widely used property for this product category in e-commerce.\n\n# Respond with a JSON object like:\n{{\n\t\"questions\": [<str>, ...], // Type I and II questions\n\t\"properties\": [<str>, ...] // sorted list of mergied property names, for Type III questions\n}}\n\n# Return JSON only. No plain text."
    LOGISTICS_USER = "# Shopping Requirements\n{requirements}\n\n# Sample Product Details\n{samples}"
    CHECKOUT_SYS = "# You are given:\n\t- product IDs and quantities to submit\n\t- previous history of cart operations, extracted from the complete shopping tool call history\n\t- cart operation tools (add, remove, etc.)\n\t- submit tool(s) for placing order\n\n# Normal submission process:\n\t- Add the products to the cart.\n\t- Place order.\n\n# By optimizing the tool call steps, you must:\n\t- Analyze the previous history to identify the non-requested products in the cart.\n\t- If non-requested products remaining in the cart, remove them immediately.\n\t- Identify the requested products that are outside the cart.\n\t- Add the missing requested products to the cart.\n\t- Place order to submit.\n\n# Guidelines:\n\t- Do at your best effort for adding or removing.\n\t\t* If add/remove tool supports multiple product IDs, leverage this feature to the full to optimize the tool call steps.\n\t\t* If it only accepts a single ID, add or remove one by one.\n\t- Derive the current cart content from the history correctly!\n\t\t* Carefully review the history to prevent mis-estimation.\n\t\t* Note that all cart operations are always carried out successfully by design.\n\t\t* Therefore, all add/remove operation succeeds, regardless of refusal-like messages included in their response objects.\n\t- Do NOT trigger the cart review tool call for the only purpose of checking the content. It's your own duty to derive it from the history without extra tool calls.\n\t- Optimize the number of tool calls by avoiding redudnant actions.\n\t\t* Do NOT add the product that has been already added. Do NOT add the non-requested product.\n\t\t* Do NOT remove the product that has been already removed. Do NOT remove the requested product.\n\t- Correctly pass the quantity value when you add a product to the cart, if it's the required argument.\n\t- NEVER give up or decline! You MUST submit all the requested products without failure!\n\t- Only one tool call per each turn. Do NOT trigger multiple tool calls at a time.\n\t- Respond with tool calls only. No plain text."
    CHECKOUT_USER = "# Products vs Quantities to Submit\n{candidates}\n\n# Cart Operation History\n{history}"
    DECLINE_USER = "# You are given:\n\tsubmit/decline tool(s) for e-commerce shopping assistance\n\n# Task:\n\t- Do nothing but call the decline tool once.\n\t- Respond with one tool call only. No plain text."
    QUESTION_TEMPLATE = "What {q} would you like{intent}? Please let me know your exact need or preference for one or more properties among them.\n(If you can't accept multiple properties at one question, no worries, just answer me with one property that you can tell your exact need or preference, and ignore other properties.)"

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
    def shorten_listing(v: Dict | List | str, stop_keys: List[str]) -> Dict:
        def __clean__(value: Any):
            if isinstance(value, dict):
                return {k: __clean__(val) for k, val in value.items() if k not in stop_keys}
            elif isinstance(value, list):
                return [__clean__(item) for item in value]
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

    @staticmethod
    def has_evidence(js: Dict, val: str) -> bool:
        def __check__(node: Any) -> bool:
            if isinstance(node, dict): return any(__check__(v) for _, v in node.items())
            if isinstance(node, list): return any(not isinstance(v, str) and __check__(v) for v in node)
            if isinstance(node, str): return node.lower().strip() == val.lower().strip()
            return False

        return __check__(js)

@dataclass
class GeneralInfo(JsonModel):
    max_messages: int
    max_steps: int
    substitute_allowed: bool
    speak: Optional[str]    
    budget: Optional[int]
    dependencies: Optional[List[str]]
    optimize: bool
    ask_one_by_one: bool = False    

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
    is_strict: bool = False
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

        for k, t in [("max_messages", (int, NoneType)), ("max_steps", (int, NoneType)), ("substitute_allowed", (bool)), ("speak", (str, NoneType))]:
            if k not in js["agent"]: return None, f"Key `agent->{k}` not found."
            if not isinstance(js["agent"][k], t): return None, f"Value for `agent->{k}` expects {' or '.join([str(x) for x in t])}."

            v = js["agent"][k]

            if k == "guide":
                v = v or []
                if any(not isinstance(x, str) for x in v): return None, "Value for `agent->guide` expects a list of strings."

            general[k] = v

        for k, t in [("budget", (dict, NoneType)), ("dependencies", (list, NoneType))]:
            if k not in js["collection"]: return None, f"Key `collection->{k}` not found."
            if not isinstance(js["collection"][k], t): return None, f"Value for `collection->{k}` expects {' or '.join([str(x) for x in t])}."

            if k == "budget":
                v = js["collection"][k].get("amount")

                if not isinstance(v, (int, NoneType)): return None, f"Value for `collection->budget->amount` expects <class int> or <class NoneType>."
                general[k] = v

                v = js["collection"][k].get("optimize")

                if not isinstance(v, (bool)): return None, f"Value for `collection->budget->optimize` expects <class bool>."
                general["optimize"] = v
            elif k == "dependencies":
                v = js["collection"][k] or []

                if any(not isinstance(x, str) for x in v): return None, "Value for `collection->dependencies` expects a list of string."
                general[k] = v

        toolmap, seen = {}, []

        for k in ["search", "inspect", "cart", "finish", "message", "other"]:
            if k not in js["tools"]: return None, f"Key `tools->{k}` not found."
            
            if not isinstance(js["tools"][k], (list, NoneType)): return None, f"Value for `tools->{k}` expects <class list> or <class NoneType>."
            v = list(set(js["tools"][k] or []))

            if any(not isinstance(x, str) for x in v): return None, f"Value for `tools->{k}` expects a list of string."
            
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
    
                    if any(not isinstance(x, str) for x in v): return None, "Value for `[intent]->queries` expects a list of string."
                    if not v: return None, "Value for `[intent]->queries` cannot be empty."

            obj = {k: v for k, v in it.items() if k != "price"}

            obj["min_price"] = it["price"].get("min")
            obj["max_price"] = it["price"].get("max")

            intents.append(obj)

        return {"general": general, "intents": intents, "toolmap": toolmap}, ""

    @staticmethod
    def parse_explore_content(content: str) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."
        return js, ""

    @staticmethod
    def parse_mapping_content(content: str) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        if "result" in js:
            if js["result"] is not None: return None, f"Value for `result` must be null if you didn't find any valid product details."
            if len(js) > 1: return None, f"Key `result` must be the only key if you didn't find any valid product details."

            return js, ""

        for k, t in [
            ("product_id", (str)), ("category", (str)), ("price", (str)), ("is_available", (str, NoneType)),
            ("stock_amount", (str, NoneType)), ("free_text", (list))
        ]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], t): return None, f"Value for `{k}` expects {' or '.join([str(x) for x in t])}."
            if k == "free_text" and any(not isinstance(x, str) for x in js[k]): return None, f"Value for `{k}` expects a list of string."

        return js, ""

    @staticmethod
    def parse_meta_content(content: str) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k, t in [("category", (str)), ("shop", (str, NoneType)), ("brand", (str, NoneType))]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], t): return None, f"Value for `{k}` expects {' or '.join([str(x) for x in t])}."

        return js, ""

    @staticmethod
    def parse_inspect_content(content: str) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k, t in [("best", (str)), ("capacity", (int)), ("free_text", (list))]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], t): return None, f"Value for `{k}` expects {' or '.join([str(x) for x in t])}."
            if k == "free_text" and any(not isinstance(x, str) for x in js[k]): return None, f"Value for `{k}` expects a list of string."

        return js, ""

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

            for k, t in [("budget", (int, NoneType)), ("ask_one_by_one", (bool))]:
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
    def parse_evaluate_content(content: str, product_ids: List[str]) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k, t in [("shortlisted", (dict)), ("priority_preference", (str, NoneType))]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], t): return None, f"Value for `{k}` expects {' or '.join([str(x) for x in t])}."

        if js["priority_preference"] not in ["low", "high", None]:
            return None, f"Value for `priority_preference` must be one of 'low', 'high' and null."

        for pid, scores in js["shortlisted"].items():
            if pid not in product_ids: return None, f"Invalid product ID key '{pid}' for `shortlisted`."
            
            if not isinstance(scores, list) or len(scores) != 2:
                return None, f"Invalid score pair {scores} for `shortlisted->{pid}`. It must be a list of two numbers."

            for x in scores:
                try:
                    x = float(x)
                except Exception:
                    return None, f"Invalid score pair {scores} for `shortlisted->{pid}`. It must be a list of two numbers."

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
    def parse_logistics_content(content: str) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k, t in [("questions", (list)), ("properties", (list))]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], t): return None, f"Value for `{k}` expects {' or '.join([str(x) for x in t])}."            
            if any(not isinstance(x, str) for x in js[k]): return None, f"Value for `{k}` expects a list of string."

        return js, ""

    @staticmethod
    def parse_resolve_content(content: str, product_ids: List[str]) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        for k, t in [("shortlisted", (list)), ("result", (str, NoneType)), ("new_requirements", (dict, NoneType))]:
            if k not in js: return None, f"Key `{k}` not found."
            if not isinstance(js[k], t): return None, f"Value for `{k}` expects {' or '.join([str(x) for x in t])}."
            if k == "shortlisted" and any(not isinstance(x, str) for x in js[k]): return None, f"Value for `{k}` expects a list of string."

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

        for pid in js["shortlisted"]:
            if pid not in product_ids: return None, f"Invalid product ID '{pid}' found in `shortlisted`."

        return js, ""

    @staticmethod
    def parse_rank_content(content: str, intent_count: int, candidates: Dict[int, List[str]]) -> Tuple[Optional[Dict], str]:
        js: Dict[str, Dict[str, Dict] | List[Dict[str, Dict]]] = Misc.parse_json(content or "")
        if js is None: return None, "JSON parsing failed."

        if sorted(list(js.keys())) != [str(x) for x in range(intent_count)]: return None, f"Keys are incorrect. Make sure to include all intent IDs from 0 to {intent_count - 1}."

        for iid, ranked in js.items():
            if not isinstance(ranked, list): return None, f"Value for intent `{iid}` expects a list of product IDs."
            if any(not isinstance(x, str) for x in ranked): return None, f"Value for intent `{iid}` expects a list of product IDs."
            if sorted(ranked) != sorted(candidates[int(iid)]): return None, f"Value for intent `{iid}` expects a list of all its candidates' product IDs."

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
    MAX_NEEDS = 8
    
    INSPECT_BATCH = 5
    REF_BATCH = 3

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
        self.event_msg_len = 1
        
        self.last_asked_iid = None        
        self.rank_prompt = None
        self.bundle_prompt = None

        self.view_batch = 1
        self.viewer = None        

        self.messages = 0
        self.steps = 0

        self.products: Dict[str, Dict[str, Dict | List | str]] = {}
        self.dlg = []

        self.history: List[Tuple[Optional[Dict], Optional[int], bool, List, Any]] = []
        self.explore_pending: Dict[Tuple[int, bool], int] = defaultdict(lambda: 0)

        self.best_bundle = []
        self.alias = {}

        self.meta_try: Dict[Tuple[int, bool], int] = defaultdict(lambda: 0)
        self.meta_len: Dict[Tuple[int, bool], int] = defaultdict(lambda: 0)

    def run(self) -> List[Dict[str, Any]]:
        self.__summarize__()
        round = 0

        while not self.finished:
            round_eta = (self.__remaining_steps__() - len(self.intents) - 3) // self.round_volume
            self.__event__()

            for iid in range(len(self.intents)):
                if not self.intents[iid].is_safe:
                    self.__evaluate__(iid, False, True)

                    if round_eta <= 1:
                        if not self.matches[iid]: self.__evaluate__(iid, True, True)
                        if self.general.substitute_allowed and not self.matches[iid]: self.__evaluate__(iid, True, False)

            if not self.__is_urgent__() and (self.__remaining_steps__() > 5 + len(self.intents) or (self.ever_stock_event and self.__remaining_steps__() > 4)):
                round_volume = min(
                    (self.__remaining_steps__() - len(self.intents) - 5) if not self.ever_stock_event else (self.__remaining_steps__() - 4),
                    self.round_volume
                )
                jobs = self.__prioritize_jobs__()

                for iid, is_ref, w in jobs:
                    if self.finished: break
                    asked = False

                    if not is_ref and self.toolmap.message and self.general.max_messages > self.messages:
                        if self.matches[iid] or round >= 2: asked = self.__reveal__(iid)

                    if asked: continue
                    
                    energy = self.__explore__(iid, is_ref, math.ceil(w * round_volume))
                    self.__meta__(iid, is_ref)

                    if not is_ref:
                        queue = self.__get_view_queue__(iid, False)
                        if queue: energy = self.__view__(iid, is_ref, queue, energy)
                    else:
                        self.__reference__(iid)
            else:
                self.__rank__()
                self.__bundle__()
                self.__finish__()

            round += 1

        return self.dlg

    def __summarize__(self) -> None:
        messages = [
            {"role": "system", "content": Prompt.SUMMARIZE_SYS},
            {"role": "user", "content": Prompt.SUMMARIZE_USER.format(query = self.query, tools = Misc.dump_json(self.tools, 2))}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_summarize_content(content, [t["function"]["name"] for t in self.tools])

            if js:
                self.__update_with_summarize_report__(js)
                break

            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

    def __explore__(self, iid: int, is_ref: bool, energy: int) -> int:
        while energy > 0 and not self.finished:
            if self.__remaining_steps__() <= 5 + len(self.intents) and not self.ever_stock_event: break
            if self.__remaining_steps__() <= 4: break

            valid_ids = self.__get_product_ids__(iid, False, is_ref)
            if len(valid_ids): break

            messages = self.__get_explore_messages__(iid, is_ref)
            tools = self.__get_tools__(["search"])

            while energy > 0 and not self.finished:
                if self.__remaining_steps__() <= 5 + len(self.intents) and not self.ever_stock_event: break
                if self.__remaining_steps__() <= 4: break

                content, tool_calls = Proxy.llm(messages, tools, True, LLM.GPT6)

                if tool_calls:                    
                    for tc in tool_calls:
                        tc["function"]["arguments"] = Misc.lower_vals(tc["function"]["arguments"])

                        raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)
                        self.steps += 1

                        self.dlg.append({"role": "assistant", "content": content, "tool_call": tc, "environment_result": raw})
                        energy -= 1

                        if not self.alias: self.__mapping__(obs)

                        if self.alias:
                            ids = self.__read_listings__(obs, False, iid, is_ref)
                            obs = Misc.eliminate_listings(obs, self.alias["product_id"])
                        else:
                            ids = []

                        self.history.append((tc, iid, is_ref, ids, obs))
                        self.explore_pending[(iid, is_ref)] += 1

                        break

                    break
                else:
                    js, err = Parser.parse_explore_content(content)

                    if js:
                        if not self.explore_pending[(iid, is_ref)]:
                            if IS_DEBUG: print(f"### Invalid termination.")
                            
                            messages.extend(
                                [
                                    {"role": "assistant", "content": content},
                                    {
                                        "role": "user",
                                        "content": """
Termination rejected! You can't terminate without search. Start your own session.
(If no new promising keywords choice you can take, try filters-based mode with category and price cap options.)
""".strip()
                                    }
                                ]
                            )
                            continue

                        self.explore_pending[(iid, is_ref)] = 0
                        break

                    if IS_DEBUG: print(f"### Bad output: {err}")
                    messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

        return energy

    def __mapping__(self, obs: Any):
        messages = [
            {"role": "system", "content": Prompt.MAPPING_SYS},
            {"role": "user", "content": Prompt.MAPPING_USER.format(samples = obs)}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_mapping_content(content)

            if js:
                if "result" not in js: self.alias = js
                break

            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

    def __meta__(self, iid: int, is_ref: bool) -> None:
        if self.meta_try[(iid, is_ref)] >= 3: return

        candidates = {
            pid: Misc.shorten_listing(details, self.__stop_keys__(False))
            for pid, details in self.products.items() if iid in (details["ref_iids"] if is_ref else details["base_iids"])
        }
        if len(candidates) < 5 and not is_ref: return
        if len(candidates) < 3 and is_ref: return

        if self.meta_len[(iid, is_ref)] == len(candidates): return
        self.meta_len[(iid, is_ref)] = len(candidates)

        messages = [
            {"role": "system", "content": Prompt.META_SYS},
            {
                "role": "user",
                "content": Prompt.META_USER.format(
                    requirements = self.intents[iid].reference if is_ref else self.intents[iid].queries, candidates = candidates
                )
            }
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_meta_content(content)

            if js:
                old_category = self.intents[iid].ref_category if is_ref else self.intents[iid].base_category

                if is_ref:
                    self.intents[iid].ref_category = js["category"]
                    self.intents[iid].ref_brand = js["brand"]
                    self.intents[iid].ref_shop = js["shop"]
                else:
                    self.intents[iid].base_category = js["category"]
                    self.intents[iid].base_brand = js["brand"]
                    self.intents[iid].base_shop = js["shop"]

                if old_category is not None and old_category != js["category"]:
                    self.meta_try[(iid, is_ref)] = 0
                else:
                    self.meta_try[(iid, is_ref)] += 1

                break
            
            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

    def __inspect__(self, pid: str) -> None:
        info = [tool for tool in self.tools if tool["function"]["name"] in self.toolmap.inspect]
        tc_buff, obs_buff = self.__examine_inspect_tools__(pid, False)        

        messages = [
            {"role": "system", "content": Prompt.INSPECT_SYS},
            {
                "role": "user",
                "content": Prompt.INSPECT_USER.format(
                    info = info, view_results = obs_buff, search_results = Misc.shorten_listing(self.products[pid], self.__stop_keys__(False))
                )
            }
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_inspect_content(content)

            if js:
                self.viewer = js["best"]
                self.view_batch = js["capacity"]
                
                self.alias["free_text"] = list(set(js["free_text"] + self.alias.get("free_text", [])))
                break
            
            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

        for func in tc_buff:
            self.history.append((tc_buff[func], None, False, [], Misc.eliminate_listings(obs_buff[func], self.alias["product_id"])))

        self.__read_listings__(obs_buff[self.viewer], True, None, False)

    def __view__(self, iid: int, is_ref: bool, queue: List[str], energy: int) -> None:
        if not queue: return
        cursor = 0

        if self.viewer is None:
            self.__inspect__(queue[0])
            offset = 1
        else:
            offset = 0
        
        for tool in self.tools:
            if tool["function"]["name"] == self.viewer:
                args: Dict = tool["function"]["parameters"]["properties"]
                arg_key = list(args.keys())[0]
                is_list = args[arg_key]["type"] == "array"
                break

        while energy > 0 and not self.finished:
            if self.__remaining_steps__() <= 5 + len(self.intents) and not self.ever_stock_event: break
            if self.__remaining_steps__() <= 4: break

            batch = queue[offset + cursor * self.view_batch:offset + (cursor + 1) * self.view_batch]
            if not batch: break

            arg_val = batch if is_list else ','.join(batch)
            if not is_ref: self.intents[iid].is_safe = False

            tc = {
                "type": "function",
                "index": iid,
                "id": Misc.random_id(),
                "function": {"name": self.viewer, "arguments": Misc.dump_json({arg_key: arg_val})}
            }
            raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)            
            self.steps += 1

            self.dlg.append({"role": "assistant", "content": "", "tool_call": tc, "environment_result": raw})
            self.__read_listings__(obs, True, None, False)

            self.history.append((tc, None, False, [], Misc.eliminate_listings(obs, self.alias["product_id"])))
            
            energy -= 1
            cursor += 1

        return energy

    def __reference__(self, iid: int) -> None:
        queue = self.__get_view_queue__(iid, True)
        if not queue: return

        shortlisted = self.__resolve__(iid, queue, True)
        if not self.intents[iid].reference: return

        queue = shortlisted + [pid for pid in queue if pid not in shortlisted]
        matches, cursor = [], 0

        while not self.finished:
            if self.__remaining_steps__() <= 5 + len(self.intents) and not self.ever_stock_event: break
            if self.__remaining_steps__() <= 4: break
            
            batch = queue[cursor * self.view_batch * self.REF_BATCH:(cursor + 1) * self.view_batch * self.REF_BATCH]
            if not batch: break

            self.__view__(iid, True, batch, 100)
            cursor += 1

            matches += self.__resolve__(iid, batch, True)
            if not self.intents[iid].reference: return

        for pid in matches:
            tc_buff, obs_buff = self.__examine_inspect_tools__(pid, True)

            for func in tc_buff:
                self.history.append((tc_buff[func], None, False, [], Misc.eliminate_listings(obs_buff[func], self.alias["product_id"])))    
                self.__read_listings__(obs_buff[func], True, iid, True)

        self.__resolve__(iid, matches, True)

    def __reveal__(self, iid: int) -> bool:
        if not self.logistics[iid]: self.logistics[iid] = self.__logistics__(iid)
        if not self.logistics[iid]: return False
        
        return self.__ask__(iid)

    def __ask__(self, iid: int) -> bool:
        question = self.__next_question__(iid)
        if not question: return False
        
        for tool in self.tools:
            if tool["function"]["name"] == self.toolmap.message[0]:
                args: Dict = tool["function"]["parameters"]["properties"]
                arg_key = list(args.keys())[0]

                messenger = tool
                break
            
        tc = {
            "type": "function",
            "index": 0,
            "id": Misc.random_id(),
            "function": {"name": messenger["function"]["name"], "arguments": Misc.dump_json({arg_key: question})}
        }
        raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)
        if IS_DEBUG: print(f"**** QUESTION: {question}")
        
        self.messages += 1
        self.steps += 1

        self.dlg.append({"role": "assistant", "content": "", "tool_call": tc, "environment_result": raw})
        self.history.append((tc, None, False, [], obs))

        self.ever_asked[iid] = True
        self.last_asked_iid = iid
        
        return True

    def __finish__(self) -> None:
        messages = self.__get_finish_messages__()        
        tools = self.__get_tools__(["finish"] + (["cart"] if self.best_bundle else []))

        while not self.finished:
            content, tool_calls = Proxy.llm(messages, tools, True, LLM.GPT6)
            
            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                
                for tc in tool_calls:
                    raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)
                    self.steps += 1

                    self.dlg.append({"role": "assistant", "content": content, "tool_call": tc, "environment_result": raw})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(obs)})

                    self.history.append((tc, None, False, [], obs))

                break                    
            else:
                messages += [{"role": "assistant", "content": content}, {"role": "user", "content": "Error: Respond with tool calls only."}]

    def __event__(self) -> None:
        def __has_message__(js: Union[Dict, List, str]) -> bool:
            if isinstance(js, dict): return any(__has_message__(value) for value in js.values())
            if isinstance(js, list): return any(__has_message__(item) for item in js)
            if isinstance(js, str): return len(js) > 20
            return False

        def __check__(js: Union[Dict, List, str]) -> bool:
            if isinstance(js, (dict, list)) and "product_id" in self.alias: js = Misc.eliminate_listings(js, self.alias["product_id"])
            return __has_message__(js)

        def __format_requirements__(general: GeneralInfo, intents: List[IntentInfo]) -> str:
            req = {
                "general": {
                    "budget": general.budget, "ask_one_by_one": general.ask_one_by_one
                },
                "intents": {
                    i: {
                        "queries": it.queries,
                        "reference": it.reference,
                        "min_price": it.min_price,
                        "max_price": it.max_price,
                        "quantity": it.quantity
                    }
                    for i, it in enumerate(intents)
                }
            }
            return Misc.dump_json(req, 2)

        history = [
            {
                "tool": "initialize", "arguments": None,
                "response": f"# Initial Shopping Requirements\n\n{__format_requirements__(self.general_init, self.intents_init)}"}
        ]
        msg_len = 0

        for tc, _, _, _, obs in self.history:
            if tc is None:
                history.append({"tool": "trigger_event", "arguments": obs, "response": "Successfully handled."})
            elif tc["function"]["name"] in self.toolmap.cart or __has_message__(obs):
                history.append({"tool": tc["function"]["name"], "arguments": tc["function"]["arguments"], "response": obs})
                if __has_message__(obs): msg_len += 1

        if msg_len == self.event_msg_len: return
        self.event_msg_len = msg_len

        messages = [
            {"role": "system", "content": Prompt.EVENT_SYS},
            {
                "role": "user",
                "content": Prompt.EVENT_USER.format(requirements = __format_requirements__(self.general, self.intents), history = history)
            }
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_event_content(content, len(self.intents), list(self.products.keys()))

            if js:
                self.__update_with_event_report__(js)
                break

            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

    def __evaluate__(self, iid: int, with_full: bool, inspected: bool) -> None:
        if not self.intents[iid].base_category: return

        candidates = {
            pid: Misc.shorten_listing(self.products[pid], self.__stop_keys__(not with_full)) for pid in self.__get_product_ids__(iid, inspected, False)
        }
        if not candidates: return

        requirements = {"queries": self.intents[iid].queries, "priority": self.intents[iid].priority}

        messages = [
            {"role": "system", "content": Prompt.EVALUATE_SYS},
            {"role": "user", "content": Prompt.EVALUATE_USER.format(requirements = requirements, candidates = candidates)}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_evaluate_content(content, list(candidates.keys()))

            if js:
                self.__update_with_evaluate_report__(js, iid, inspected and not with_full)
                break

            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

    def __rank__(self) -> None:
        if any(not self.matches[iid] for iid in range(len(self.intents))): return
        if self.general.optimize: return

        n = len([iid for iid, it in enumerate(self.intents) if len(self.matches[iid]) > 1 and not it.priority])
        if not n: return

        candidates: List[Dict[str, Dict]] = [
            {
                "intent_id": iid,
                "category": it.base_category,
                "priority": it.priority,
                "candidates": {
                    pid: Misc.shorten_listing(self.products[pid], self.__stop_keys__(False)) for pid in self.matches[iid]
                }
            }
            for iid, it in enumerate(self.intents)
        ]
        messages = [
            {"role": "system", "content": Prompt.RANK_SYS},
            {"role": "user", "content": Prompt.RANK_USER.format(requirements = self.query, candidates = candidates)}
        ]
        prompt = Misc.dump_json(messages)

        if self.rank_prompt == prompt: return
        self.rank_prompt = prompt

        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_rank_content(content, len(self.intents), {group["intent_id"]: list(group["candidates"].keys()) for group in candidates})

            if js:
                for iid, it in enumerate(self.intents):
                    if it.priority: continue
                    
                    ranked: List = js[str(iid)]
                    ranked.reverse()

                    for pid in self.matches[iid]:
                        self.matches[iid][pid][0] = ranked.index(pid)

                break

            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

    def __bundle__(self) -> None:
        quick = self.__quick_bundle__()

        if quick:
            self.best_bundle = self.__deep_bundle__() if self.general.dependencies else quick
        else:
            self.best_bundle = []

    def __deep_bundle__(self) -> List:
        requirements = {
            "general": {
                "budget": self.general.budget,
                "dependencies": self.general.dependencies
            },
            "intents": {
                iid: {
                    "queries": it.queries,
                    "reference": it.reference,
                    "priority" : it.priority,
                    "quantity": it.quantity                    
                }
                for iid, it in enumerate(self.intents)
            }
        }
        candidates = {
            iid: {
                pid: {                    
                    "details": Misc.shorten_listing(self.products[pid], self.__stop_keys__(False)),
                    "matchness_score": self.matches[iid][pid][0],
                    "priority_score": self.matches[iid][pid][1]
                } for pid in sorted(list(self.matches[iid].keys()))
            } for iid in range(len(self.intents))
        }
        messages = [
            {"role": "system", "content": Prompt.BUNDLE_SYS},
            {"role": "user", "content": Prompt.BUNDLE_USER.format(requirements = requirements, candidates = candidates)}
        ]
        prompt = Misc.dump_json(messages)

        if prompt == self.bundle_prompt: return self.best_bundle
        self.bundle_prompt = prompt

        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_bundle_content(content, len(self.intents), list(self.products.keys()))

            if js: return [js["best"][str(iid)] for iid in range(len(self.intents))] if js["best"] else []

            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])        

    def __quick_bundle__(self) -> List:
        candidates = [list(ids) for ids in product(*(list(group.keys()) for group in self.matches))]                
        bundles_ex = []

        for ids in candidates:
            cost_sum = sum([self.products[pid]["price"] * (self.intents[i].quantity or 1) for i, pid in enumerate(ids)])
            if self.general.budget is not None and cost_sum > self.general.budget: continue

            match_sum = sum(self.matches[i][pid][0] for i, pid in enumerate(ids))
            priority_sum = sum(self.matches[i][pid][1] * (1 if self.intents[i].sort_dir == "high" else -1) for i, pid in enumerate(ids))

            bundles_ex.append((ids, (-match_sum, -priority_sum, cost_sum)))

        if not bundles_ex: return []
        bundles_ex.sort(key = lambda be: be[1])

        return bundles_ex[0][0]

    def __logistics__(self, iid: int) -> Optional[Dict]:
        if not self.intents[iid].base_category: return None

        ids = self.__get_product_ids__(iid, True, False)
        matches = list(self.matches[iid].keys())

        ids = matches + [pid for pid in ids if pid not in matches]
        if len(ids) < 5 and not matches: return None
        
        samples = {pid: Misc.shorten_listing(self.products[pid], self.__stop_keys__(True)) for pid in ids[:6]}

        req = {
            "encourage": self.general.speak,
            "category": self.intents[iid].base_category,
            "queries": self.intents[iid].queries,            
            "priority": self.intents[iid].priority
        }
        messages = [
            {"role": "system", "content": Prompt.LOGISTICS_SYS},
            {"role": "user", "content": Prompt.LOGISTICS_USER.format(requirements = req, samples = samples)}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            
            js, err = Parser.parse_logistics_content(content)
            
            if js:
                js["properties"] = js["properties"][:Engine.MAX_NEEDS]
                js["questions"] = []                
                return js

            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

    def __resolve__(self, iid: int, ids: List[str], allow_discovery: bool) -> List[str]:
        candidates = {pid: Misc.shorten_listing(self.products[pid], self.__stop_keys__(False)) for pid in ids}
        if not candidates: return []

        requirements = {
            "base_queries": self.intents[iid].queries, "reference": self.intents[iid].reference
        }
        messages = [
            {"role": "system", "content": Prompt.RESOLVE_SYS},
            {"role": "user", "content": Prompt.RESOLVE_USER.format(requirements = requirements, candidates = candidates)}
        ]
        while True:
            content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
            js, err = Parser.parse_resolve_content(content, ids)

            if js: return self.__update_with_reference_report__(js, iid, allow_discovery)

            if IS_DEBUG: print(f"### Bad output: {err}")
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": f"Error: {err}"}])

    def __examine_inspect_tools__(self, pid: str, exclude_viewer: bool) -> Tuple[Dict, Dict]:
        tc_buff, obs_buff = {}, {}

        for func in self.toolmap.inspect:
            if exclude_viewer and func == self.viewer: continue

            for tool in self.tools:
                if tool["function"]["name"] == func:
                    args: Dict = tool["function"]["parameters"]["properties"]
                    arg_key = list(args.keys())[0]
                    is_list = args[arg_key]["type"] == "array"
                    break

            tc = {
                "type": "function",
                "index": 0,
                "id": Misc.random_id(),
                "function": {"name": func, "arguments": Misc.dump_json({arg_key: [pid] if is_list else pid})}
            }
            raw, obs, self.finished = Proxy.env(self.binding, self.steps + 1, tc)
            self.steps += 1

            self.dlg.append({"role": "assistant", "content": "", "tool_call": tc, "environment_result": raw})
            
            obs_buff[func] = obs
            tc_buff[func] = tc

        return tc_buff, obs_buff

    def __read_listings__(self, obs: List | Dict, is_inspect: bool, iid: Optional[int], is_ref: bool) -> List[str]:
        records = Misc.extract_listings(obs, self.alias["product_id"])
        ids = []

        for r in records:
            pid = r[self.alias["product_id"]]
            ids.append(pid)
            
            if pid in self.products:
                self.products[pid]["base_iids"] = list(set(self.products[pid]["base_iids"] + ([] if iid is None or is_ref else [iid])))
                self.products[pid]["ref_iids"] = list(set(self.products[pid]["ref_iids"] + ([] if iid is None or not is_ref else [iid])))
                
                if not is_inspect: continue

            listing = {"is_inspected": is_inspect, "is_available": None, "stock_amount": None}
            
            for k, v in self.alias.items():
                if k in ["free_text", "product_id"]: continue
                listing[k] = r.get(v) if v else None

            for k, v in r.items():
                if any(k == a for a in self.alias.values()): continue
                listing[k] = v

            if pid not in self.products:
                if listing["is_available"] is None: listing["is_available"] = True

                listing["base_iids"] = [] if iid is None or is_ref else [iid]
                listing["ref_iids"] = [] if iid is None or not is_ref else [iid]
                
                self.products[pid] = listing
            else:
                for k, v in listing.items():
                    if self.products[pid].get(k) is None or k == "is_inspected": self.products[pid][k] = v

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

                if it.base_shop is not None and not Misc.has_evidence(details, it.base_shop): is_valid_base = False
                if it.ref_shop is not None and not Misc.has_evidence(details, it.ref_shop): is_valid_ref = False

                if it.base_brand is not None and not Misc.has_evidence(details, it.base_brand): is_valid_base = False
                if it.ref_brand is not None and not Misc.has_evidence(details, it.ref_brand): is_valid_ref = False
                
                if is_ref is not None:
                    if not is_ref and not is_valid_base: continue
                    if is_ref and not is_valid_ref: continue
                else:
                    if not is_valid_base and not is_valid_ref: continue

            ids.append(pid)
        
        return ids

    def __update_with_summarize_report__(self, js: Dict) -> None:
        self.general: GeneralInfo = GeneralInfo.from_json(js["general"])
        self.intents: List[IntentInfo] = [IntentInfo.from_json(obj) for obj in js["intents"]]
        self.toolmap: ToolMap = ToolMap.from_json(js["toolmap"])

        for it in self.intents:
            if not it.quantity: it.quantity = 1

            if self.general.budget is not None:
                it.max_price = min(it.max_price if it.max_price is not None else 1e10, self.general.budget // it.quantity)

        self.intents_init: List[IntentInfo] = [IntentInfo.from_json(it.to_json()) for it in self.intents]
        self.general_init: GeneralInfo = GeneralInfo.from_json(self.general.to_json())
        
        self.ever_asked = [False for _ in self.intents]
        self.matches = [{} for _ in self.intents]
        
        self.logistics: List[Dict[str, List[str]]] = [None for _ in self.intents]
        self.logistics_cursor = [0 for _ in self.intents]

        self.round_volume = (self.general.max_steps - self.general.max_messages - len(self.intents) - len(self.toolmap.inspect) - 2) // 6
        self.round_volume = min(max(self.round_volume, len(self.intents) * 4), len(self.intents) * 10)

    def __update_with_event_report__(self, js: Dict[str, Dict]) -> None:
        event = js["event"]

        if event in ["requirements", "both"]:
            affected_iids = [int(i) for i in js["affected_intents"]]
            req = js["new_requirements"]            

            if req["general"]["ask_one_by_one"] and self.last_asked_iid is not None and not self.general.ask_one_by_one:
                self.logistics_cursor[self.last_asked_iid] = len(self.logistics[self.last_asked_iid]["questions"])

            self.general.ask_one_by_one = req["general"]["ask_one_by_one"]
            self.general.budget = req["general"]["budget"]            

            for iid in affected_iids:
                it = req["intents"][str(iid)]

                self.intents[iid].min_price = it["min_price"]
                self.intents[iid].max_price = it["max_price"]
                self.intents[iid].quantity = it["quantity"]
                self.intents[iid].queries = it["queries"]
                self.intents[iid].reference = it["reference"]
                self.intents[iid].is_safe = False

                self.matches[iid].clear()

        if event in ["stock", "both"]:
            stock = js["new_stock_status"]

            for pid, inf in stock.items():
                prod = self.products[pid]

                if inf["new_availibility"] is not None: prod["is_available"] = inf["new_availibility"]
                if inf["new_stock_amount"] is not None: prod["stock_amount"] = inf["new_stock_amount"]
                if inf["new_price"] is not None: prod["price"] = inf["new_price"]
                if inf["suspicious"]: prod["is_available"] = False

            self.ever_stock_event = True

            for iid, shortlist in enumerate(self.matches):
                valid_ids = self.__get_product_ids__(iid, None, False)

                for pid in list(shortlist.keys()):
                    if pid not in valid_ids: del shortlist[pid]

        if event: self.history.append((None, None, False, [], js))

    def __update_with_evaluate_report__(self, js: Dict, iid: int, is_strict: bool) -> None:
        self.intents[iid].sort_dir = js["priority_preference"]
        self.intents[iid].is_safe = True
        self.intents[iid].is_strict = is_strict and js["shortlisted"]

        self.matches[iid] = js["shortlisted"]

    def __update_with_reference_report__(self, js: Dict, iid: int, allow_discovery: bool) -> List[str]:
        result = js["result"]
        if not result: return js["shortlisted"]

        if result == "discovered":
            if not allow_discovery: return js["shortlisted"]

            self.intents[iid].queries = js["new_requirements"]["base_queries"]
            self.intents[iid].reference = None
            self.intents[iid].is_safe = False
        else:
            self.intents[iid].reference = js["new_requirements"]["reference"]
            self.explore_pending[(iid, True)] = 0

        for _, details in self.products.items():
            if iid in details["ref_iids"]: details["ref_iids"].remove(iid)

        self.intents[iid].ref_category = None
        self.intents[iid].ref_brand = None
        self.intents[iid].ref_shop = None

        return []

    def __get_explore_messages__(self, iid: int, is_ref: bool) -> List:
        valid_ids = self.__get_product_ids__(iid, None, is_ref)
        
        history, ids_seen = [], []
        it = self.intents[iid]

        for tc, i, ref, ids, obs in self.history:
            if tc is None or tc["function"]["name"] not in self.toolmap.search: continue
            if i != iid or ref != is_ref: continue

            listings = {}
            
            for pid in ids:
                if pid in valid_ids:
                    if pid not in ids_seen:
                        details = Misc.shorten_listing(self.products[pid], self.__stop_keys__(True))
                        ids_seen.append(pid)
                    else:
                        details = "[SEEN]"
                else:
                    details = "[MISMATCH]"

                listings[pid] = details

            obs = {
                "statistics": obs,
                "items_in_this_page": len(listings),
                "retrievals": listings
            }
            history.append({"Tool parameters used": tc, "Search result": obs})

        if not is_ref:
            req = {"queries": it.queries, "min_price": it.min_price, "max_price": it.max_price}
            if it.base_category: req["category"] = it.base_category

            if len(self.matches[iid]) >= 2 and (self.intents[iid].priority or self.general.optimize):
                prices = sorted([self.products[pid]["price"] for pid in self.matches[iid]])
                req["max_price"] = prices[1]
        else:
            req = {"queries": [it.reference]}
            if it.ref_category: req["category"] = it.ref_category

        fixed_history = history[:-self.explore_pending[(iid, is_ref)]]
        if fixed_history: fixed_history[-1]["History end"] = "This is the end of the previous history. Start a new session!"

        messages = [
            {"role": "system", "content": Prompt.EXPLORE_SYS},
            {"role": "user", "content": Prompt.EXPLORE_USER.format(requirements = req, history = fixed_history)}
        ]
        for h in history[-self.explore_pending[(iid, is_ref)]:]:
            tc, obs = h["Tool parameters used"], h["Search result"]

            messages.append({"role": "assistant", "content": "", "tool_calls": [tc]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(obs)})

        return messages

    def __get_finish_messages__(self) -> List:
        if self.best_bundle:
            history, candidates = [], []

            for tc, _, _, _, obs in self.history:
                if tc is None or tc["function"]["name"] not in self.toolmap.cart: continue
                history.append({"tool_call": tc, "tool_response": obs})

            for i, pid in enumerate(self.best_bundle):
                candidates.append({"product_id": pid, "quantity": self.intents[i].quantity or 1})

            return [
                {"role": "system", "content": Prompt.CHECKOUT_SYS},
                {"role": "user", "content": Prompt.CHECKOUT_USER.format(candidates = candidates, history = history)}
            ]
        else:
            return [
                {"role": "user", "content": Prompt.DECLINE_USER}
            ]

    def __next_question__(self, iid: int) -> Optional[str]:
        questions = self.logistics[iid]["questions"]
        props = self.logistics[iid]["properties"]

        category = self.intents[iid].base_category
        cursor = self.logistics_cursor[iid]
        
        if cursor < len(questions):
            self.logistics_cursor[iid] += 1
            q = questions[cursor]

            return f"For {category}, {q.lower()}" if len(self.intents) > 1 else q
        
        if cursor < len(questions) + len(props):
            if self.general.ask_one_by_one:
                p = props[cursor - len(questions)]
                self.logistics_cursor[iid] += 1

                return f"For {category}, what {p} would you like?" if len(self.intents) > 1 else f"What {p} would you like?"

            self.logistics_cursor[iid] = len(questions) + len(props)

            if len(props) > 1:
                q = ", ".join(props[:-1]) + " or " + props[-1]
                return Prompt.QUESTION_TEMPLATE.format(intent = f" for {category}" if len(self.intents) > 1 else "", q = q).replace('\n', " ")
            else:
                q = props[0]
                return f"For {category}, what {q} would you like?" if len(self.intents) > 1 else f"What {q} would you like?"

        return None

    def __get_view_queue__(self, iid: int, is_ref: bool) -> List[str]:
        ids = self.__get_product_ids__(iid, False, is_ref)
        
        if not is_ref: ids = sorted(ids, key = lambda pid: self.products[pid]["price"])
        return ids[:Engine.INSPECT_BATCH * self.view_batch]

    def __prioritize_jobs__(self) -> List[Tuple[int, bool, float]]:
        jobs_ex = []

        for iid in range(len(self.intents)):
            if self.intents[iid].reference: jobs_ex.append((True, True, True, True, True, True, 0, -iid))

            jobs_ex.append((
                False,
                len(self.matches[iid]) == 0,
                not self.intents[iid].is_strict,
                len(self.matches[iid]) < 2 and not self.ever_stock_event,
                not self.ever_asked[iid],
                bool(self.intents[iid].priority),
                -len(self.__get_product_ids__(iid, True, False)),
                -iid
            ))

        jobs_ex.sort(reverse = True)
        
        weight = [sum(jx[:-2]) for jx in jobs_ex]
        wsum = sum(weight)

        if not wsum:
            weight = [1 for _ in jobs_ex]
            wsum = len(jobs_ex)
        
        return [(-jx[-1], jx[0], w / wsum) for jx, w in zip(jobs_ex, weight)]

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

    def __stop_keys__(self, free_text: bool) -> List[str]:
        return ["is_inspected", "base_iids", "ref_iids", "stock_amount", "is_available"] + (self.alias.get("free_text", []) if free_text else [])

    def __is_urgent__(self) -> bool:
        if IS_DEBUG: return False

        elapsed = time.monotonic() - self.start_tick
        return elapsed > Engine.MAX_DURATION - 40.0

def agent_main(prob_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    eng = Engine(prob_data)
    return eng.run()
