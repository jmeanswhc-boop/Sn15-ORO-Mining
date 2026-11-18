from src.agent.proxy_client import ProxyClient
from collections import defaultdict
from typing import Optional, Any, Dict, List, Tuple
from itertools import product
from enum import Enum

import secrets
import string
import math
import copy
import time
import json
import re

class LLM(str, Enum):
    DEEPSEEK = "deepseek-ai/DeepSeek-V3.2-TEE"    
    SONNET = "anthropic/claude-sonnet-5.5"
    HAIKU = "anthropic/claude-haiku-5.5"
    GPT6 = "openai/gpt-6-luna"

class Prompt:
    PARSE_SYS = """
You are an agent to analyze the user's shopping requirements and available tools. You are given:
    - a raw text input from user
    - a set of tools for shopping assistance

Your mission:
    - parse user's verbose input to get the structured analysis
    - classify the given tools

Return JSON object like:
{{
    "general": {{
        "budget": {{
            "amount": <int|null>, // user's budget if mentioned, otherwise null
            "unit": <str|null> // currency unit of user budget if mentioned, otherwise null
        }},
        "max_messages": <int|null>, // maximum number of messages that the agent can send to the user, if mentioned
        "max_steps": <int|null>, // maximum number of interaction moves that the assistance agent can take, if mentioned
        "substitute_allowed": <bool>, // true if substitute submission is allowed if no matching product is found, otherwise false
        "fallback": <str|null>, // the principle of agent for when no product is matching (set null if substitute submission is not allowed)
        "instructions": [ // additional instructions that are beyond the above fields but agent must adhere
            <str>, ...
        ]
    }},
    "intents": [ // list the product requirements per product intent (user may have multiple different product intents)
        {{
            "price": {{
                "min": <int|null>, // lower bound of price if mentioned, otherwise null
                "max": <int|null>, // upper bound of price if mentioned, otherwise null
                "unit": <str|null> // currency unit of price range if mentioned, otherwise null
            }},
            "quantity": <int>, // quantity that user wants to buy for this product if mentioned, otherwise 1
            "priority": <str|null>, // a short phrase for special condition with which user prefers most, if exists
            "queries": [
                <str>, ...
            ] // a text dedicated to the requirements and informative contexts for this product intent beyond the above fields
        }},
        ... ...
    ],
    "tools": {{
        "search": [ // names of the tool functions for finding products with search keywords and/or options
            <str>, ...
        ],
        "inspect": [ // name of tool functions for viewing/comparing details of product(s)
            <str>, ...
        ],
        "cart": [ // names of the tool functions for cart operations: add, remove, review, confirm, etc.
            <str>, ...
        ],
        "finish": [ // names of the tool functions for finalization: order/submission or give-up
            <str>, ...
        ],
        "message": [ // names of the tool functions for messaging to the user
            <str>, ...
        ],
        "other": [ // names of the tool functions that can be classified into none of above
            <str>, ...
        ]
    }}
}}

Field Reference:
    - "general" is the requirements for the agent and all product intents commonly, and "intents" are the list of requirements per individual intent.
    - "general/budget" is the upper limit of the sum of `price * quantity` across all intents. Set null if not mentioned.
    - "general/fallback" is the rule that agent MUST follow when no product is found matching for an intent. Examples:
        * "If nothing fits all conditions, say so and don't recommend a substitute."
        * "If nothing fits the bill, choose the one that falls short on the fewest requirements. Stock and budget non-negotiable. One costing up to 5% more than the cheapest of those would work."
    - "general/instructions" is the list of all the rules that agent MUST adhere during the process, except for those related to "budget", "max_messages", "max_steps" and "fallback". Examples:
        * ["Check at least 3 candidates per intent.", "Always say you're ready just before ordering."]
        * ["Don't bother me with messages."]
    - Each intent's "quantity" is the answer to "How many for this product?". Pack size (e.g. "20 pieces" or "50ml volume") is NOT the "quantity" but a required property that must be in "queries".
    - Each intent's "priority" is a short phrase for the criterion based on which the agent must pick the top one from the matching candidates. Always start with "Choose". Examples:
        * "Choose the cheapest."
        * "Choose the one with the shortest diameter."
    - Each intent's "queries" is the list of short sentences including all the requirements as well as the informative contexts, except for those related to "price", "quantity" and "priority". Examples:
        * ["Comic style book written by Arthur Kailey.", "It must be in French.", "Story about dragons."]
        * ["Fans with 12v input voltage compatibility."]
        * ["Basketball shoes with whatever size is given on the listing sold by Levin's in silver.", "Fit for students."]

Guidelines:
    - Ignore user's verbose story and subjective explanation that are not meaningful for e-commerce search and ranking.
        * General phrases for emphasis of needing match like "I'm short of money so strictly meet the budget." are non-informative.
        * Phrases like "must be in stock" or "must be on hand" are non-informative, because search always tries to find the stock-available ones.
        * Phrases like "don't suggest substitute" are non-informative, because they must be specified in "general/instructions".
        * Phrases like "its appearance should be assessable from its listing photos ..." are non-informative, because search engine is only text-based.
        * Phrases related to listing photos are non-informative. Exclude them from "queries".
        * Phrases like "... for class this winter" must be included in "queries" because it may include some requirements.
        * Remove what are trivially non-informative for search and ranking.
    - Each object of "intents" list must be dedicated to one product intent. Do NOT mix it with "general", which is for the common requirements.
        * For exmple, a phrase related to "general" should not be included in an individual intent object's "queries" field.
        * For completeness, repeat the corresponding fields for several intents if they explicitly and clearly share some common requirements.
    - For each intent object, the children field values ("queries", "priority", "price", "quantity") must be strictly non-overlapped.
        * Every distinct requirement must be assigned to exactly one field within its corresponding intent object.
        * Do not duplicate or paraphrase the same requirement across fields.
        * When a requirement combines multiple dimensions, split it into independent requirements where possible.
    - Respect the user's original wording and number/unit formatting for "instructions" and "queries", while keeping the writing style formal.
        * Do NOT convert price/budget numbers and currency units by your own discretion. Just use the format as mentioned by user.
        * If price/budget was mentioned with both large and small units, use the small unit for output. e.g. "My budget is 210cts (2.1$)." -> `"budget": {{ "amount": 210, "unit": "cts" }}`.
        * Never replace a specific model or attribute with a broader interpretation.
        * Do NOT change wording. For example, "ridge line" -> "stripe pattern" is forbidden!
    - In "queries" field, write each sentence clearly and accurately to clarify which requirements are must-have and which requirements are optional.
        * In later process, only your output for requirements will be used for searching and ranking, without user's raw input.
        * Therefore, you MUST write "queries" sentences correctly without any possibility of confusing or misunderstanding.
        * Write like "... is preferred but not strictly required." for the requirements which user explicitly mentioned is not a must.        
    - Do NOT omit or paraphrase any informative details in "queries" to prevent losing requirements.
    - Make sure all given tools are included in "tools". Every tool must be classified into exactly one tool class.
    - Return JSON only. No plain text.

Examples of Incorrect Item for "queries" List:
    - "2 sun glasses ..." ("2" must be moved to "quantity" field, not here.)
    - "... with the least price." ("with the least price" must be moved to "priority" field as "Choose the cheapest.", not here.)
    - "... less than 200 cents." ("less than 200 cents" must be moved to "price" field, not here.)
    - "A dress for visiting Maxim on Thursday." ("for visiting Maxim on Thursday" must be removed, because it's non-informative for e-commerce search.)
""".strip()

    PARSE_USER = """
# User Input
{query}

# Available Tools
{tools}
""".strip()

    EXERCISE_SYS = """
You are an agent to assist user's shopping. You are given:
    - user's shopping requirements
    - a set of special tools
        * a report tool for sending the required information
        * an event tool for triggering shopping requirements or database update
    - a set of shopping/interaction tools
        * search tool(s) for finding products with search keywords and/or options
        * inspect tool(s) for viewing product details        
        * other tools

You are responsible with the initial a few steps. While strictly adhering the rules listed in "general/instructions" field of shopping requirements, you MUST:
    - try ALL search and inspect tools with non-empty results, to understand the way they work.
    - analyze the search tool results to identify:
        * the field names used for product ID, category, title, price, shop, free texts, etc. in the retrieval listing details        
        * the correct category value for every product intent
        * the correct shop names for reference products if exist
    - analyze the inspect tool results to identify:
        * the best inspect tool that returns the property map
        * the field names used for product ID and free texts in the best inspect tool's product details
    - report the identified information as soon as possible.
    - optimize the number of effective tool calls.

Challenges:
    - User may have several product intents.
    - Search tools may have switch arguments.
        * Certain switches may enable/disable other arguments.
        * For example, a certain switch may disable "max_price"-like option argument.
    - Different inspect tools showcase the product details with different levels.
        * Certain inspect tools only reveal the product listing details similar to what search tools produced.
        * Certain inspect tools reveal the comprehensive property map that search tools don't output.
    - We have no pre-knowledge about the terminology of product database and the structure of each tool's result JSON object.

Termination criteria:
    - You have got non-empty listings from every search tool at least once.
    - You have seen non-empty product details from every inspect tool at least once.
    - You have identified the category value for all product intents.
    - You have identified the best inspect tool that gives the property map that search tool can't.

{category_hint}
    - And then you must identify the correct category value for the intent, which is most aligned to the user's requirements.
        * Example 1
            User said: "shoes"
            Category value candidates: "Shoes", "Clothing & Shoes"
            Most aligned choice: "Shoes"
        * Example 2
            User said: "shirts or jackets"
            Category value candidates: "Shirts", "T-shirts", "Jackets", "Shirts and Jackets"
            Most aligned choice: "Shirts and Jackets"
    - Validity of candidate listings can only be confirmed after the category value has been identified.

Best inspect tool priority:
    (1) The tool that produces the property map details that are not retrieved from search tools, is preferred.
    (2) The tool that can accept more product IDs as an argument at one call, is preferred over less or single ID.
    (3) The tool with less number of arguments, is preferred.

Recommended strategy:
    - In the early steps, try ALL search tools for the first product intent, with different argument setups. That way, you can get initial insight in how search tools work.
    - In later steps, try to retrieve matching candidates and identify the category value for all intents. Keep updating your insight in search tool arguments behavior by learning the tool responses.
    - Identify the category values for all intents from the search retrievals.
    - Choose one promising candidate. (From any intent, that's alright.)
    - Try ALL inspect tools for the chosen candidate, and identify the best inspect tool.
    - Use the report tool to send the collected information.

{search_hint}

For report tool, you must pass a JSON-parsable string for the following object to the argument `value`:
{{
    "categories": {{ // correct category values for product intents
        <intent_id_1, str>: <category_value_1, str>, ...
    }},
    "reference_shops": {{ // reference product's shop name required for product intents (set null if the intent has no reference product or its reference product has no shop name required)
        <intent_id_1, str>: <reference_shop_name_1, str|null>, ...
    }},
    "search_tool_insight": {{
        "product_id": <str>, // key name of the field used for product ID
        "category": <str>, // key name of the field used for category
        "price": <str>, // key name of the field used for price
        "title": <str|null>, // key name of the field used for title, if corresponding field exists
        "shop": <str|null>, // key name of the field used for shop, if corresponding field exists
        "is_available": <str|null>, // key name of the field used for stock-availability as boolean, if corresponding field exists
        "stock_amount": <str|null>, // key name of the field used for current stock amount as integer, if corresponding field exists
        "free_text": [<str>, ...] // key names of the fields used for free lengthy texts like title, description, summary, etc.
    }},
    "inspect_tool_insight": {{
        "best": <str>, // name of the best inspect tool
        "capacity": <int>, // maximum number of product IDs that the best inspect tool can accept (1 if it only accepts single)
        "product_id": <str>, // key name of the field used for product ID (with best inspect tool)
        "free_text": [<str>, ...] // key names of the fields used for free lengthy texts like title, description, summary, etc. (with best inspect tool)
    }}
}}

(Important) For "free_text", list only the field key names that have free lengthy text values like title, description, summary, hint, etc.
    - Do NOT list the field name whose value type is a dictionary (like attribute map).
    - Do NOT list the field name whose value type is not string.
    - Do NOT list the field name whose value type is string but not as lengthy as title for most cases by common sense.

Guidelines:
    - The instructions from "general/instructions" field of shopping requirements is your TOP PRIORITY rules!
        * If there is a conflict between some instructions and strategies, the one listed in "instructions" field overrides another.
        * Proceed while strictly keeping the rules of "general/instructions".
    - As with Recommended Strategy, optimize the effective tool call steps before reporting. But some delay is acceptible if it's enforced by "general/instructions".
        * Each effective tool call consumes one step.
    - Before reporting, make sure you have tried all search tools and all inspect tools, at least once with non-empty result for each one. Do NOT skip any.
    - The `value` argument for reporting tool must be a JSON-parsable string. Strictly keep the JSON structure as instructed above.
    - You are responsible with initial a few steps. Therefore,
        * Do NOT aggressively dig into one intent. Instead, allocate the tool call resources equally across the intents.
        * Do NOT trigger cart operations and order/decline actions even if you found a good matching product.
        * Do NOT send messages (including questions) to the user yet, unless it's enforced by "general/instructions".
    - Focus on the base products themselves.
        * Your key task of initial steps is to identify the category value of the base product.
        * Therfore, do NOT waste resources trying to search for the reference products to reveal the hidden property values.
        * (Important) Ignore the "Reference First" search strategy hint for this initial phase!
    - Respond with tool calls only. No plain text.
""".strip()

    EXERCISE_USER = """
# Initial Shopping Requirements
{requirements}
""".strip()

    EVALUATE_SYS = """
You are given:
    - user's shopping requirements (user may have several product intents)    
    - a list of candidate product details per intent
    - a list of details for the products that are potential references

You must process the intents with the following IDs:
{ids}

Your process for each intent:
    - Identify the hidden property values from the reference products, if applicable.
    - Interpret requirements from "queries" into key-value formatted template.
    - Separate must-have and optional requirements from the context.
    - Evaluate the candidate products.
    - Sort the candidates to get the top 7.

Hidden property values from the reference:
    - An intent may point to a reference product for a required property, instead of explicit value. Example:
        * "Color must be like of shoes sold in Karl's shop in white."
    - You must identify the correct reference product from the potential reference list, to extract the required property value and complete the must-have requirements.
    - If you can't find the reference product from the potential reference list, pend this requirement and mark all candidates as unmatched.

Interpret requirements:
    - An intent's "queries" contain user's raw input.
        * You must identify the phrases that have evidence key-value pairs in the candidate details.
        * The phrases that have no corresponding evidence pair, are the non-informative contexts that must be ignored in evaluation.
    - Make template pairs by learning the given candidate details. Example:
        * "suitable for men and women" -> `"gender": "unisex"`
        * "that comes with plastic bag" -> `"pack_type": "plastic bag"`
        * "made with cotton" -> `"material": "cotton"`
    - Do NOT derive pairs without corresponding evidence. For example,
        * "queries": "It must be well aligned to my boots."
        * `"fit": "air-conditioner"` is NOT evidence.
        * Do NOT derive `"fit": "boots"` unless the exact pair or `"fit": "shoes"` exists.
    - Do NOT create a pair with key "category".
        * The "category" value specified in the requirements is non-negotiable, and all candidates are confirmed by system that they meet "category".
    - You are making a template, not a strict string-matching standard. Therefore, choose the one pair that is most aligned to the original query from multiple similar pairs. Example:
        * "queries": "with cartoon style"
        * `"style": "cartoon"`, `"fa.style": "cartoon"`, `"type": "cartoon"`, ...
        * Choose `"style": "cartoon"`, because all others are paraphrased pairs or less aligned to the original wording.
    - The phrases without corresponding evidence pairs are not requirements. Ignore them for subsequent processes.

Must-have and optional requirements:
    - Optional requirements are what user prefers but not strictly requests.
    - Typically they are explicitly mentioned like "... is preferred but not a must" in "queries".
    - Sometimes they are described as situational context. Example:
        * "I need school bags for winter."
        * `"occasion": "winter"` may be optional requirement, because "for winter" is user's subjective context not likely related to school bag itself.

Principles of evaluating candidates with requirements:
    - Comma allowed for joined values, but plus not allowed.
        * Required `cotton` - `Cotton, Polyester` meets.
        * Required `wool` - `Wool + Nylon` fails.
    - Allow synonyms. Allow extra token for more specified value. But do NOT derive matching aggressively.
        * Required `class` - `school` meets.
        * Required `fit for all genders` - `unisex` meets.
        * Required `green` - `forest green` meets.
        * Required `fruit` - `apple` fails.
        * Required `winter` - `all seasons` fails.
        * Required `zip-up` - `pullover` fails.
    - Do NOT aggressively derive untold requirements.
        * Queries `gift for my dad` - do NOT derive a requirement `gender: men` from it - do NOT reject a candidate due `gender: women`.
    - A or B
        * Required `cotton or linen and nothing else` - `cotten` meets, `linen` meets, but `cotton + polyester` fails.

Choose the top 7 sorted candidate list:
    - A product is a match if it matches all the must-have requirements    
    - Ranking priorities (top to bottom):
        * A matching product is better than unmatching product.
        * A matching product (A) is better than other matching product (B) if:
            (1) A wins B on "priority" criterion, if "priority" specified (e.g. "priority": "Choose the cheapest." --> A is cheaper than B.)
            (2) A satisfies optional requirements that B doesn't
            (3) A's evidences are more aligned than B's to the original wording of "queries"
            (4) (Tie-breaking) A is cheaper than B
        * An unmatching product (A) is better than other unmatching product (B) if:
            (1) A falls short on the fewer must-have requirements
            (2) (Tie-breaking) A is cheaper than B

Return JSON object like:
{{
    <intent_id_1, str>: {{
        "reference_revealed": <bool>, // true if this intent has no reference product pointed or the reference-dependant property value has been completely disclosed, otherwise false
        "template": {{
            <key_1, str>: <value_1, str>, ... // template key-value pairs with evidences
        }},
        "top_7": [ // the sorted list of top 7 candidates (can be less if total candidates are less than 7)
            {{
                "id": <product_id_1, str>,
                "match": <bool> // true if it's matching product, otherwise false
            }},
            ... ...
        ],
    }},
    ... ...
}}

Guidelines:
    - Ignore too verbose phrases that are situational context or subjective explanations.
    - Ignore the requirements like "must be in stock or on hand", because all products are available in stock.
    - (Sort direction) The earlier product of the sorted list "top_7" is the better candidate.
    - Intents MUST be strictly isolated with each other. Do NOT include other intent's candidates in the output list.
    - ALL candidate products are confirmed that they meet "category" values, price range and stock amount conditions. Do NOT double check those requirements.
    - Return JSON only. No plain text.
""".strip()

    EVALUATE_USER = """
# Shopping Requirements
{requirements}

# Candidate Products
{candidates}

# Potential Reference Products
{references}
""".strip()

    LOGISTICS_SYS = """
You are given:
    - user's shopping requirements (user may have several product intents)
    - sample product details for each intent

Now you must prepare questions to ask to the user. There are 2 types of questions:
    (1) The questions that come from the intent's "queries" directly.
        - Sometimes user represented needs ambiguously with several value options, like "A or B". (e.g. "universal plug with 2 or 3 pins")
            * You must ask user to clarify the exact need: "Which matters more to you, 2-pins or 3-pins?"
    (2) The question for a hidden property.
        - User may have untold needs or preferences. User clarifies them only when the agent asks questions correctly targeting the their property names.
            * You must predict the properties that are most likely to hide user's requirement, and ask about them to reveal the hidden needs.

General Guidelines:
    - If user has multiple product intents, always specify the identifiable product name in the question to avoid misunderstanding the questioned intent. Examples:
        * "Which brand do you prefer?" --> Good if user has only one product intent, but very bad if user has multiple product intents.
        * "For motorcycle accessories, which brand do you prefer?" --> Good if multiple intent exist, because it explicitly specified which intent you are asking about. But not good if only one intent exists.
    - Limit the total number of questions per intent to {count}.
    - Type 1 questions are more prioritized over Type 2 questions.

Guidelines for Type 1 Questions:
    - Explicitly ask to clarify which is more important among several values like "Which matters more to you, A or B?"
    - Make one question for every ambiguous pair. Skip Type 1 if no such ambiguous pair exists.

Guidelines for Type 2 Questions:
    - Every question must be about exactly one property for exactly one product intent.
        * Do NOT ask about two or more properties in one question.
        * Do NOT ask about two or more product intents in one question.    
    - Questions must be concise, clear and short. Do NOT add examples when asking, because it may restrict answers unnecessarily. No example is the best.
        * "Which hair color do you like?" --> Good
        * "Which hair color do you prefer, blond or black?" --> Bad (If user prefers bald head, it's awkward to answer to it quickly.)
    - Do NOT ask about the property that has been already specified in "queries" (including ambiguous pairs). Examples:
        * Quesition about color -> Bad, if "navy blue" has been mentioned in the initial requirements
        * Question about length -> Bad, if "choose the longest" has been mentioned in the initial requirements
    - Do NOT ask about price, budget or quantity.
    - Do NOT waste questions. Do NOT ask two or more questions about the same property of one product intent.

Process for each intent:
    - Make Type 1 questions first, if ambiguous pairs to clarify exist.
    - Collect all property names appeared in the product details for this intent, and merge similar ones.
        * Sometimes similar property names are used. (e.g. "style", "fa.style" and "styles")
        * Merge them into one simplest name. (e.g. "style")
    - Drop invalid property names from consideration:
        * Exclude the property names that have been mentioned in the original requirements.
        * Exclude "price"-like property names.
        * Exclude "category"-like property names, because user will not change the category requirement.
        * Exclude "availability"-like property names, because user always requests available products.
        * Exclude all other property names which you can never expect user to have a preference in, by common sense.
    - Count the occurencies of every property name across the product details for this intent.
        * For a property name, add the number of appearances in one product details. (Count the merged ones. Therefore, you may add more than 1 for one product.)
        * Calculate the sum across all product details for a property name. It's the total occurencies of this property.
    - Sort the property names by descending occurencies. Prioritize by common sense for the same occurency number properties.
    - Choose the top promising property names, and make questions from them.

Respond with a JSON object like:
{{
    <intent_id_1, str>: [ // list questions in the priority order
        <question_1, str>, ... ... // limit to {count} questions
    ],
    ... ...
}}

Return JSON only. No plain text.
""".strip()

    LOGISTICS_USER = """
# Shopping Requirements
{requirements}

# Sample Product Details
{samples}
""".strip()

    ASK_SYS = """
You are given:
    - user's shopping requirements (user may have several product intents)
    - questions to ask to the user
    - message tool(s) to send question
    - an event tool to trigger events

You must use message tool to send the following questions one by one in order:
{questions}

{event_hint}

Whenever an event happens, shopping requirements may update.
    - The new requirements are specified in the event tool's response.
    - The new requirements is your new baseline. Ignore old requirements.

Guidelines:
    - Strictly keep the order of the questions. Do NOT skip. Do NOT ask the question again that was answered to.
    - Watch the tool response carefully, and trigger event using the event tool
    - Do NOT trigger event for dummy non-meaningful answers.
    - Respond with tool calls only. No plain text.
""".strip()

    ASK_USER = """
# Shopping Requirements
{requirements}
""".strip()

    EXPLORE_SYS = """
You are an agent to assist shopping. You are given:
    - user's shopping requirements
    - an event tool for triggering shopping requirements or database update
    - search tool(s)
    - and more tools

You must complete the shopping mission
    - using the given tools,
    - by strictly adhering the agent rules specified

Challenges:
    - User may have several product intents.
    - Search tools may have switch arguments.
        * Certain switches may enable/disable other arguments.
        * For example, a certain switch may disable "max_price"-like option argument.
    - Identifying one matching product is NOT the end of the game!
        * If "priority" is specified, you must find the top one based on this criterion among all matching candidates.
        * There may exists a better matching product that meets the optional requirements as well.
    - Shopping requirements may update during the process.
    - Some intents may have reference products. (Such intents' requirements are completed only when reference products have been identified.)

You can be interrupted during the process.
    - Shopping requirements or product price may update by events, which may enforce you to explore more.
    - User may ask a sudden question buried inside a tool response object, which enforces you to answer to it using the messaging tool.

You MUST carefully review the previous tool responses for each turn, because it may include:
    - event message(s)
    - the most recent shopping requirements
    - the insight of matching product information for intents currently discovered
    - available messages left
    - recommendation for your next action

Some product details from the previous tool responses may be masked, to prevent distracting you.
    - Product details for other intent than what you must focus on now, may be masked.
    - Product details that are trivially non-matching for the intent you must focus on now, may be masked. (Price or category mismatch, etc.)
    - Product details that have been showcased in a former tool response, may be masked for simplification.

{search_hint}

{event_hint}

Action priority for your each turn (top to bottom):
    - (TOP PRIORITY) "general/instructions" of the recent shopping requirements are the king rules for you.
        * It overrides all other conflicting recommendations!
        * If it requests you to call a certain tool or send a certain message for this turn, prioritize it most!
    - If the previous tool responses include event message(s) that have not been handled, trigger event using the event tool!
    - If user asks a question to you to clarify something (such question may be buried inside the previous tool responses), answer to it by a message tool, if you have available messages left.
    - Follow the recommendation in the latest tool response.

Guidelines:
    - Always watch the tool responses carefully to prevent missing event messages. If you found, trigger event using event tool.
    - Do NOT call cart operations or submit/decline tools by your own discretion unless it's instructed in "general/instructions" or recommended.
    - Do NOT call the same search tool with the same arguments.
    - Respect the recommended intent you must focus on for exploration. Do NOT call search tool for non-recommended intent!
    - Respond with tool calls only. No plain text.
""".strip()

    CHECKOUT_SYS = """
You are an agent to assist shopping. You are given:
    - user's shopping requirements
    - an event tool for triggering shopping requirements or database update
    - cart operation tools (add, remove, review, etc. for cart)
    - submit tool(s) for placing order
    - and more tools

Normal submission process:
    - Add the products to the cart.
    - Review the cart content.
    - Place order.

You are responsible with the final submission process. For each turn, you MUST:
    - Analyze the previous steps (add/remove history) to make sure there is no non-requested product in the cart.
    - If you find that there are non-requested products added in the cart, remove them.
    - Add the requested candidate products to the cart correctly.
    - Review and order.

Challenges:
    - User may have several product intents. Exactly one candidate must be submitted per intent.
    - Shopping requirements may update during the process.
    - Stock status may update during the process. (e.g. stock-out, price change, etc.)

You may be interrupted during the process.
    - Shopping requirements or product price may update by events, which may enforce to reselect the candidates.
    - A carted product has been found unavailable (stock-out event), which may enforce to reselect the candidates.
    - User may ask a sudden question buried inside a tool response object, which enforces you to answer to it using the messaging tool.

You MUST carefully review the previous tool responses for each turn, because it may include:
    - event message(s)
    - the most recent shopping requirements
    - available messages left

{event_hint}

Action priority for your each turn (top to bottom):
    - (TOP PRIORITY) "general/instructions" of the recent shopping requirements are the king rules for you.
        * It overrides all other conflicting recommendations!
        * If it requests you to call a certain tool or send a certain message for this turn, prioritize it most!
    - If the previous tool responses include event message(s) that have not been handled, trigger event using the event tool!
    - If user asks a question to you to clarify something (such question may be buried inside the previous tool responses), answer to it by a message tool, if you have available messages left.
    - Proceed submission process.

Guidelines:
    - Analyze the previous cart add/remove history carefully to make sure only and all requested candidates are in the cart, before review and order.        
        * Do NOT order while there exist non-requested products in the cart.
        * Do NOT order while there exist requested products outside the cart.
        * Do NOT call cart review tool for the only purpose of cart content check. It's your task to review the add/remove history to derive the current content. Only call review tool just before order.
    - Optimize the number of tool calls by avoiding redudnant actions.
        * Do NOT add the product that has been already added. Do NOT add the non-requested product.
        * Do NOT remove the product that has been already removed. Do NOT remove the requested product.
    - Correctly pass the quantity value when you add a product to the cart, if it's the required argument.
    - Even the submit tool call may be interrupted by event. You must capture it correctly and trigger event.
        * When submit tool call has been refused by stock-out event, remove the invalid product from the cart and re-initiate the submission process from adding cart.
        * When submit tool call has been interrupted by requirements update event,
            remove invalid product and re-initiate the submission process, if the carted product is invalid now (not in the requested list)
            Do NOT get stunned but CALL SUBMIT TOOL ONCE MORE, if the carted product is still valid (in the requested list)
    - Always watch the tool responses carefully to prevent missing event messages. If you found, trigger event using event tool.
    - Do NOT call search tool(s). You are now responsible with submission process only.
    - NEVER give up or decline order! You MUST submit the requested product(s) without fail!
    - (Critical) If only 1 tool call step is left, skip cart review and just call submit tool! (or you will die!)
    - Respond with tool calls only. No plain text.

# Candidate Products Requested to Submit:
{candidates}
""".strip()

    DECLINE_SYS = """
You are an agent to assist shopping. You are given:
    - user's shopping requirements
    - an event tool for triggering shopping requirements or database update
    - decline tool(s) to give up the shopping process
    - and more tools

You are responsible with the decline process. You must use the decline tool to give up the shopping process.

Challenges:
    - Shopping requirements may update during the process.
    - Stock status may update during the process. (e.g. stock-out, price change, etc.)

You may be interrupted during the process.
    - Shopping requirements or product price may update by events, which may provide a chance of retry.
    - A carted product has been found unavailable (stock-out event), which may provide a chance of retry.
    - User may ask a sudden question buried inside a tool response object, which enforces you to answer to it using the messaging tool.

You MUST carefully review the previous tool responses for each turn, because it may include:
    - event message(s)
    - the most recent shopping requirements
    - available messages left

{event_hint}

Action priority for your each turn (top to bottom):
    - (TOP PRIORITY) "general/instructions" of the recent shopping requirements are the king rules for you.
        * It overrides all other conflicting recommendations!
        * If it requests you to call a certain tool or send a certain message for this turn, prioritize it most!
    - If the previous tool responses include event message(s) that have not been handled, trigger event using the event tool!
    - If user asks a question to you to clarify something (such question may be buried inside the previous tool responses), answer to it by a message tool, if you have available messages left.
    - Proceed decline process.

Guidelines:
    - Just use the decline tool to terminate shopping.
    - If stock status event happens, analyze the previous cart add/remove history carefully to identify the product ID which became unavailable or price-changed.
    - Always watch the tool responses carefully to prevent missing event messages. If you found, trigger event using event tool.
    - Do NOT call search tool(s) or cart operation too(s) or submit tool(s). You are now responsible with decline process only.
    - Respond with tool calls only. No plain text.
""".strip()

    ERRAND_USER = """
# Initial Shopping Requirements
{requirements}

(The requirements may update during the process by events.)
""".strip()

    CATEGORY_HINT_PARTIAL = """
What is category?
    - Category is a word or a short phrase representing the core product form or type.
    - It must be included in each of the listing details retrieved by search tools.
    - The exact field name in the details for category may be paraphrased or obfusticated like "department", "prod_type", "xyblz", etc. But note that this terminology is universal across the system.
    - You must recognize the field that is corresponding to the category in the listing details.
""".strip()

    SEARCH_HINT_PARTIAL = """
Search tool hints:
    - Search tools can be narrowed by:
        * a category filter
        * a keyword mode that may only read the first a few tokens
        * a page index
        * an offset for the next page
    - Empty retrieval listing from a search tool may occur when:
        * the passed argument values with switches don't work as you expected, or
        * there exists no product matching all passed search options
    - Search tool argument for free keywords input strictly requires the retrieved products include all the given keywords in its details, if it's enabled. Therefore,
        * do NOT include non-essential tokens like "and", "or", "with", "for", "compatible", "suitable", ... in the free keywords argument.
        * do NOT include keywords like "in stock", "on hand", "available", ... to represent user's requirement that the product must be available in stock, because it's useless. Search results always retrieve stock-available products.
        * if you passed many tokens for free keywords argument and got empty listing, test and see removing which token(s) gives what you are expecting.
    - Search tool argument for free keywords input is insensitive with token-reordering. No need to test this statement.
    - Respect the min/max length and min/max page restrictions of search tool arguments.    

Search strategy hints:
    - Learn by Doing
        * Learn how search tool arguments work with different switches, by doing.
        * Keep updating your insight in arguments behavior from the past tool responses.
    - Reference First!
        * Some intents may point to reference product for a required property, instead of explicit value. (e.g. "Color must be like of shoes sold in Karl's shop in white.")
        * You must identify the reference product first, to reveal the hidden property value required.
        * Only use search keywords for the target product you are searching! Do NOT mix up search keywords.
            Do NOT add keywords dedicated only to the reference product, when you are targeting the base product.
            Do NOT add keywords dedicated only to the base product, when you are targeting the reference product.
    - Good Keywords
        * Search keywords work best when short and value-oriented.
        * The most preferable keywords are the category value plus the wanted values. (if category value has been specified for the intent)
        * If it's found too long from the past results, you should try shorter by removing some tokens one by one.
        * If category value has been specified,
            Keep in mind that there includes at least one token for the category value in the keywords.
            Trying keywords without any category-related token is pointelss, because category value is the non-negotiable requirement.
            Unless there is no wanted value beside category value, trying keywords with category-related tokens only is not recommended too. It may result in too broad.
    - Pagination
        * Compare the reported total count against the rows you have seen, considering the page metrics like index or offset.
        * If it seems to have more listings in the next pages, you can paginate for further retrievals.        
    - Sort
        * Sorting may help you have different listings, if it's supported by arguments.
    - Extending Retrievals
        * Try the lower price cap intentionally to extend search and explore cheaper candidates, if price cap is a supported argument.
        * Try with less search keywords intentionally to get different listings, if free keywords input is a supported argument.
    - Handling Unclear Numbers
        * User may mention ambiguous requirement like "diameter less than 40mm".
        * In that case, "40mm" is the first keyword you must try, because the matching product is most likely to have an attribute like `"diameter": "less than 40mm"` in its details. That's the best match.
        * However, if you can't find such details as expected from several attempts with "40mm", then you should consider other fallback values.
        * Learn from the past retrievals to predict the diameter value user needs. It may be "35mm", "20mm", etc.
    - Never Use Exclusive Keywords
        * User may exclude a value for a certain property like "any brand but Samsung".
        * Adding "samsung" in keywords never helps. It will narrow search for the retrievals including "Samsung" in its details, unlike your target.
        * You must try keywords without "samsung", and manually review the retrieval details to exclude those with brand "Samsung".
    - Handling `"priority": "Choose the cheapest"`
        * Identify the cheapest matching product (P) currently known.
        * Try with the P's price as the price cap to explore cheaper candidates, if price cap is a supported argument.
        * For this trick, you can refer which arguments setup retrieved P and learn from it, except for price cap value. It will increase efficiency.
    - Not Recommended, but Final Fallback
        * If you tried every imaginable methods above but didn't find the best one,
            consider paraphrasing keywords slightly and carefully like "250mm" -> "250", "tshirt" -> "t-shirt", etc.
            consider adding plausible keywords slightly and carefully for undiscovered listings retrieval

(Very Important!) For each search tool call, always clarify the purposes by the following arguments to clarify what this call is being triggered:
    - "intent_id": the intent ID which this tool call is targeting (zero-based index as specified in shopping requirements)
    - "is_reference": true if this tool call is targeting a reference product, otherwise false
""".strip()

    UPDATE_REQUIREMENTS_SYS = """
You are given:
    - user's original shopping requirements (user may want several product intents)
    - event messages that updates the requirements (with questions optionally)

Shopping requirements should update by event messages.
    - An event message is a free text which enforces the update of the original shopping requirements. Possible events include but are not limited to:
        * user brings a requirement (e.g. "I just want gothic style, nothing else.")
        * user modifies an existing requirement (e.g. "Changed my mind. Color must be blue.")
        * user withdraws a requirement (e.g. "Never mind pack type. Let's forget it.")
        * user moves a must-have requirement to optional requirement (e.g. "CPU speed is a wish than a need.")
        * user clarifies the exact need from the ambiguous requirement (e.g. "I'd prefer 3-pin over 2-pin.")
        * user claims or changes the budget. (e.g. "Oh, I forgot to let you know that my budget is 10 dollars.", "Sorry but I can't spend more than 120cts.")
        * user claims or changes the price range of a product. (e.g. "This TV must be less expensive than 1000USD.")
        * user claims or modifies the agent rules. (e.g. "Hey, say 'Hello' to me just before placing order, or I will reject your submission.")
        * and anything that needs update of shopping requirements
    - (WARNING!) User NEVER changes "category" value of any intent in the requirements.
    - An event message can be:
        * a clarified answer to the question that the agent asked ("question" exists)
        * a free text that user voluntarily produced without the question asked ("question" = null)

Return JSON object like:
{{
    "new_requirements": {{ // a complete new shopping requirement object
        "general": {{ ... }},
        "intents": {{ ... }}
    }},
    "affected_intents": [<int>, ...] // list zero-based indice of intents that were modified by this update
}}

Guidelines:
    - The "new_requirements" value must be a complete shopping requirements strictly keep the structure of the original one.
    - This is not a patch. Therefore, you must carefully check how to reflect the updates to the original shopping requirements and output the complete one.
    - (Important!) Do NOT mis-override the existing part of the current requirements that should keep. It may break the requirements! Example:
        * In the current requirements, `"queries": ["I need no-fat ice-cream that come with plastic bags."]`
        * Event message: "Pack type is not a need but a wish."
        * Correct output: `["queries": "I need no-fat ice-cream. Package type plastic bags is preferred but not strictly required."]`
        * Incorrect output: `["queries": "Plastic bags pack type is preferred."]`
    - If event message claims a new requirement, it must be added to the corresponding field. If it conflicts with an existing requirement, only patch the conflicted part. Do NOT mechanically override the whole. Example:
        * In the current requirements, `"queries": ["I need yellow volleyball shoes with wave decoration"]`
        * Event message: "Color must be blue."
        * Correct output: `"queries": ["I need blue volleyball shoes with wave decoration"]` (patch "yellow" with "blue")
        * Incorrect output: `"queries": ["I need blue volleyball shoes"]` ("with wave decoration" MUST be kept!)
    - Output "affected_intents" correctly!
        * List all intent IDs whose requirements have been updated.
        * Do NOT include intent ID whose requirements have not been affected by this update.
        * Set empty list if the update has nothing to do to the original requirements, or is only modifying the "general" requirements.
    - Return JSON only. No plain text.
""".strip()

    UPDATE_REQUIREMENTS_USER = """
# Original Shopping Requirements
{requirements}

# Event Messages
{qa}
""".strip()

    EVENT_HINT_PARTIAL = """
Shopping requirements may update by events during the process. Events happen by text messages in the tool responses of effective tool calls.
    - An event message is a free text which enforces the update of the current shopping requirements or the database. Possible events include but are not limited to:
        * user brings a requirement (e.g. "I just want gothic style, nothing else.")
        * user modifies an existing requirement (e.g. "Changed my mind. Color must be blue.")
        * user withdraws a requirement (e.g. "Never mind pack type. Let's forget it.")
        * user moves a must-have requirement to optional requirement (e.g. "CPU speed is a wish than a need.")
        * user clarifies the exact need from the ambiguous requirement (e.g. "I'd prefer 3-pin over 2-pin.")
        * user claims or changes the budget. (e.g. "Oh, I forgot to let you know that my budget is 10 dollars.", "Sorry but I can't spend more than 120cts.")
        * user claims or changes the price range of a product. (e.g. "This TV must be less expensive than 1000USD.")
        * user recognizes that a carted product has gone or run out. (e.g. "The one you just picked disappeared from the shelf.")
        * user recognizes that a product's price has been changed. (e.g. "Sorry but the price of the one you selected has been updated to 2100.")
        * user claims or modifies the agent rules. (e.g. "Hey, say 'Hello' to me just before placing order, or I will reject your submission.")
        * and anything that needs update of shopping requirements or the stock database
    - (WARNING!) User NEVER changes "category" value of any intent in the requirements.
    - An event message can be:
        * a clarified answer to the question that the agent asked in the previous tool call (a dummy answer without meaningful information is not event)
        * a free text buried inside the tool response object (voluntary )
    - The followings are NOT event messages:
        * normal result data for the search and inspect tool calls like product details and statistics
        * error messages dedicated to the previous tool call
        * meta data with short technical messages like `"code": 0`, `"done": false`, `"result": "canceled"`
        * user asks the agent about something to clarify like "I can use coupon. Can you double check the picks with it?" (in such case, you must use message tool to answer user, instead of calling event tool! if no available message left, ignore it.)
        * a message looking like a stock-out event but that is happened just after removing cart operation
        * dummy answer to the previous question asked by a message tool, that does not update the shopping requirements like "Whatever you like. I trust you."
    - The same event message may appear repeatedly for reminder. Trigger only once. Do NOT call event tool multiple times for the same event message.
        * Ignore the stock-out event message happened just after removing cart operation, because it's the repeated mention of stock-out event for the removed product.
    - You MUST review previous tool responses carefully to identify all the events embedded in them.
        * One tool response may include multiple events.
        * User may use informal language or slang. Understand the nuance correctly.
        * Ignore the texts that are non-meaningful for requirements or database update.
        * Ignore the events that have beeen already triggered.
        * Do NOT check the latest tool response only! Unhandled event messages may be hidden in the older tool responses.
    - You MUST identify the product IDs for stock database update.
        * Event message may not explicitly mention the product ID whose stock status (stock amount, availibility, price, etc.) has been changed.
        * In that case, you must infer its correct ID from the context of the previous tool calls and responses.

Mapping ambiguous property name in event messages.
    - Sometimes you get confused with an event message because:
        * the event message mentions about a certain property name to clarify, modify, withdraw or move an existing requirement (e.g. "Now I'll forget occasion.")
        * but you can't find it in the existing "queries" for the intent (e.g. "nylon tshirts for winter")
    - If it's obvious what property name it is pointing to, continue to trigger event.
    - If you're confused, the Evidence Pairs list may save you.
        * Evidence Pairs are real key-value pairs from the product database.
        * You can find the property name mapping from this list. (e.g. `"occasion": "winter"`)
        * Then you can trigger event to update the requirement. (e.g. remove the phrase "for winter")
    - Evidence Pairs list is useful for clarifying, modifying, withdrawing or moving existing requirement only.
        * It's not your reference if event message is about adding a new requirement.
        * It's not your reference if event message is about stock status update or the answer to the question.

Pass a string array to the event tool's `value` argument. Each string item MUST be JSON-parsable for the following object:
{{
    "message": <str>, // a raw event message identified
    "question": <str|null> // the question asked by the agent, that was followed by the message (set null if this message is not for agent's question, but voluntary)
}}

Pass a JSON-parsable string to the event tool's `db_patch` argument if stock update is applicable. The string MUST be for the following object:
{{
    // list all affected product IDs
    <product_id_1, str>: {{
        // at least one of the following values must not be null

        "new_stock_amount": <int|null>, // new stock amount if applicable, otherwise null
        "new_availibility": <bool|null>, // new stock availibility if applicable, otherwise null
        "new_price": <int|null> // new price if applicable, otherwise null
    }},
    ... ...
}}

The response of event tool call will accept or reject your event tool call.
    - It rejects if you triggered event without meaningful new event message that can update the current requirements or database.
    - Check carefully to prevent rejection.

If accepted, it's responded with a new shopping requirements and the optional statistics. Then,
    - this new requirements is what you MUST adhere and base on from the next steps
    - the new requirements override all the old requirements including the initial one
""".strip()

    NOTICE_EXPLORE_PARTIAL = """
# Current Shopping Requirements
{requirements}
(
    - This is the most update shopping requirements.
    - It overrides all conflicts with old shopping requirements.
)

# Match Information per Intent
{match_info}
(
    - The list for each intent is sorted by matchness.
    - Matched products are with "match": true. Earlier occurence in the list is the better match.
    - Unmatched products are with "match": false, but you can consider them for substitute submission. Earlier occurence in the list falls short on the fewer requirements.
)

# Recommended Intent ID for Exploration
It's important to allocate tool call resources fairly and intelligently across intents, because you get scored 1 ONLY IF you found the best matching products for ALL intents.
Based on the deep analysis for the current Matching Information per Intent, the system strongly recommends you to search for intent ID `{hid}` if your next action will be exploration.

# Recommendation for Your Next Exploration
{explore_hint}

# Remaining Messages Available: {remaining_messages}

{call_hint}
""".strip()

    NOTICE_FINAL_PARTIAL = """
# Current Shopping Requirements
{requirements}
(
    - This is the most update shopping requirements.
    - It overrides all conflicts with old shopping requirements.
)

# Remaining Messages Available: {remaining_messages}

{call_hint}
""".strip()

    REQUIREMENTS_DYNAMIC_PARTIAL = """    
# Glossary for Requirements:
    - "general" is the requirements for the agent, and "intents" are the list of requirements per product intent.
    - "budget": upper limit of the sum of `product price * quantity` across all intents
    - "fallback": the rule that agent MUST follow when no product is found matching for an intent (set null if substitute submission is not allowed)
    - "instructions": a list of all the rules that agent MUST adhere during the process, except for those related to "budget", "max_messages", "max_steps" and "fallback"
    - "max_messages": maximum number of messages that the agent can send to the user
    - "max_steps": maximum number of effective tool calls that the assistance agent can use
        * It can be modified by an event like "Alert! Complete the process within 3 steps from now on.".
        * In that case, update this value as `Currently consumed steps + Steps left`.
    - "category": the product category requested (non-negotiable if set)
    - "queries": a list of short sentences including all the requirements as well as the informative contexts, except for those related to "price", "quantity" and "priority". Examples:
        * ["Comic style book written by Arthur Kailey.", "It must be in French.", "Story about dragons."]
        * ["Fans with 12v input voltage compatibility."]
        * ["Basketball shoes with whatever size is given on the listing sold by Levin's in silver.", "Fit for students."]
    - "priority": a short phrase for the criterion based on which the agent must pick the top one from the matching candidates. Always start with "Choose". Examples:
        * "Choose the cheapest."
        * "Choose the one with the shortest diameter."
    - "min_price": lower bound of product price (integer) if mentioned, otherwise null
    - "max_price": upper bound of product price (integer) if mentioned, otherwise null
    - "quantity": quantity that user wants to buy for this product (integer) if mentioned, otherwise 1
""".strip()

    REQUIREMENTS_STATIC_PARTIAL = """
# Glossary for Requirements:
    - "category": the product category requested (non-negotiable)
    - "queries": a list of short sentences including all the requirements as well as the informative contexts
    - "priority": a short phrase for the criterion based on which the agent must pick the top one from the matching candidates
    - "min_price": lower bound of product price if mentioned, otherwise null
    - "max_price": upper bound of product price if mentioned, otherwise null
    - "quantity": quantity that user wants to buy for this product
    - "fallback": the rule for the agent when no product is found matching for an intent
""".strip()

class CustomTools:
    FINISH_EXERCISE = {
        "type": "function",
        "function": {
            "name": "report_requested_information",
            "description": "Report the information requested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "A JSON-parsable string with keys `categories`, `search_tool_insight` and `inspect_tool_insight`."
                    }
                },
                "required": ["value"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    EVENT = {
        "type": "function",
        "function": {
            "name": "trigger_event",
            "description": "Trigger event to update the shopping requirements or the stock database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "An array of JSON-parsable string. Each string is for a JSON object containing an identified raw event messages and its question if exists."
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["requirements", "database", "both"],
                        "description": "`requirements` if these events only updates the shopping requirements. `database` if they only updates the stock database. `both` if they should update both."
                    },
                    "db_patch": {
                        "type": ["string", "null"],
                        "description": "A JSON-parsable string representing the stock database update needed. Null if stock database update is not applicable for these events."
                    }
                },
                "required": ["value", "kind", "db_patch"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    FINISH_VIEW_EXERCISE = {
        "type": "function",
        "function": {
            "name": "finish_process",
            "description": "Terminate the process with the identified best view tool name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "best_view_tool": {
                        "type": "string",
                        "description": "Name of the tool for inspecting the most comprehensive property map details."
                    }
                },
                "required": ["best_view_tool"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    INFORM_ERRAND = {
        "type": "function",
        "function": {
            "name": "report_action",
            "description": "Report the current action you have just triggered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "review", "submit", "decline", "finish"],
                        "description": "Action type."
                    },
                    "products": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "An array of product IDs being added (for add action) or removed (for remove action). Pass empty array for other actions."
                    }
                },
                "required": ["action", "products"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    DUMMY_FINISH = {
        "type": "function",
        "function": {
            "name": "finish_process",
            "description": "Finish the process.",
            "parameters": {
                "additionalProperties": False,
                "properties": {},
                "required": [],
                "type": "object"
            },
            "strict": True
        }
    }

    PURPOSE_PARTIAL = {
        "intent_id": {
            "type": "integer",
            "description": "Zero-based index of the intent which this search call is for."
        },
        "is_reference": {
            "type": "boolean",
            "description": "True if it's targeting a reference product. False if it's targeting the base product itself."
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
    def shorten_listing(v: Dict | List | str, stop_keys: List[str]) -> Dict:
        def __clean__(value: Any):
            if isinstance(value, dict):
                return {k: __clean__(val) for k, val in value.items() if k not in stop_keys}
            elif isinstance(value, list):
                return [__clean__(item) for item in value]
            elif isinstance(value, str):
                return value if len(value) < 50 else ""
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
    def extract_listings(js: List | Dict, id_key: str) -> List[Dict]:
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
    def eliminate_listings(js: List | Dict, id_key: str) -> List | Dict:
        if isinstance(js, dict):
            if id_key in js: return {}

            return {
                k: Misc.eliminate_listings(v, id_key) if isinstance(v, (dict, list)) else v for k, v in js.items()
            }
        elif isinstance(js, list):
            return [Misc.eliminate_listings(item, id_key) if isinstance(item, (dict, list)) else item for item in js]

        return js

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

        #print("### LLM RES:", content, tool_calls)
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

        #print("### ENV OBS:", tool_call["function"]["name"], obs)
        return result, obs, finished

    def __get_args__(tool_call: Dict[str, Dict]) -> Dict:
        raw = tool_call["function"].get("arguments", "{}")
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        
        if not isinstance(parsed, dict): raise ValueError("tool arguments must be a JSON object")
        return parsed

class Engine(object):
    MAX_DURATION = 300.0

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
        self.planned = False
        
        self.message_used = 0
        self.step = 1

        self.products: Dict[str, Dict] = {}
        self.dlg = []

        self.history: List[Tuple[Dict, int, bool, List, List]] = []        
        self.questions = {}

    def run(self) -> List[Dict[str, Any]]:
        self.__proc_parse__()

        self.__proc_exercise__()
        self.__post_exercise__()

        self.__proc_errand__()
        return self.dlg

    def __proc_parse__(self) -> None:
        content, _ = Proxy.llm(
            [
                {"role": "system", "content": Prompt.PARSE_SYS},
                {"role": "user", "content": Prompt.PARSE_USER.format(query = self.query, tools = Misc.dump_json(self.tools))}
            ],
            None, True, LLM.GPT6
        )
        js: Dict[str, Dict | List[Dict[str, Dict]]] = Misc.parse_json(content)

        general: Dict[str, Dict] = js.get("general") or {}
        intents = js.get("intents") or []

        self.general: Dict = {
            "max_messages": general.get("max_messages") or 0,
            "max_steps": general.get("max_steps") or self.max_steps,
            "budget": general.get("budget", {}).get("amount") or None,
            "fallback": general.get("fallback") or None,
            "instructions": general.get("instructions") or None,
            "substitute_allowed": general.get("substitute_allowed") or False
        }
        self.intents: List[Dict[str, Dict]] = []
        self.tool_map = defaultdict(lambda: [])

        for k, v in js.get("tools").items():
            if v and isinstance(v, str): v = [v]
            if not isinstance(v, list) or not v: continue

            self.tool_map[k] = v

        self.reference_reveals = [False for _ in intents]
        self.reference_shops = [None for _ in intents]
        self.top_7 = [[] for _ in intents]
        self.digs = [0 for _ in intents]
        self.templates = [{} for _ in intents]        

        budget = self.general["budget"]

        for i in intents:
            max_price = i.get("price").get("max") or None
            quantity = i.get("quantity") or 1

            if budget is not None:
                derived_max = budget // quantity
                max_price = min(derived_max, max_price if max_price is not None else 1e10)

            self.intents.append({
                "category": None,                
                "priority": i.get("priority") or None,
                "min_price": i.get("price").get("min") or None,
                "max_price": max_price,
                "quantity": quantity,
                "queries": i.get("queries"),
            })

        self.general_init = copy.deepcopy(self.general)
        self.intents_init = copy.deepcopy(self.intents)

    def __proc_exercise__(self) -> None:
        tools = (
            self.__get_tools__(["cart", "message", "finish", "other", "inspect"]) +
            self.__get_tools__(["search"], CustomTools.PURPOSE_PARTIAL) +
            [CustomTools.FINISH_EXERCISE, CustomTools.EVENT]
        )
        messages = [
            {
                "role": "system",
                "content": Prompt.EXERCISE_SYS.format(
                    category_hint = Prompt.CATEGORY_HINT_PARTIAL,
                    search_hint = Prompt.SEARCH_HINT_PARTIAL
                )
            },
            {"role": "user", "content": Prompt.EXERCISE_USER.format(requirements = self.__get_requirements_partial__("dynamic"))}
        ]
        self.exercise_dump = []
        stage_finished = False

        while not self.finished and not stage_finished:
            content, tool_calls = Proxy.llm(messages, tools, True, LLM.GPT6)

            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                for tc in tool_calls:
                    if self.finished: break

                    args = json.loads(tc["function"]["arguments"]) or {}
                    func = tc["function"]["name"]

                    if func == CustomTools.FINISH_EXERCISE["function"]["name"]:
                        self.__update_with_exercise_report__(Misc.parse_json(args["value"]))
                        stage_finished = True
                    elif func == CustomTools.EVENT["function"]["name"]:
                        event_messages, notice = self.__handle_event__(
                            [Misc.parse_json(pair) for pair in args["value"]], args["kind"], Misc.parse_json(args["db_patch"] or "{}")
                        )
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(notice)})
                        self.exercise_dump.append((tc, None, False, {"messages": event_messages, "notice": notice}))
                    elif func in (self.tool_map["cart"] + self.tool_map["finish"]):
                        messages.append({
                            "role": "tool", "tool_call_id": tc["id"],
                            "content": Misc.dump_json({"response": f"`{func}` tool is not allowed in initial steps."})
                        })
                    else:
                        if func in self.tool_map["search"]:
                            try:
                                intent_id, is_reference = args["intent_id"], args["is_reference"]
                            except KeyError:
                                messages.append({
                                    "role": "tool", "tool_call_id": tc["id"],
                                    "content": Misc.dump_json({"result": "Arguments `intent_id` and `is_reference` must be specified."})
                                })
                                continue

                            del args["intent_id"]
                            del args["is_reference"]

                            tc["function"]["arguments"] = Misc.dump_json(Misc.lower_vals(args))
                            self.digs[intent_id] += 1
                        else:
                            if func in self.tool_map["message"]: self.message_used += 1
                            intent_id, is_reference = None, False

                        raw, obs, self.finished = Proxy.env(self.binding, self.step, tc)
                        self.step += 1

                        self.dlg.append({"role": "assistant", "content": content, "tool_call": tc, "environment_result": raw})
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(obs)})

                        self.exercise_dump.append((tc, intent_id, is_reference, obs))
            else:
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Only respond with tool call(s)."},
                ]

    def __post_exercise__(self) -> None:
        for tc, intent_id, is_reference, obs in self.exercise_dump:
            func = tc["function"]["name"]

            if func == CustomTools.EVENT["function"]["name"]:
                for ltc, _, _, _, lobs in self.history[::-1]:
                    if ltc["function"]["name"] != CustomTools.EVENT["function"]["name"]:                        
                        lobs.append(obs["messages"])
                        break

                self.history.append((tc, None, False, [], [obs["notice"]]))
            elif func in self.tool_map["search"]:
                ids = self.__read_listings__(obs, self.search_alias, False, intent_id if is_reference else None)
                self.history.append((tc, intent_id, is_reference, ids, [Misc.eliminate_listings(obs, self.search_alias["product_id"])]))
            elif func == self.inspect_tool:
                self.__read_listings__(obs, self.inspect_alias, True, None)
            elif func not in self.tool_map["inspect"]:
                self.history.append((tc, None, False, [], [obs]))

        ids = [i for intent_id in range(len(self.intents)) for i in self.__get_product_ids__(intent_id, False, None)]
        obs_batch = self.__batch_inspect__(ids)

        for ltc, _, _, _, lobs in self.history[::-1]:
            if ltc["function"]["name"] != CustomTools.EVENT["function"]["name"]:                        
                lobs.extend(obs_batch)
                break

        self.__batch_evaluate__(list(range(len(self.intents))))

    def __proc_errand__(self) -> None:
        tools = (
            self.__get_tools__(["cart", "message", "finish", "other"]) + self.__get_tools__(["search"], CustomTools.PURPOSE_PARTIAL) + [CustomTools.EVENT]
        )
        self.ever_stock_event = False
        self.stage = "explore"
        self.bundle = None

        while not self.finished:
            if not self.planned:
                self.__plan_logistics__()
                if self.planned: self.__batch_ask__()

            self.__update_stage__()

            messages = self.__get_errand_messages__()
            content, tool_calls = Proxy.llm(messages, tools, True, LLM.GPT6)

            if tool_calls:
                for tc in tool_calls:
                    if self.finished: break

                    args = json.loads(tc["function"]["arguments"]) or {}
                    func = tc["function"]["name"]

                    if func in self.tool_map["search"]:
                        try:
                            intent_id, is_reference = args["intent_id"], args["is_reference"]
                        except KeyError:
                            self.history.append((tc, None, False, [], [{"result": "Arguments `intent_id` and `is_reference` must be specified."}]))
                            continue

                        del args["intent_id"]
                        del args["is_reference"]

                        tc["function"]["arguments"] = Misc.dump_json(Misc.lower_vals(args))
                    else:
                        intent_id, is_reference = None, False

                    if not self.__validitate_func__(func):
                        self.history.append((tc, intent_id, is_reference, [], [Misc.dump_json({"result": "Rejected. This tool is not allowed in this stage."})]))
                        continue

                    if func == CustomTools.EVENT["function"]["name"]:
                        _, notice = self.__handle_event__(
                            [Misc.parse_json(pair) for pair in args["value"]], args["kind"], Misc.parse_json(args.get("db_patch") or "{}")
                        )
                        self.history.append((tc, None, False, [], [notice]))
                    else:
                        raw, obs, self.finished = Proxy.env(self.binding, self.step, tc)
                        self.step += 1

                        if func in self.tool_map["message"]: self.message_used += 1
                        self.dlg.append({"role": "assistant", "content": content, "tool_call": tc, "environment_result": raw})

                        if func in self.tool_map["search"]:
                            ids = self.__read_listings__(obs, self.search_alias, False, intent_id if is_reference else None)
                            ids_to_view = self.__get_product_ids__(intent_id, False, is_reference)

                            obs_batch = self.__batch_inspect__(ids_to_view)
                            if ids_to_view: self.__batch_evaluate__([intent_id])

                            self.history.append(
                                (
                                    tc, intent_id, is_reference, ids,
                                    [Misc.eliminate_listings(obs, self.search_alias["product_id"])] + obs_batch
                                )
                            )
                            self.digs[intent_id] += 1
                        else:
                            self.history.append((tc, intent_id, is_reference, [], [obs]))
            else:
                self.history.append((None, None, False, [], [content]))
    
    def __plan_logistics__(self) -> None:
        if len(self.tool_map["message"]) == 0: return
        if self.general["max_messages"] == 0: return
        if self.__get_remaining_steps__() < 10 + len(self.intents): return

        samples = {
            i: self.__get_product_details__(self.__get_product_ids__(i, True, False)) for i in range(len(self.intents))
        }
        if any(len(v) < 3 for v in samples.values()): return

        content, _ = Proxy.llm(
            [
                {"role": "system", "content": Prompt.LOGISTICS_SYS.format(count = math.ceil(self.general["max_messages"] / len(self.intents)))},
                {
                    "role": "user",
                    "content": Prompt.LOGISTICS_USER.format(
                        requirements = self.__get_requirements_partial__("eval"),
                        samples = samples
                    )
                }
            ],
            None, True, LLM.GPT6
        )
        js: Dict = Misc.parse_json(content) or {}
        
        self.questions = {int(i): v for i, v in js.items()}
        self.planned = True

    def __get_errand_messages__(self) -> List:
        if self.stage == "explore":
            sys_prompt = Prompt.EXPLORE_SYS.format(event_hint = Prompt.EVENT_HINT_PARTIAL, search_hint = Prompt.SEARCH_HINT_PARTIAL)
        elif self.stage == "checkout":
            sys_prompt = Prompt.CHECKOUT_SYS.format(event_hint = Prompt.EVENT_HINT_PARTIAL, candidates = self.bundle)
        else:
            sys_prompt = Prompt.DECLINE_SYS.format(event_hint = Prompt.EVENT_HINT_PARTIAL)

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": Prompt.ERRAND_USER.format(requirements = self.__get_requirements_partial__("dynamic", True))
            }
        ]
        hid = self.__highlight_intent__()
        notice = self.__get_notice__(hid)

        valid_ids = self.__get_product_ids__(hid, None, None) if self.stage == "explore" else []
        shown_ids, search_hit = [], 0

        for i, (tc, intent_id, is_reference, ids, obs) in enumerate(self.history):
            if tc is not None:
                tc = copy.deepcopy(tc)

                if tc["function"]["name"] in self.tool_map["search"]:
                    args = Misc.parse_json(tc["function"]["arguments"])
                    
                    args.update({"intent_id": intent_id, "is_reference": is_reference})
                    tc["function"]["arguments"] = Misc.dump_json(args)

                    search_hit += 1
                    listings = {}

                    for pid in ids:
                        if pid in valid_ids:
                            if pid not in shown_ids:
                                details = {
                                    k: v for k, v in self.products[pid].items()
                                    if k not in ["is_inspected", "reference_intents", "title", "is_available", "stock_amount"]
                                }
                                shown_ids.append(pid)
                            else:
                                details = "[MASKED] (Already shown above.)"
                        elif search_hit < 6:
                            details = {
                                "price": self.products[pid]["price"], "extra": "[MASKED]"
                            }
                        else:
                            details = "[MASKED]"

                        listings[pid] = details

                    obs = [listings] + obs
                
                if i == len(self.history) - 1: obs += [{"recent_notice": notice}]

                messages += [
                    {"role": "assistant", "content": "", "tool_calls": [tc]},
                    {"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(obs)}
                ]
            else:
                messages += [
                    {"role": "assistant", "content": obs[0]}, {"role": "user", "content": "Only respond with tool call(s)."},
                ]

        # print("????????????????????????")
        # print(self.stage, self.bundle, notice)
        # print("????????????????????????")

        return messages

    def __highlight_intent__(self) -> int:
        vals = []

        for i, it in enumerate(self.intents):
            match_count = len([t for t in self.top_7[i] if t["match"]])
            digs = self.digs[i] - (0 if it["priority"] is None else 2)

            vals.append(
                (
                    i,
                    (match_count > 0, match_count > 1, self.reference_reveals[i], digs, match_count, -i)
                )
            )

        vals.sort(key = lambda v: v[1])
        return vals[0][0]

    def __update_stage__(self) -> None:
        bundle, is_substitute = self.__get_best_bundle__()
        if is_substitute and not self.general["substitute_allowed"]: bundle = None

        if self.stage == "explore":
            if (self.__get_remaining_steps__() <= len(self.intents) + 3) or (self.__get_remaining_steps__() <= len(self.intents) + 8 and not self.ever_stock_event):
                self.stage = "checkout" if bundle else "decline"
        elif self.stage == "checkout":
            if bundle is None:
                self.stage = "explore" if self.__get_remaining_steps__() >= len(self.intents) + 5 else "decline"
        elif self.stage == "decline":
            if bundle:
                self.stage = "checkout"
            elif self.__get_remaining_steps__() >= len(self.intents) + 5:
                self.stage = "explore"

        if self.__is_urgent__(): self.stage = "checkout" if bundle else "decline"
        self.bundle = bundle

    def __get_notice__(self, hid: int) -> str:
        requirements = self.__get_requirements_partial__("notice")

        match_info = {
            i: [
                {
                    "product_id": t["id"],
                    "price": self.products[t["id"]]["price"],
                    "match": t["match"]
                } for t in self.top_7[i]
            ]
            for i in range(len(self.intents))
        }
        remaining_messages = self.general["max_messages"] - self.message_used

        if self.__get_remaining_steps__() == 1:
            call_hint = "(CRITICAL!) You have only one tool call step left. Do NOTHING but use the {action} tool immediately! (Skip cart review)".format(
                action = "decline" if self.stage != "checkout" else "submit"
            )
        else:
            call_hint = ""

        explore_hint = ""

        if any(t["match"] for t in self.top_7[hid]):
            if self.intents[hid]["priority"]:
                explore_hint = f"""
As you can see, the matching product has been found for intent ID `{hid}`.
However, there exists a "priority" criterion - "{self.intents[hid]['priority']}"
Therefore, you must do your best to find the better candidate.
For example, if "priority" requires to choose the cheapest, you can try more search tools with intentionally lowered price cap. (if price cap argument is supported)
""".strip()
            else:
                explore_hint = f"""
As you can see, the matching product has been found for intent ID `{hid}`.
Explore more products. There may exist candidates which meet optional requirements as well or the prices are cheaper than the current one.
""".strip()
        else:
            if self.reference_reveals[hid]:
                explore_hint = "Do NOT search for reference product, because it's unspecified or already revealed. Focus on base product."
            else:
                explore_hint = "Reference product has not been revealed. Search it first."

        if self.stage == "explore":
            return Prompt.NOTICE_EXPLORE_PARTIAL.format(
                requirements = requirements, match_info = match_info, remaining_messages = remaining_messages,
                call_hint = call_hint, hid = hid, explore_hint = explore_hint
            )
        else:
            return Prompt.NOTICE_FINAL_PARTIAL.format(
                requirements = requirements, remaining_messages = remaining_messages, call_hint = call_hint
            )

    def __read_listings__(self, obs: List | Dict, alias: Dict, is_inspect: bool, reference_intent_id: Optional[int]) -> List[str]:
        records = Misc.extract_listings(obs, alias["product_id"])
        ids = []

        for r in records:
            pid = r[alias["product_id"]]
            ids.append(pid)
            
            if pid in self.products and (self.products[pid]["is_inspected"] or not is_inspect): continue
            listing = {"is_inspected": is_inspect, "is_available": None, "stock_amount": None}
            
            for k, v in alias.items():
                if k in ["free_text", "product_id"]: continue
                listing[k] = r[v] if v else None

            for k, v in r.items():
                if any(k == a for a in alias.values()): continue
                if k in alias["free_text"]: continue

                listing[k] = Misc.shorten_listing(v, alias["free_text"]) if isinstance(v, (dict, list, str)) else v

            if pid not in self.products:
                if listing["is_available"] is None: listing["is_available"] = True
                listing["reference_intents"] = [] if reference_intent_id is None else [reference_intent_id]

                self.products[pid] = listing
            else:
                del listing["is_available"]
                del listing["stock_amount"]

                listing["reference_intents"] = list(set(
                    self.products[pid]["reference_intents"] + ([] if reference_intent_id is None else [reference_intent_id]))
                )
                self.products[pid].update(listing)
        
        return ids

    def __batch_inspect__(self, ids: List) -> List:
        for tool in self.tools:
            if tool["function"]["name"] == self.inspect_tool:
                args: Dict = tool["function"]["parameters"]["properties"]
                arg_key = list(args.keys())[0]
                is_list = args[arg_key]["type"] == "array"
                break

        obs_batch, i = [], 0

        while True:
            if self.__get_remaining_steps__() <= 3 + len(self.intents): break
            if self.__get_remaining_steps__() <= 6 + len(self.intents) and not self.ever_stock_event: break

            to_view = ids[i * self.inspect_capacity:(i + 1) * self.inspect_capacity]
            if not to_view: break

            arg_val = to_view if is_list else ','.join(to_view)

            tc = {
                "type": "function",
                "index": i,
                "id": Misc.random_id(),
                "function": {
                    "name": self.inspect_tool,
                    "arguments": Misc.dump_json({arg_key: arg_val})
                }
            }
            raw, obs, self.finished = Proxy.env(self.binding, self.step, tc)
            self.step += 1

            self.dlg.append({"role": "assistant", "content": "", "tool_call": tc, "environment_result": raw})
            self.__read_listings__(obs, self.inspect_alias, True, None)

            obs_batch.append(Misc.eliminate_listings(obs, self.inspect_alias["product_id"]))
            i += 1

        return obs_batch

    def __batch_evaluate__(self, intent_ids: List) -> None:
        candidates = {
            i: self.__get_product_details__(self.__get_product_ids__(i, True, False)) for i in intent_ids
        }
        references = {
            i: self.__get_product_details__(self.__get_product_ids__(i, True, True)) for i in intent_ids
        }
        if sum(len(v) for v in candidates.values()) + sum(len(v) for v in references.values()) == 0: return

        messages = [
            {"role": "system", "content": Prompt.EVALUATE_SYS.format(ids = [str(i) for i in intent_ids])},
            {
                "role": "user",
                "content": Prompt.EVALUATE_USER.format(
                    requirements = self.__get_requirements_partial__("eval"),
                    candidates = candidates,
                    references = references
                    )
            }
        ]
        content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
        js = Misc.parse_json(content)

        for i, result in js.items():
            self.reference_reveals[int(i)] = result["reference_revealed"]
            self.templates[int(i)] = result["template"]
            self.top_7[int(i)] = result["top_7"]

    def __batch_ask__(self) -> None:
        tools = self.__get_tools__(["message"]) + [CustomTools.EVENT]
        stage_finished = False

        messages = [{"role": "user", "content": Prompt.ASK_USER.format(requirements = self.__get_requirements_partial__("dynamic"))}]
        questions_done, intents_done, cursor, last_intent = [], [], 0, None

        while not self.finished and not stage_finished:
            questions = copy.deepcopy(questions_done)
            
            for q in self.questions[list(self.questions.keys())[cursor]]:
                if q not in questions_done:                
                    questions.append(q)
                    break
            else:
                break

            content, tool_calls = Proxy.llm(
                [{"role": "system", "content": Prompt.ASK_SYS.format(event_hint = Prompt.EVENT_HINT_PARTIAL, questions = questions)}] + messages,
                tools, True, LLM.GPT6
            )
            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                for tc in tool_calls:
                    if self.finished or stage_finished: break

                    args = json.loads(tc["function"]["arguments"]) or {}
                    func = tc["function"]["name"]

                    if func == CustomTools.EVENT["function"]["name"]:
                        _, notice = self.__handle_event__(
                            [Misc.parse_json(pair) for pair in args["value"]], args["kind"], Misc.parse_json(args.get("db_patch") or "{}")
                        )
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(notice)})
                        self.history.append((tc, None, False, [], [notice]))

                        if last_intent is not None: intents_done.append(last_intent)
                    else:
                        if self.message_used >= self.general["max_messages"]:
                            stage_finished = True
                            break

                        raw, obs, self.finished = Proxy.env(self.binding, self.step, tc)
                        self.step += 1

                        self.message_used += 1

                        self.dlg.append({"role": "assistant", "content": content, "tool_call": tc, "environment_result": raw})
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": Misc.dump_json(obs)})

                        self.history.append((tc, None, False, [], [obs]))
                        new_cursor = None
                        
                        last_intent = list(self.questions.keys())[cursor]
                        questions_done.append(questions[-1])

                        for _ in range(len(self.intents)):
                            cursor = (cursor + 1) % len(self.intents)                            
                            i = list(self.questions.keys())[cursor]
                            
                            if i in intents_done: continue
                            if all(q in questions_done for q in self.questions[i]): continue

                            new_cursor = cursor
                            break

                        if new_cursor is None: stage_finished = True
                        if self.__get_remaining_steps__() < 10 + len(self.intents): stage_finished = True
            else:
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Only respond with tool call(s)."},
                ]

    def __on_stock_event__(self) -> None:
        for i, req in enumerate(self.intents):
            top_7 = self.top_7[i]
            new_top_7 = []

            for t in top_7:
                pid = t["id"]
                
                details = self.products[pid]
                if not details["is_available"]: continue

                if details["price"] is not None:
                    if req["min_price"] is not None and details["price"] < req["min_price"]: continue
                    if req["max_price"] is not None and details["price"] > req["max_price"]: continue

                if details["stock_amount"] is not None:
                    if req["quantity"] is not None and details["stock_amount"] < req["quantity"]: continue

                new_top_7.append(t)

            self.top_7[i] = new_top_7

    def __update_requirements__(self, qa: List):
        messages = [
            {"role": "system", "content": Prompt.UPDATE_REQUIREMENTS_SYS},
            {"role": "user", "content": Prompt.UPDATE_REQUIREMENTS_USER.format(requirements = self.__get_requirements_partial__("dynamic"), qa = qa)}
        ]
        content, _ = Proxy.llm(messages, None, True, LLM.GPT6)
        js: Dict[str, Dict[str, Dict]] = Misc.parse_json(content)

        self.general = js.get("new_requirements").get("general")

        for i in range(len(self.intents)):
            obj = js.get("new_requirements").get("intents")

            intent = obj.get(str(i)) if isinstance(obj, dict) else obj[i]            
            if not intent: continue

            self.intents[i] = intent

        affected = [int(x) for x in (js.get("affected_intents") or [])]
        self.__batch_evaluate__(affected)

    def __get_product_ids__(self, intent_id: Optional[int], is_inspected: Optional[bool], is_reference: Optional[bool]) -> List:
        ids = []
        
        for pid, details in self.products.items():
            if is_inspected is not None and details["is_inspected"] != is_inspected: continue

            if intent_id is not None:
                req = self.intents[intent_id]
                is_valid = True

                if details["category"] and req["category"] and details["category"].lower() != req["category"].lower(): is_valid = False
                if not details["is_available"]: is_valid = False
                
                if details["price"] is not None:
                    if req["min_price"] is not None and details["price"] < req["min_price"]: is_valid = False
                    if req["max_price"] is not None and details["price"] > req["max_price"]: is_valid = False

                if details["stock_amount"] is not None:
                    if req["quantity"] is not None and details["stock_amount"] < req["quantity"]: is_valid = False

                if is_reference is not None and not is_reference and not is_valid: continue
                has_ref = intent_id in details["reference_intents"]

                if self.reference_shops[intent_id] and details.get("shop"):
                    if self.reference_shops[intent_id].lower() != details["shop"].lower(): has_ref = False

                if is_reference is not None and is_reference and not has_ref: continue

                if is_reference is None and not is_valid and not has_ref: continue

            ids.append(pid)
        
        return ids

    def __get_product_details__(self, ids: List) -> Dict:
        return {
            i: {
                k: v
                for k, v in self.products[i].items()
                if k not in ["is_inspected", "reference_intents", "title", "is_available", "stock_amount"]
            }
            for i in ids
        }

    def __update_with_exercise_report__(self, report: Dict[str, Dict]) -> str:
        self.inspect_tool = report["inspect_tool_insight"]["best"]
        self.inspect_capacity = report["inspect_tool_insight"]["capacity"]
        
        self.inspect_alias = {
            "product_id": report["inspect_tool_insight"]["product_id"],
            "free_text": report["inspect_tool_insight"]["free_text"]
        }
        self.search_alias = report["search_tool_insight"]

        for i, intent in enumerate(self.intents):
            intent["category"] = report["categories"][str(i)]
            self.reference_shops[i] = report["reference_shops"].get(str(i))

    def __handle_event__(self, qa: List, kind: string, db_patch: Optional[Dict[str, Dict]]) -> Tuple[List, Dict]:
        notice = ""
        
        if db_patch:
            is_patched = False

            for pid, patch in db_patch.items():
                if pid not in self.products: continue
                
                details = self.products[pid]
                is_patched = True

                if patch.get("new_stock_amount") is not None: details["stock_amount"] = patch["new_stock_amount"]
                if patch.get("new_availibility") is not None: details["is_available"] = patch["new_availibility"]
                if patch.get("new_price") is not None: details["price"] = patch["new_price"]

            if is_patched:
                notice += f"\n\n# Stock event identified."
                self.ever_stock_event = True

                self.__on_stock_event__()

        if qa and kind in ["requirements", "both"]:
            self.__update_requirements__(qa)

            notice += f"""

# Requirements updated. New shopping requirements are:
{self.__get_requirements_partial__("notice")}
""".strip()

        messages = [pair["message"] for pair in qa]
        return messages, {"result": notice.strip()}

    def __get_tools__(self, types: List[str], injection: Dict = None) -> List:
        chosen = []

        for tool in self.tools:
            if any(tool["function"]["name"] in self.tool_map[t] for t in types):
                copied: Dict[str, Dict[str, Dict[str, Dict]]] = copy.deepcopy(tool)

                if injection:
                    copied["function"]["parameters"]["properties"].update(injection)
                    copied["function"]["parameters"]["required"] = copied["function"]["parameters"].get("required", []) + list(injection.keys())
                
                chosen.append(copied)

        return chosen

    def __get_requirements_partial__(self, mode: str, is_init: bool = False) -> Dict:
        if is_init:
            general, intents = self.general_init, self.intents_init
        else:
            general, intents = self.general, self.intents

        if mode != "eval":
            req = {"general": general}
        else:
            req = {"fallback": general["fallback"]}

        req["intents"] = {i: it for i, it in enumerate(intents)}

        if mode == "dynamic":
            evidence = "\n\n# Evidence Pairs\n\n" + Misc.dump_json(self.templates)
            return Misc.dump_json(req, indent = 2) + evidence + "\n\n" + Prompt.REQUIREMENTS_DYNAMIC_PARTIAL
        elif mode == "eval":
            return Misc.dump_json(req, indent = 2) + "\n\n" + Prompt.REQUIREMENTS_STATIC_PARTIAL
        else:
            return Misc.dump_json(req, indent = 2)

    def __get_remaining_steps__(self):
        return self.general["max_steps"] - self.step + 1

    def __get_best_bundle__(self) -> Tuple[Optional[List], bool]:
        all_indices = [list(indices) for indices in product(*(range(len(group)) for group in self.top_7))]                
        bundles_ex = []

        for indices in all_indices:
            b = [self.top_7[i][idx] for i, idx in enumerate(indices)]

            if self.general["budget"] is not None:
                price_sum = sum([self.products[t["id"]]["price"] * self.intents[i]["quantity"] for i, t in enumerate(b)])
                if price_sum > self.general["budget"]: continue

            matches = len([t for t in b if t["match"]])
            priority_sum = sum([idx for i, idx in enumerate(indices) if self.intents[i]["priority"]])

            bundles_ex.append((b, (-matches, priority_sum, sum(indices))))

        if not bundles_ex: return None, False
        bundles_ex.sort(key = lambda be: be[1])

        return [t["id"] for t in bundles_ex[0][0]], (-bundles_ex[0][1][0]) != len(self.intents)       

    def __validitate_func__(self, func: str) -> bool:
        if self.stage == "explore":
            if func in (self.tool_map["cart"] + self.tool_map["finish"]): return False
        elif self.stage == "checkout":
            if func in self.tool_map["search"]: return False
        elif self.stage == "decline":
            if func in (self.tool_map["cart"] + self.tool_map["search"]): return False

        return True

    def __is_urgent__(self) -> bool:
        elapsed = time.monotonic() - self.start_tick
        return elapsed > Engine.MAX_DURATION - 40.0

def agent_main(prob_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    eng = Engine(prob_data)
    return eng.run()
