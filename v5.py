import streamlit as st
import os
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import random
import time
import re

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# --- 2. Define Tools (UNCHANGED) ---

if "shopping_list" not in st.session_state:
    st.session_state.shopping_list = []

@tool
def finalize_cart() -> str:
    """
    Use this tool to finalize the user's entire shopping list and present it as a final order.
    Call this tool ONLY when the user indicates they are completely finished and want to "checkout," "proceed," "finalize the order," or "place the order" for the regular items in their cart.
    """
    print(f"\n--- 🛠️ DEBUG: Executing tool `finalize_cart` ---")
    if not st.session_state.shopping_list:
        return "Your shopping cart is empty. Please add some items before finalizing."

    order_summary = "\n".join(f"- {item}" for item in st.session_state.shopping_list)
    confirmation_num = random.randint(100000, 999999)
    result = f"Thank you! Your order has been placed. Here is your summary:\n\n{order_summary}\n\nYour confirmation number is #{confirmation_num}. The items will be gathered and ready for you."
    
    st.session_state.shopping_list = []
    
    print(f"--- ✅ DEBUG: Tool `finalize_cart` returning: '{result}' ---\n")
    return result


# *** FIX: Updated tool prompt to enforce pricing and remove location references ***
@tool
def get_party_item_suggestions(occasion: Optional[str] = None, num_people: Optional[int] = None) -> str:
    """
    Use this tool to get a list of general party snacks, drinks, and platters with quantities correctly scaled for the number of guests.
    It is CRITICAL to have BOTH the occasion and the number of guests before calling this tool successfully.
    """
    print(f"\n--- 🛠️ DEBUG: Executing tool `get_party_item_suggestions` for {num_people} people, occasion: {occasion}. ---")
    
    if not num_people or num_people <= 0:
        return "User action required: The number of guests is missing. Ask the user how many people they are expecting before proceeding."
    if not occasion:
        return "User action required: The occasion is missing. Ask the user what the occasion for the party is before proceeding."

    planner_prompt = ChatPromptTemplate.from_messages(
        [
(
    "system",
    """
You are an expert party planner limited to suggesting only items commonly found in supermarkets. Your task is to recommend general, easy-to-serve party snacks, platters, and drinks for a given number of guests and occasion.

You MUST:
1. Calculate **realistic quantities** of each item based on the number of guests.
2. Assign a **reliable, unit price in USD ($)** for each item.
3. Calculate the **Total Cost** for all guests for each item.
4. If a promotion exists (using `get_promotions_and_deals`), apply it to the pricing and mention it beside the item.

You MUST NOT:
- Mention "US supermarkets" or "prices may vary by location/store."
- Make up fictional items or prices.

Use this exact format for each item:
<Item> – <Unit Price>$ (<Qty per guest> per guest) – <Total Cost>$ (for <number of guests> guests)
(Optional Deal: **<deal text>**)
"""
),
(
    "human",
    """
Please generate a list of supermarket-available party food and drinks with scaled quantities and accurate pricing for the following event:

Occasion: {occasion}
Number of People: {people}
"""
)
        ]
    )
    planner_llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)
    planner_chain = planner_prompt | planner_llm
    
    result = planner_chain.invoke({ "occasion": occasion, "people": num_people }).content
    print(f"--- ✅ DEBUG: Tool `get_party_item_suggestions` returning plan: '{result}' ---\n")
    return result

# *** FIX: Updated tool prompt to enforce pricing and remove location references ***
@tool
def get_meal_plan_and_quantities(dish_names: List[str], num_people: int) -> str:
    """
    Use this specialized tool ONLY to generate a detailed ingredient list WITH quantities for a specific hot meal.
    Call this tool ONLY after the user has explicitly requested a hot meal AND you have the number of people.
    """
    print(f"\n--- 🛠️ DEBUG: Executing specialized tool `get_meal_plan_and_quantities` for {num_people} people. ---")
    planner_prompt = ChatPromptTemplate.from_messages(
        [
  (
    "system",
    """
You are an expert meal planner that only uses items available in supermarkets.

You MUST:
1. Calculate the **realistic quantity** of each item based on the number of guests.
2. Assign a **reliable unit price in USD ($)** based on typical supermarket prices.
3. Calculate the **total cost for all guests** for each item.
4. If a promotion exists (using `get_promotions_and_deals`), apply it to the pricing and mention it beside the item.
5. Do NOT provide recipes or cooking instructions.
6. Do NOT suggest any ingredients that are not typically found in supermarkets.

You MUST NOT:
- Mention "US supermarkets" or "prices may vary by location/store."

Use this exact format for each item:
<Item> – <Unit Price>$ (<Qty per guest> per guest) – <Total Cost>$ (for <number of guests> guests)
(Optional Deal: **<deal text>**)
"""
),
(
    "human",
    """
Please generate a supermarket ingredient list with scaled quantities and accurate pricing for the following meal:

Dishes: {dishes}
Number of People: {people}
"""
)
        ]
    )
    planner_llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
    planner_chain = planner_prompt | planner_llm
    result = planner_chain.invoke({ "dishes": ", ".join(dish_names), "people": num_people }).content
    print(f"--- ✅ DEBUG: Tool `get_meal_plan_and_quantities` returning plan: '{result}' ---\n")
    return result

@tool
def manage_shopping_list(action: str, items: List[str]) -> str:
    """
    Use this simple tool ONLY to add or remove item NAMES from the shopping list. Call with `action='add'` AFTER a plan has been proposed and CONFIRMED.
    """
    print(f"\n--- 🛠️ DEBUG: Executing tool `manage_shopping_list` with action='{action}' and items={items} ---")
    if "shopping_list" not in st.session_state: st.session_state.shopping_list = []
    
    cleaned_items = [re.split(r'[:(]', item)[0].strip() for item in items if item]

    if action.lower() == "add":
        if not cleaned_items: return "Could you please specify which items you'd like me to add?"
        st.session_state.shopping_list.extend(cleaned_items)
        st.session_state.shopping_list = sorted(list(set(st.session_state.shopping_list)))
        result = f"Great! I've added those items to your list. It now contains: {', '.join(st.session_state.shopping_list)}."
    elif action.lower() == "view":
        result = f"Here is your current list: {', '.join(st.session_state.shopping_list)}" if st.session_state.shopping_list else "Your shopping list is currently empty."
    else: result = f"Error: Invalid action '{action}'."
    
    print(f"--- ✅ DEBUG: Tool `manage_shopping_list` returning: '{result}' ---\n")
    return result

@tool
def place_order(item: str, details: str) -> str:
    """
    Places a special order for a single, specific, custom item like a decorated birthday cake.
    Use this ONLY when the user asks for a 'custom cake' or to 'place an order' for a special bakery item.
    """
    print(f"\n--- 🛠️ DEBUG: Executing tool `place_order` for item='{item}' with details='{details}' ---")
    confirmation_num = random.randint(10000, 99999)
    result = f"Your special order for a '{item}' has been placed! The details are: '{details}'. Your confirmation number is #{confirmation_num}. It will be ready for pickup tomorrow after 1 PM."
    print(f"--- ✅ DEBUG: Tool `place_order` returning: '{result}' ---\n")
    return result

@tool
def get_promotions_and_deals(category: str) -> str:
    """
    Fetches current promotions and deals for a given supermarket category.
    """
    print(f"\n--- 🛠️ DEBUG: Executing tool `get_promotions_and_deals` with category='{category}' ---")
    deals = {
    "produce": "Buy 1 get 1 free on organic strawberries and blueberries.",
    "bakery": "Freshly baked sourdough and multigrain loaves – 30% off this weekend only.",
    "meat": "Get a family-size pack of boneless chicken breasts for just $6.49 (3 lbs).",
    "pantry": "Mix & match: 3 for $5 on select pasta sauces (Rao’s, Barilla, Classico).",
    "snacks": "Buy 2 get 1 free on Doritos, Lays, and Ruffles party-size bags.",
    "dairy": "Half-gallon almond and oat milks – just $2.99 each (was $4.49).",
    "frozen": "Save $3 when you buy any 2 frozen pizzas (DiGiorno, Red Baron, Amy's).",
    "beverages": "12-pack Coca-Cola or Pepsi – $5.99 with loyalty card (limit 2).",
    "household": "Buy 2 get 1 free on paper towels and toilet paper (select brands).",
    "breakfast": "Cereal blowout: 2 for $4 on Kellogg’s and General Mills varieties (12–18 oz)."
}

    result = deals.get(category.lower(), "No specific deals for that category right now.")
    print(f"--- ✅ DEBUG: Tool `get_promotions_and_deals` returning: '{result}' ---\n")
    return result

# --- 3. Define Agent State and Graph (UNCHANGED) ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# *** FIX: Updated SYSTEM_PROMPT to enforce tool usage for specific steps ***
SYSTEM_PROMPT = """
### IDENTITY & CORE DIRECTIVE
You are Shelf-help, a helpful and efficient AI assistant for a supermarket. Your primary goal is to help users plan for parties by suggesting items and managing their shopping list. You MUST follow a strict, logical flow.

### MASTER CONTROL FLOW (ReAct Protocol)

**STATE: AWAITING_TASK**
1.  **Goal:** Determine user's task (party planning, meal planning, or general assistance).
2.  **Action:** Greet the user and ask about party planning.

**STATE: GATHERING_CONTEXT (Party Planning)**
1.  **Goal:** Get the necessary information for party planning.
2.  **Action:** You MUST ask for the **number of guests** and the **occasion**.
3.  **Transition:** Once you have BOTH `num_people` AND `occasion`, transition to **STATE: GENERATING_SUGGESTIONS**.

**STATE: GENERATING_SUGGESTIONS**
1.  **Goal:** Generate a priced and scaled list of general party items.
2.  **Action:** You MUST immediately call the `get_party_item_suggestions` tool. Provide the `num_people` and `occasion` you gathered. Do not talk to the user first. Your only output in this state is the tool call.
3.  **Transition:** After the tool returns its result, transition to **STATE: PRESENTING_SUGGESTIONS**.

**STATE: PRESENTING_SUGGESTIONS**
1.  **Goal:** Show the user the generated list.
2.  **Action:** Complete list of items from the tool's output. After the list, ask the user if they want to add the items to their shopping list.
3.  **Transition:** Await user confirmation. If they confirm, use the `manage_shopping_list` tool. Then, ask the "hot meal" off-ramp question.

### MEAL PLANNING PROTOCOL
1. **Goal:** Plan a hot meal.
2. **Action:** Ask for specific dish names. Once you have the dish name(s) and the `num_people`, you **MUST call the `get_meal_plan_and_quantities` tool.**
3. **Presenting:** Present the resulting ingredient list verbatim and ask for confirmation before using `manage_shopping_list`.

### FINALIZE CART PROTOCOL
1. **Trigger:** User says "checkout," "proceed," "finalize," or "place the order" for regular items.
2. **Action:** Acknowledge, then call the `finalize_cart` tool.

### SPECIAL ORDER PROTOCOL (HIGH PRIORITY)
1. **Trigger:** User asks for "custom cake," "order a cake," or similar.
2. **Action:** Immediately stop current flow. Ask for details (flavor, size, writing).
3. **Tool Call:** Call the `place_order` tool.

### CONVERSATIONAL RULES
- **You MUST NOT generate your own lists of items.** Your ONLY source for suggestions is your tools.
- **NEVER mention your "tools."** Speak naturally.
- **NEVER apologize for technical errors.** If confused, ask a clarifying question.
- **Proactive Deals:** If discussing items from a category with a promotion, you can use `get_promotions_and_deals` proactively.
- **Price Reliability:** Acknowledge that you are providing reliable estimates based on supermarket averages, but do not emphasize location or variability.

---

### TOOL SUMMARY FOR INTERNAL REASONING

- `get_party_item_suggestions`: Get supermarket items for party (based on guests and occasion)
- `get_meal_plan_and_quantities`: Get ingredient list for specified dishes and guest count
- `get_promotions_and_deals`: Check for available deals per item category
- `manage_shopping_list`: Add or remove items from the shopping list
- `finalize_cart`: Finalize the entire cart when the user is done
- `place_order`: Submit special/custom orders (e.g., cakes)

"""


tools = [finalize_cart, get_party_item_suggestions, get_meal_plan_and_quantities, manage_shopping_list, place_order, get_promotions_and_deals]
tool_node = ToolNode(tools)

llm = ChatGroq(model="llama3-70b-8192", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    print("\n" + "="*80 + "\n--- 🧠 DEBUG: Agent Node Running ---")
    response = llm_with_tools.invoke(state["messages"])
    print(f"LLM Response received: {response.pretty_repr()}")
    if response.tool_calls: print(f"--- 📞 DEBUG: LLM decided to call tools: {response.tool_calls} ---")
    else: print("--- 💬 DEBUG: LLM decided to respond directly to the user. ---")
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    if isinstance(state["messages"][-1], ToolMessage): return "agent"
    if state["messages"][-1].tool_calls: return "use_tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("use_tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"use_tools": "use_tools", END: END})
workflow.add_edge("use_tools", "agent")

app = workflow.compile()

# --- 4. Streamlit User Interface (Robust Loop) ---

st.set_page_config(page_title="🛒 Shelf-help", layout="wide")
st.title("Shelf-help: Supermarket AI assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        AIMessage(content="""
Hello! 👋 Here are this week's hot supermarket deals you won't want to miss:

- 🥭 **Produce:** Buy 1 get 1 free on organic strawberries and blueberries.
- 🥖 **Bakery:** Freshly baked sourdough and multigrain loaves – 30% off this weekend only.
- 🍗 **Meat:** Family-size pack of boneless chicken breasts for just $6.49 (3 lbs).
- 🥫 **Pantry:** Mix & match: 3 for $5 on select pasta sauces (Rao’s, Barilla, Classico).
- 🍟 **Snacks:** Buy 2 get 1 free on Doritos, Lays, and Ruffles party-size bags.
- 🥛 **Dairy:** Half-gallon almond and oat milks – just $2.99 each (was $4.49).
- 🍕 **Frozen:** Save $3 when you buy any 2 frozen pizzas (DiGiorno, Red Baron, Amy's).
- 🥤 **Beverages:** 12-pack Coca-Cola or Pepsi – $5.99 with loyalty card (limit 2).
- 🧻 **Household:** Buy 2 get 1 free on paper towels and toilet paper (select brands).
- 🥣 **Breakfast:** Cereal blowout: 2 for $4 on Kellogg’s and General Mills varieties (12–18 oz).

Need help planning a party 🎉 or building your shopping list 🛒? Just let me know!
"""
)
    ]

# Display all past messages
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage): continue
    if isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input("What would you like to plan?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = app.invoke(
                {"messages": st.session_state.messages}, 
                {"recursion_limit": 25}
            )
            
            final_messages = response['messages']
            message_to_display = final_messages[-1]

            if len(final_messages) > 2:
                second_to_last = final_messages[-2]
                last = final_messages[-1]
                
                # Check for the specific failure condition: a successful tool call followed by a short, unhelpful AI summary.
                if (isinstance(second_to_last, ToolMessage) and 
                    second_to_last.name in ['get_party_item_suggestions', 'get_meal_plan_and_quantities'] and 
                    isinstance(last, AIMessage) and
                    len(last.content.split()) < 30):
                    
                    full_list = second_to_last.content
                    follow_up_question = last.content
                    
                    # The prompt dictates a specific intro phrase
                    if second_to_last.name == 'get_party_item_suggestions':
                        intro_phrase = ""
                    else:
                        intro_phrase = ""

                    corrected_content = f"{intro_phrase}\n\n{full_list}\n\n{follow_up_question}"
                    message_to_display = AIMessage(content=corrected_content)

            if isinstance(message_to_display, AIMessage) and message_to_display.content:
                st.session_state.messages.append(message_to_display)
                st.write(message_to_display.content)