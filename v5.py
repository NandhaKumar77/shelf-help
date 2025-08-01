import streamlit as st
import os
from dotenv import load_dotenv
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
            ("system", "You are an expert party planner. Given a number of people and an occasion, suggest a list of common, easy-to-serve party snacks, platters, and drinks. You MUST calculate and include realistic quantities for each item to serve the specified number of guests."),
            ("human", "Please generate a list of general party food and drinks with scaled quantities for the following event:\n\nOccasion: {occasion}\nNumber of People: {people}")
        ]
    )
    planner_llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)
    planner_chain = planner_prompt | planner_llm
    
    result = planner_chain.invoke({ "occasion": occasion, "people": num_people }).content
    print(f"--- ✅ DEBUG: Tool `get_party_item_suggestions` returning plan: '{result}' ---\n")
    return result

@tool
def get_meal_plan_and_quantities(dish_names: List[str], num_people: int) -> str:
    """
    Use this specialized tool ONLY to generate a detailed ingredient list WITH quantities for a specific hot meal.
    Call this tool ONLY after the user has explicitly requested a hot meal AND you have the number of people.
    """
    print(f"\n--- 🛠️ DEBUG: Executing specialized tool `get_meal_plan_and_quantities` for {num_people} people. ---")
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert chef. Your task is to create a practical supermarket shopping list. Given a list of dishes and a number of people, calculate realistic quantities for each raw ingredient needed."),
            ("human", "Please generate an ingredient list with quantities for the following meal:\n\nDishes: {dishes}\nNumber of People: {people}")
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
        "produce": "Buy one, get one free on all berries!", "bakery": "25% off all artisanal breads.",
        "meat": "Family pack of chicken thighs for $5.99.", "pantry": "2 for $5 on all pasta sauces.",
        "snacks": "$1 off any two bags of potato chips."
    }
    result = deals.get(category.lower(), "No specific deals for that category right now.")
    print(f"--- ✅ DEBUG: Tool `get_promotions_and_deals` returning: '{result}' ---\n")
    return result

# --- 3. Define Agent State and Graph (UNCHANGED) ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

SYSTEM_PROMPT = """
### IDENTITY & CORE DIRECTIVE
Your name is Super-Plan. You are a helpful and efficient AI assistant for a SUPERMARKET. Your primary goal is to help users plan for parties by suggesting items and managing their shopping lists.

### MANDATORY PARTY PLANNING PROTOCOL
This is your primary workflow when a user wants to plan a party. Follow it strictly.
**Step 1: Initial Inquiry & Context Gathering.** Acknowledge the party and ask clarifying questions. You MUST get the **number of guests** and the **occasion**.
**Step 2: Use the General Planner Tool (Default Action).** Once you have the number of guests and occasion, your FIRST action is to call the `get_party_item_suggestions` tool.
**Step 3: Present General Suggestions.** After the tool returns a list of suggestions, your next response to the user **MUST** be the output from that tool's output. Display the full list of items from the tool's output, and then ask if they would like to add these items. Strictly do not reply only like 'list of party items', you must provide the list of items which you got from 'get_party_item_suggestions'
**Step 4: The Meal "Off-Ramp".** After handling general items, ask an open-ended question like "Were you also thinking of preparing a more substantial hot meal?". **DO NOT** proceed to meal planning unless the user explicitly confirms.

### CONDITIONAL MEAL PLANNING PROTOCOL (ONLY IF USER ASKS FOR A MEAL)
- If the user says yes to a hot meal, ask them what specific dish(es) they'd like to make.
- Once you have the dish name and the `num_people`, call the `get_meal_plan_and_quantities` tool.
- After the tool returns a list of suggestions, your next response to the user **MUST** be the output from that tool's output. Display the full list of items from the tool's output, and then ask if they would like to add these items. Strictly do not reply only like 'list of party items', you must provide the list of items which you got from 'get_meal_plan_and_quantities' and ask for confirmation before using `manage_shopping_list`.

### FINALIZE CART PROTOCOL (For Regular Items)
1. **Trigger:** This protocol is for finalizing the entire list of regular, off-the-shelf items. Use it when the user says **"proceed," "checkout," "finalize," "I'm done,"** or **"place the order"** in the context of their shopping list.
2. **Action:** Acknowledge their request to finalize.
3. **Tool Call:** You MUST call the `finalize_cart` tool. It takes no arguments.
4. **Confirm:** Present the confirmation message from the tool to the user.

### OVERRIDE PROTOCOL: SPECIAL ORDERS (HIGH PRIORITY)
1. **Trigger:** This protocol overrides all others if the user uses keywords like **"order a cake," "custom cake," "decorated cake."**
2. **Action:** Immediately stop the general planning flow. Ask for the necessary details (flavor, size/servings, and specific writing).
3. **Tool Call:** Once you have the details, you MUST call the `place_order` tool.
4. **Confirm:** Present the confirmation message from the tool to the user.

### CONVERSATIONAL RULES
- **NEVER mention your "tools."** Speak naturally.
- **Be Proactive with Deals:** If a user discusses items from a category that has a promotion, you can proactively use the `get_promotions_and_deals` tool and mention the deal.
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

st.set_page_config(page_title="🛒 AI Planning Agent", layout="wide")
st.title("AI Shopping and Event Planner")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        AIMessage(content="""Hello! I can certainly help with that. Here are this week's special offers:

- **Produce:** Buy one, get one free on all berries!
- **Bakery:** 25% off all artisanal breads.
- **Meat:** Family pack of chicken thighs for $5.99.
- **Pantry:** 2 for $5 on all pasta sauces.
- **Snacks:** $1 off any two bags of potato chips.

How can I help you? I can assist with planning a party or finding more product details.""")
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

            # *** THE ONLY CHANGE IS HERE: The interception logic is now generalized for BOTH planning tools. ***
            if len(final_messages) > 2:
                second_to_last = final_messages[-2]
                last = final_messages[-1]
                
                # Check if the last action was a call to EITHER of our planning tools,
                # followed by an unhelpful, short summary.
                if (isinstance(second_to_last, ToolMessage) and 
                    second_to_last.name in ['get_party_item_suggestions', 'get_meal_plan_and_quantities'] and 
                    isinstance(last, AIMessage) and
                    len(last.content.split()) < 30): # Heuristic to catch summaries
                    
                    # CONSTRUCT THE CORRECT RESPONSE by combining the tool's detailed output
                    # with the agent's (usually correct) follow-up question.
                    full_list_from_tool = second_to_last.content
                    follow_up_question = last.content
                    
                    corrected_content = f"{full_list_from_tool}\n\n{follow_up_question}"
                    message_to_display = AIMessage(content=corrected_content)

            # Display and save the (potentially corrected) final message
            if isinstance(message_to_display, AIMessage) and message_to_display.content:
                st.session_state.messages.append(message_to_display)
                st.write(message_to_display.content)