import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, WebsiteSearchTool
import requests

st.set_page_config(page_title="Personalized Nutrition Advisor", page_icon="🥗", layout="wide")

load_dotenv()

# Make sure the API key is set before initializing the tool
serper_api_key = os.getenv("SERPER_API_KEY")
os.environ["serper_api_key"] = serper_api_key

# Initialize the search tools with the API key
search_tool = SerperDevTool(api_key=serper_api_key)
website_search_tool = WebsiteSearchTool()

os.environ["serper_api_key"] = os.getenv("SERPER_API_KEY")
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'

def check_ollama_server():
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            st.success("Ollama server is running")
            return True
        else:
            st.error("Ollama server is not running")
            return False
    except Exception as e:
        st.error(f"Error checking Ollama server: {e}")
        return False

def get_ollama_llm(llm_model="ollama/deepseek-r1:7b"):
    ollama_llm = LLM(
        model=llm_model,
        base_url='http://localhost:11434',
        api_key="",
    )
    return ollama_llm


def create_agents():
    """Create the specialized nutrition agents."""
    llm = get_ollama_llm()
    
    # Nutrition Researcher
    nutritionist = Agent(
        role='Nutrition Specialist',
        goal='Research and develop personalized nutritional recommendations based on scientific evidence',
        backstory='''You are a highly qualified nutritionist with expertise in therapeutic diets, 
                    nutrient interactions, and dietary requirements across different health conditions. 
                    Your recommendations are always backed by peer-reviewed research.''',
        tools=[search_tool],
        llm=llm,
        verbose=True
    )
    
    # Medical Nutrition Specialist
    medical_specialist = Agent(
        role='Medical Nutrition Therapist',
        goal='Analyze medical conditions and provide appropriate dietary modifications',
        backstory='''With dual training in medicine and nutrition, you specialize in managing 
                    nutrition-related aspects of various medical conditions. You understand 
                    medication-food interactions and how to optimize nutrition within medical constraints.''',
        tools=[search_tool],
        llm=llm,
        verbose=True
    )
    
    # Diet Plan Creator
    diet_planner = Agent(
        role='Therapeutic Diet Planner',
        goal='Create detailed, practical and enjoyable meal plans tailored to individual needs',
        backstory='''You excel at transforming clinical nutrition requirements into delicious, 
                    practical eating plans. You have extensive knowledge of food preparation, 
                    nutrient preservation, and food combinations that optimize both health and enjoyment.''',
        llm=llm,
        verbose=True
    )
    
    return nutritionist, medical_specialist, diet_planner

def create_tasks(nutritionist, medical_specialist, diet_planner, user_info):
    """Create tasks for each agent based on user information."""
    
    # First task: Research nutrition needs based on demographics
    demographics_research = Task(
        description=f'''Research nutritional needs for an individual with the following demographics:
            - Age: {user_info['age']}
            - Gender: {user_info['gender']}
            - Height: {user_info['height']}
            - Weight: {user_info['weight']}
            - Activity Level: {user_info['activity_level']}
            - Goals: {user_info['goals']}
            
            Provide detailed nutritional requirements including:
            1. Caloric needs (basal and adjusted for activity)
            2. Macronutrient distribution (proteins, carbs, fats)
            3. Key micronutrients particularly important for this demographic
            4. Hydration requirements
            5. Meal timing and frequency recommendations''',
        agent=nutritionist,
        expected_output="A comprehensive nutritional profile with scientific rationale"
    )
    
    # Second task: Analyze medical conditions and adjust nutritional recommendations
    medical_analysis = Task(
        description=f'''Analyze the following medical conditions and medications, then provide dietary modifications:
            - Medical Conditions: {user_info['medical_conditions']}
            - Medications: {user_info['medications']}
            - Allergies/Intolerances: {user_info['allergies']}
            
            Consider the baseline nutritional profile and provide:
            1. Specific nutrients to increase or limit based on each condition
            2. Food-medication interactions to avoid
            3. Potential nutrient deficiencies associated with these conditions/medications
            4. Foods that may help manage symptoms or improve outcomes
            5. Foods to strictly avoid''',
        agent=medical_specialist,
        context=[demographics_research],
        expected_output="A detailed analysis of medical nutrition therapy adjustments"
    )
    
    # Third task: Create the comprehensive diet plan
    diet_plan = Task(
        description=f'''Create a detailed, practical diet plan incorporating all information:
            - User's Food Preferences: {user_info['food_preferences']}
            - Cooking Skills/Time: {user_info['cooking_ability']}
            - Budget Constraints: {user_info['budget']}
            - Cultural/Religious Factors: {user_info['cultural_factors']}
            
            Develop a comprehensive nutrition plan that includes:
            1. Specific foods to eat daily, weekly, and occasionally with portion sizes
            2. A 7-day meal plan with specific meals and recipes
            3. Grocery shopping list with specific items
            4. Meal preparation tips and simple recipes
            5. Eating out guidelines and suggested restaurant options/orders
            6. Supplement recommendations if necessary (with scientific justification)
            7. Hydration schedule and recommended beverages
            8. How to monitor progress and potential adjustments over time''',
        agent=diet_planner,
        context=[demographics_research, medical_analysis],
        expected_output="A comprehensive, practical, and personalized nutrition plan"
    )
    
    return [demographics_research, medical_analysis, diet_plan]

def create_crew(agents, tasks):
    """Create the CrewAI crew with the specified agents and tasks."""
    return Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )

def get_food_image(food_name):
    """Search for a food image using SerperDev."""
    try:
        search_query = f"{food_name} food dish high quality photo"
        print(f"Searching for image: {search_query}")
        results = search_tool.search_images(search_query)
        if results and len(results) > 0:
            print(f"Found {len(results)} image results for {food_name}")
            # Try to get a good quality image
            for result in results[:3]:  # Check first 3 results
                if 'imageUrl' in result:
                    print(f"Using image URL: {result['imageUrl'][:50]}...")
                    return result['imageUrl']
        print(f"No suitable images found for {food_name}")
        return None
    except Exception as e:
        print(f"Error fetching image for {food_name}: {e}")
        return None

def format_with_images(nutrition_plan, show_images=True):
    """Format the nutrition plan with images for food items."""
    if not show_images:
        return nutrition_plan
        
    formatted_plan = []
    lines = str(nutrition_plan).split('\n')
    
    for line in lines:
        formatted_plan.append(line)
        
        # Only process lines that likely contain food items
        if (line.strip().startswith('-') or line.strip().startswith('*') or 
            any(line.strip().startswith(str(i)+'.') for i in range(10))):
            
            # Extract the food item name - look for specific foods, not general categories
            food_item = line.split(':')[0] if ':' in line else line
            food_item = food_item.replace('-', '').replace('*', '').strip()
            
            # Clean up the food item name
            for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.']:
                if food_item.startswith(prefix):
                    food_item = food_item[len(prefix):].strip()
            
            # Skip very short items or items that are likely not specific foods
            if len(food_item) > 5 and not any(x in food_item.lower() for x in ['daily', 'weekly', 'avoid', 'limit']):
                image_url = get_food_image(food_item)
                if image_url:
                    # Use HTML img tag with proper formatting
                    formatted_plan.append(f"<img src='{image_url}' width='300' style='margin:10px 0;'/>\n")
    
    return '\n'.join(formatted_plan)

def run_nutrition_advisor(user_info):
    """Run the nutrition advisor with the user information."""
    try:
        # Create agents
        nutritionist, medical_specialist, diet_planner = create_agents()
        
        # Create tasks
        tasks = create_tasks(nutritionist, medical_specialist, diet_planner, user_info)
        
        # Create crew
        crew = create_crew([nutritionist, medical_specialist, diet_planner], tasks)
        
        # Execute the crew
        with st.spinner('Our nutrition team is creating your personalized plan. This may take a few minutes...'):
            result = crew.kickoff()
            
            # Format the result with images if enabled
            if user_info.get("show_images", True):
                formatted_result = format_with_images(str(result), show_images=True)
            else:
                formatted_result = str(result)
        
        return formatted_result
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def app():
    """Main Streamlit application."""
    st.title("🥗 Personalized Nutrition Advisor")
    st.markdown("""
    Get a detailed nutrition plan based on your demographics, health conditions, and preferences.
    Our AI team of nutrition specialists will create a personalized recommendation just for you.
    """)
    
    # Check Ollama server status
    check_ollama_server()
    
    # Create tabs for organization
    tab1, tab2, tab3 = st.tabs(["Basic Information", "Health Details", "Preferences & Lifestyle"])
    
    with tab1:
        st.header("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary/Other"])
            height = st.text_input("Height (e.g., 5'10\" or 178 cm)", "5'10\"")
            
        with col2:
            weight = st.text_input("Weight (e.g., 160 lbs or 73 kg)", "160 lbs")
            activity_level = st.select_slider(
                "Activity Level",
                options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"]
            )
            goals = st.multiselect(
                "Nutrition Goals",
                ["Weight Loss", "Weight Gain", "Maintenance", "Muscle Building", "Better Energy", 
                 "Improved Athletic Performance", "Disease Management", "General Health"]
            )
    
    with tab2:
        st.header("Health Information")
        
        medical_conditions = st.text_area(
            "Medical Conditions (separate with commas)",
            placeholder="E.g., Diabetes Type 2, Hypertension, Hypothyroidism..."
        )
        
        medications = st.text_area(
            "Current Medications (separate with commas)",
            placeholder="E.g., Metformin, Lisinopril, Levothyroxine..."
        )
        
        allergies = st.text_area(
            "Food Allergies/Intolerances (separate with commas)",
            placeholder="E.g., Lactose, Gluten, Shellfish, Peanuts..."
        )
    
    with tab3:
        st.header("Preferences & Lifestyle")
        
        # Add this at the top of tab3
        show_images = st.checkbox("Show food images in the plan", value=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            meal_types = st.multiselect(
                "Meal Type Preferences",
                ["Vegetarian", "Vegan", "Pescatarian", "Meat-Based", "Plant-Forward", "Mediterranean"],
                help="Select one or more meal types that match your dietary preferences"
            )
            
            food_preferences = st.text_area(
                "Additional Food Preferences & Dislikes",
                placeholder="E.g., Prefer plant-based, dislike seafood..."
            )
            
            cooking_ability = st.select_slider(
                "Cooking Skills & Available Time",
                options=["Very Limited", "Basic/Quick Meals", "Average", "Advanced/Can Spend Time", "Professional Level"]
            )
        
        with col2:
            budget = st.select_slider(
                "Budget Considerations",
                options=["Very Limited", "Budget Conscious", "Moderate", "Flexible", "No Constraints"]
            )
            
            cultural_factors = st.text_area(
                "Cultural or Religious Dietary Factors",
                placeholder="E.g., Halal, Kosher, Mediterranean tradition..."
            )
    
    # Collect all user information
    user_info = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "activity_level": activity_level,
        "goals": ", ".join(goals) if goals else "General health improvement",
        "medical_conditions": medical_conditions or "None reported",
        "medications": medications or "None reported",
        "allergies": allergies or "None reported",
        "meal_types": ", ".join(meal_types) if meal_types else "No specific meal type preferences",
        "food_preferences": food_preferences or "No specific preferences",
        "cooking_ability": cooking_ability,
        "budget": budget,
        "cultural_factors": cultural_factors or "No specific factors",
        "show_images": show_images
    }
    
    # Check if API keys are present
    if not os.getenv("SERPER_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ API keys not detected. Please add your SERPER_API_KEY and OPENAI_API_KEY to your .env file.")
    
    # Create a submission button
    if st.button("Generate Nutrition Plan"):
        if not goals:
            st.error("Please select at least one nutrition goal.")
            return
        
        # Display user information summary
        with st.expander("Summary of Your Information"):
            st.json(user_info)
        
        # Run the nutrition advisor
        result = run_nutrition_advisor(user_info)
        
        if result:
            st.success("✅ Your personalized nutrition plan is ready!")
            st.markdown("## Your Personalized Nutrition Plan")
            st.markdown(result, unsafe_allow_html=True)  # Allow HTML for images
            
            # Add download capability - convert result to string first
            result_str = str(result)
            st.download_button(
                label="Download Nutrition Plan",
                data=result_str.encode('utf-8'),
                file_name="my_nutrition_plan.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    app()