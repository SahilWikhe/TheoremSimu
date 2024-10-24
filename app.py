from flask import Flask, render_template, redirect, url_for, request, jsonify
import json
import os
import google.generativeai as genai
import logging
import threading
import time
import re
from typing import Dict, List
import random
import openai
from dotenv import load_dotenv
from flask_cors import CORS
import base64
from pathlib import Path


load_dotenv()


# Add this with your other global variables
scenario_personas = {
    "investment": {
        "name": "Investment Advisor Client",
        "profile": """You are a client seeking investment advice. Your details are:
- Age: 45
- Income: $120,000/year
- Current Portfolio: $250,000 (mostly in low-risk mutual funds)
- Risk Tolerance: Moderate
- Goals: Growing wealth for retirement while managing risk
- Concerns: Market volatility, ensuring proper diversification
- Knowledge Level: Basic understanding of investments

You're interested in diversifying your portfolio but cautious about making major changes. You ask thoughtful questions about risk management and want to understand any recommended strategies thoroughly."""
    },
    "retirement": {
        "name": "Early Retirement Planning Client",
        "profile": """You are a client planning for early retirement. Your details are:
- Age: 35
- Income: $150,000/year
- Current Savings: $200,000 in 401(k)
- Target Retirement Age: 50
- Goals: Achieve financial independence, maintain current lifestyle in retirement
- Concerns: Having enough savings, healthcare costs before Medicare
- Knowledge Level: Moderate understanding of retirement planning

You're motivated to retire early but want to ensure your plan is realistic. You ask detailed questions about savings rates, investment strategies, and potential lifestyle adjustments."""
    },
    "estate": {
        "name": "Estate Planning Client",
        "profile": """You are a client seeking estate planning advice. Your details are:
- Age: 60
- Net Worth: $2.5 million
- Family: Married with 3 adult children
- Assets: Mix of real estate, investments, and business interests
- Goals: Efficient wealth transfer, minimizing tax impact
- Concerns: Fair distribution among children, tax implications
- Knowledge Level: Limited understanding of estate planning

You want to ensure your wealth is distributed according to your wishes while minimizing taxes and potential family conflicts. You ask questions about different trust structures and tax implications."""
    }
}
app = Flask(__name__)
CORS(app)

#hello 

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("No Google API key found. Please set GOOGLE_API_KEY in your .env file")
genai.configure(api_key=GOOGLE_API_KEY)


# Ensure temp directory exists
TEMP_DIR = Path(__file__).parent / "static" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def load_personas():
    try:
        with open('data/client_personas.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        app.logger.error("client_personas.json file not found.")
        return []
    except json.JSONDecodeError:
        app.logger.error("client_personas.json is not a valid JSON.")
        return []


# Load client personas at startup
personas = load_personas()

print("Loaded personas:")
for i, persona in enumerate(personas):
    print(f"Persona {i}:")
    for key, value in persona.items():
        print(f"  {key}: {value[:50]}..." if isinstance(value, str) else f"  {key}: {value}")

def clean_response(text):
    """Clean and preprocess the AI-generated text."""
    # Remove asterisks and other formatting characters
    text = re.sub(r'[*_~`]', '', text)
    # Replace various types of dashes with regular hyphens
    text = re.sub(r'[–—−]', '-', text)
    # Add spaces after punctuation if missing
    text = re.sub(r'([.,!?;:])(?=\S)', r'\1 ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_client_response(conversation_history, user_input, persona):
    app.logger.info("Entered get_client_response function")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        app.logger.info("Created Gemini model")
        
        chat_history = "\n".join([f"{'Financial Advisor' if msg['role'] == 'user' else 'Client'}: {msg['content']}" for msg in conversation_history[-5:]])
        
        # Check if 'profile' key exists, if not use 'description' or a default value
        persona_profile = persona.get('profile', persona.get('description', "No profile available"))
        
        prompt = f"""
        You are roleplaying as {persona.get('name', 'Unknown')}. Here is your full persona and background:

        {persona_profile}

        Recent conversation:
        {chat_history}

        Financial Advisor: {user_input}

        Respond as {persona.get('name', 'Unknown')}, keeping these guidelines in mind:
        1. Stay true to your persona, financial situation, goals, and concerns.
        2. Be direct and honest about your financial information when asked specific questions.
        3. Maintain a natural, conversational tone while showing your financial literacy level.
        4. Express your specific concerns and goals when relevant.
        5. Show that you're knowledgeable about finances to the extent of your financial literacy, but open to professional advice.
        6. If the question isn't about a specific financial detail, focus on your goals, concerns, or approach to retirement planning.
        7. Feel free to ask follow-up questions to gain more insights from the advisor.
        8. Do not repeat the financial advisor's question in your response.

        {persona.get('name', 'Unknown')}:
        """
        
        app.logger.info("Sending request to Gemini API")
        response = model.generate_content(prompt)
        app.logger.info("Received response from Gemini API")
        
        if response.parts:
            app.logger.info("Processing response from Gemini API")
            return clean_response(response.parts[0].text)
        else:
            app.logger.error("Empty response from Gemini API")
            return "I'm sorry, I couldn't generate a response at this time."
    except Exception as e:
        app.logger.error(f"Error in get_client_response: {str(e)}", exc_info=True)
        raise

def get_mentor_advice(conversation_history):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    chat_history = "\n".join([f"{'Financial Advisor' if msg['role'] == 'user' else 'Client'}: {msg['content']}" for msg in conversation_history[-10:]])
    
    prompt = f"""
As an experienced financial advisor mentor, provide brief, actionable advice to the financial advisor based on their recent conversation with the client. Focus on key areas for improvement or exploration.

Recent conversation:
{chat_history}

Provide 3 concise points of advice, each no more than 2 sentences. Format your response as follows:

1. [Key Area]: [Brief advice]
2. [Key Area]: [Brief advice]
3. [Key Area]: [Brief advice]

Mentor Agent:
"""

    response = model.generate_content(prompt)

    if response.parts:
        advice = clean_response(response.parts[0].text)
        # Remove "Mentor Agent:" prefix if present
        advice = re.sub(r'^Mentor Agent:\s*', '', advice)
        # Ensure each point is on a new line
        advice = re.sub(r'(\d+\.)', r'\n\1', advice)
        return advice.strip()
    else:
        return """
1. Client Understanding: Dig deeper into the client's goals and concerns.
2. Comprehensive Planning: Address all aspects of the client's financial situation.
3. Clear Communication: Explain strategies in simple, understandable terms.
"""


def get_evaluator_feedback(conversation_history):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    chat_history = "\n".join([f"{'Financial Advisor' if msg['role'] == 'user' else 'Client'}: {msg['content']}" for msg in conversation_history])
    
    prompt = f"""
As an expert financial advisor evaluator, provide a concise evaluation of the advisor's performance based on their conversation with the client. Focus on key areas of strength and improvement.

Conversation:
{chat_history}

Provide a brief evaluation in the following format:

# Financial Advisor Evaluation

## Strengths:
1. [Key Strength]: [Brief explanation]
2. [Key Strength]: [Brief explanation]

## Areas for Improvement:
1. [Key Area]: [Brief suggestion]
2. [Key Area]: [Brief suggestion]

## Overall Rating: [X/10]

## Summary:
[2-3 sentence overall assessment]
"""

    response = model.generate_content(prompt)

    if response.parts:
        feedback = clean_response(response.parts[0].text)
        # Remove any "Evaluator Feedback:" or "Evaluator Agent:" prefixes
        feedback = re.sub(r'^(Evaluator Feedback:|Evaluator Agent:)\s*', '', feedback, flags=re.IGNORECASE)
        # Ensure each section starts on a new line with proper Markdown formatting
        feedback = re.sub(r'(# Financial Advisor Evaluation|## Strengths:|## Areas for Improvement:|## Overall Rating:|## Summary:)', r'\n\1\n', feedback)
        # Add extra newline after each list item for better readability
        feedback = re.sub(r'(\d\. .+)$', r'\1\n', feedback, flags=re.MULTILINE)
        return feedback.strip()
    else:
        return """
# Financial Advisor Evaluation

## Strengths:
1. None identified in this interaction.

## Areas for Improvement:
1. Professionalism: Develop a respectful and client-focused approach.

2. Financial Knowledge: Enhance understanding of financial planning principles and ethical practices.

## Overall Rating: 1/10

## Summary:
The advisor's performance was severely lacking in professionalism and financial expertise. Significant improvements are needed in client interaction, ethical conduct, and financial knowledge before they can effectively serve clients.
"""

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@app.route('/submit_questionnaire', methods=['POST'])
def submit_questionnaire():
    # Process and store the questionnaire data
    # For now, we'll just redirect to the client persona page
    return redirect(url_for('client_personas'))

@app.route('/client_personas')
def client_personas():
    return render_template('client_personas.html', personas=personas)

@app.route('/chat', methods=['POST'])
def chat():
    app.logger.info("Received chat request")
    data = request.get_json()
    if not data:
        app.logger.error("No input data provided")
        return jsonify({'error': 'No input data provided.'}), 400
    try:
        app.logger.info(f"Received data: {data}")
        persona_id = int(data.get('persona_id'))
        user_message = data.get('message', '').strip()
        conversation_history = data.get('conversation_history', [])
        app.logger.info(f"Persona ID: {persona_id}, User message: {user_message}")
        
        if persona_id < 0 or persona_id >= len(personas):
            app.logger.error(f"Invalid persona ID: {persona_id}")
            return jsonify({'error': 'Invalid persona ID.'}), 400
        if not user_message:
            app.logger.error("Empty user message")
            return jsonify({'error': 'Message cannot be empty.'}), 400
        
        persona = personas[persona_id]
        app.logger.info(f"Selected persona: {persona.get('name', 'Unknown')}")
        
        app.logger.info("Calling get_client_response")
        try:
            response = get_client_response(conversation_history, user_message, persona)
            app.logger.info(f"Received response from get_client_response: {response}")
        except Exception as e:
            app.logger.error(f"Error in get_client_response: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error generating response: {str(e)}'}), 500
        
        if response is None:
            app.logger.error("get_client_response returned None")
            return jsonify({'error': 'Failed to generate response.'}), 500
        
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/mentor_advice', methods=['POST'])
def mentor_advice():
    app.logger.info("Received mentor advice request")
    data = request.get_json()
    if not data:
        app.logger.error("No input data provided")
        return jsonify({'error': 'No input data provided.'}), 400
    try:
        conversation_history = data.get('conversation_history', [])
        
        app.logger.info("Calling get_mentor_advice")
        advice = get_mentor_advice(conversation_history)
        
        app.logger.info(f"Received mentor advice: {advice}")
        return jsonify({'advice': advice})
    except Exception as e:
        app.logger.error(f"Error processing mentor advice request: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

@app.route('/evaluator_feedback', methods=['POST'])
def evaluator_feedback():
    app.logger.info("Received evaluator feedback request")
    data = request.get_json()
    if not data:
        app.logger.error("No input data provided")
        return jsonify({'error': 'No input data provided.'}), 400
    try:
        conversation_history = data.get('conversation_history', [])
        
        app.logger.info("Calling get_evaluator_feedback")
        feedback = get_evaluator_feedback(conversation_history)
        
        app.logger.info(f"Received evaluator feedback: {feedback}")
        return jsonify({'feedback': feedback})
    except Exception as e:
        app.logger.error(f"Error processing evaluator feedback request: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

@app.route('/scenario_challenge')
def scenario_challenge():
    """Render the scenario challenge page."""
    scenarios = [
        {
            "id": 1,
            "category": "investment",
            "title": "Portfolio Diversification",
            "description": "Help a client understand and implement portfolio diversification strategies for long-term growth and risk management."
        },
        {
            "id": 2,
            "category": "retirement",
            "title": "Early Retirement Planning",
            "description": "Guide a client through early retirement planning considerations and create a comprehensive strategy for financial independence."
        },
        {
            "id": 3,
            "category": "estate",
            "title": "Inheritance Planning",
            "description": "Assist a client with complex inheritance and wealth transfer decisions to ensure efficient asset distribution."
        },
        {
            "id": 4,
            "category": "tax",
            "title": "Tax-Efficient Investing",
            "description": "Help a high-income client optimize their investment strategy for tax efficiency while maintaining growth potential."
        },
        {
            "id": 5,
            "category": "education",
            "title": "College Savings Planning",
            "description": "Guide parents in creating a comprehensive college savings strategy while balancing other financial goals."
        },
        {
            "id": 6,
            "category": "insurance",
            "title": "Risk Management Strategy",
            "description": "Develop a comprehensive insurance and risk management plan for a client with complex personal and business needs."
        },
        {
            "id": 7,
            "category": "business",
            "title": "Business Succession Planning",
            "description": "Assist a business owner in developing a succession plan that addresses both business continuity and personal retirement needs."
        },
        {
            "id": 8,
            "category": "debt",
            "title": "Debt Management",
            "description": "Create a strategy for a client struggling with multiple debts while building a foundation for long-term financial success."
        },
        {
            "id": 9,
            "category": "realestate",
            "title": "Real Estate Investment",
            "description": "Guide a client through real estate investment opportunities and strategies as part of their overall financial plan."
        }
    ]
    return render_template('scenario_challenge.html', scenarios=scenarios)

# Add corresponding personas for the new scenarios
scenario_personas.update({
    "tax": {
        "name": "Tax Planning Client",
        "profile": """You are a client seeking tax-efficient investment advice. Your details are:
- Age: 42
- Income: $450,000/year (Executive position)
- Current Portfolio: $1.2M (mix of stocks, bonds, and mutual funds)
- Tax Bracket: 37%
- Goals: Minimize tax impact while growing wealth
- Concerns: High tax liability, alternative minimum tax implications
- Knowledge Level: Moderate understanding of tax strategies

You're focused on optimizing your investment strategy for tax efficiency and are interested in exploring various tax-advantaged investment options."""
    },
    "education": {
        "name": "Education Planning Client",
        "profile": """You are a client planning for children's education. Your details are:
- Age: 38
- Income: $180,000/year (combined household)
- Current Savings: $50,000 in 529 plans
- Children: Two (ages 8 and 5)
- Goals: Fund private college education for both children
- Concerns: Rising college costs, balancing education saving with retirement
- Knowledge Level: Basic understanding of education savings options

You want to ensure your children's education is fully funded while maintaining your retirement savings goals."""
    },
    "insurance": {
        "name": "Insurance Planning Client",
        "profile": """You are a client seeking comprehensive risk management advice. Your details are:
- Age: 45
- Income: $220,000/year
- Family: Married with 3 children
- Current Coverage: Basic life insurance through employer
- Goals: Protect family and assets
- Concerns: Adequate coverage levels, types of insurance needed
- Knowledge Level: Limited understanding of insurance products

You're looking to create a comprehensive insurance strategy to protect your family and assets."""
    },
    "business": {
        "name": "Business Owner Client",
        "profile": """You are a business owner planning for succession. Your details are:
- Age: 58
- Business Value: $5M (manufacturing company)
- Personal Income: $400,000/year
- Retirement Timeline: 5-7 years
- Goals: Smooth business transition, secure retirement
- Concerns: Business valuation, tax implications, family dynamics
- Knowledge Level: Experienced in business, less familiar with succession planning

You want to ensure a smooth transition of your business while maximizing your retirement benefits."""
    },
    "debt": {
        "name": "Debt Management Client",
        "profile": """You are a client seeking debt management advice. Your details are:
- Age: 32
- Income: $85,000/year
- Total Debt: $120,000 (student loans, credit cards, car loan)
- Credit Score: 680
- Goals: Debt freedom in 5 years while building savings
- Concerns: High interest rates, impact on future goals
- Knowledge Level: Basic financial understanding

You're motivated to become debt-free while building a strong financial foundation."""
    },
    "realestate": {
        "name": "Real Estate Investment Client",
        "profile": """You are a client interested in real estate investing. Your details are:
- Age: 40
- Income: $250,000/year
- Investment Capital: $300,000 available
- Current Portfolio: One rental property
- Goals: Build real estate portfolio for passive income
- Concerns: Market timing, financing options, management requirements
- Knowledge Level: Moderate real estate experience

You're looking to expand your real estate investments as part of your wealth-building strategy."""
    }
})

def get_scenario_response(message: str, history: List[Dict], scenario_category: str) -> str:
    """Generate a response based on the scenario category and conversation history."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Get the appropriate persona for the scenario
        persona = scenario_personas.get(scenario_category, scenario_personas["investment"])
        
        # Convert history to a string format
        chat_history = "\n".join([f"{'Advisor' if msg['role'] == 'user' else 'Client'}: {msg['content']}" 
                                for msg in history[-5:]])  # Only use last 5 messages for context
        
        prompt = f"""
        You are roleplaying as a client seeking financial advice. Here is your persona:

        {persona['profile']}

        Recent conversation:
        {chat_history}

        Financial Advisor: {message}

        Respond as the client, keeping these guidelines in mind:
        1. Stay in character according to your financial situation and knowledge level
        2. Ask relevant follow-up questions about the advisor's suggestions
        3. Express your concerns and goals naturally
        4. Be realistic in your responses
        5. Keep responses concise but meaningful
        
        Client:
        """
        
        response = model.generate_content(prompt)
        
        if response.parts:
            cleaned_response = clean_response(response.parts[0].text)
            return cleaned_response
        else:
            app.logger.error("Empty response from Gemini API")
            return "I need some time to think about your advice. Could you explain that in a different way?"
            
    except Exception as e:
        app.logger.error(f"Error in get_scenario_response: {str(e)}", exc_info=True)
        return "I apologize, but I'm having trouble processing your advice right now. Could we revisit this point?"

@app.route('/scenario_chat', methods=['POST'])
def scenario_chat():
    """Handle chat messages for scenario challenges."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400
            
        message = data.get('message', '').strip()
        scenario_id = data.get('scenario_id')
        conversation_history = data.get('conversation_history', [])
        
        if not message:
            return jsonify({'error': 'Message cannot be empty.'}), 400
            
        # Map scenario ID to category
        scenario_categories = {
            1: "investment",
            2: "retirement",
            3: "estate"
        }
        scenario_category = scenario_categories.get(scenario_id, "investment")
        
        # Generate response
        response = get_scenario_response(message, conversation_history, scenario_category)
        
        return jsonify({'response': response})
        
    except Exception as e:
        app.logger.error(f"Error in scenario_chat: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

@app.route('/dashboard')
def dashboard():
    # Placeholder for dashboard route
    return "Dashboard Page (To be implemented)"

def generate_prompt(brief_description):
    """Generate the prompt for the AI model."""
    return f"""
Based on the following brief description, generate a comprehensive and realistic financial persona for retirement planning purposes. The response must strictly follow the structured format below, with each attribute on a new line and no additional text or explanations.

Brief Description: {brief_description}

Generate the persona profile in exactly this format:

Name: [Full Name]
Age: [Age between 25-70]
Occupation: [Specific job title]
Marital Status: [Single/Married/Divorced/Widowed]
Dependents: [Number and ages if applicable]
Location: [City, State]
Annual Income: [Dollar amount]
Current Savings: [Dollar amount]
Total Debt: [Dollar amount]
Risk Tolerance: [Low/Medium/High]
Investment Portfolio:
- Stocks: [Percentage]
- Bonds: [Percentage]
- Cash: [Percentage]
- Other Investments: [Percentage]
Retirement Savings Plans:
- 401(k): [Yes/No]
  - Balance: [Dollar amount]
  - Annual Contribution: [Dollar amount]
- IRA: [Yes/No]
  - Type: [Traditional/Roth]
  - Balance: [Dollar amount]
  - Annual Contribution: [Dollar amount]
Financial Products of Interest: [List products]
Desired Retirement Age: [Age]
Retirement Lifestyle: [Brief description of desired lifestyle]
Financial Knowledge: [Beginner/Intermediate/Advanced]
Personality: [Key traits relevant to financial planning]
Financial Concerns: [List main concerns]

Generate a realistic persona based on the description provided. Ensure all amounts and percentages are realistic and consistent with the occupation and location provided.
"""

def parse_persona_text(text):
    """Parse the generated text into a structured format."""
    try:
        # Remove any leading/trailing whitespace and split into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Initialize persona dictionary
        persona = {}
        
        # Extract basic fields
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Skip lines that are sub-bullets
                if line.startswith('  -'):
                    continue
                    
                if key and value:
                    persona[key] = value

        # Create a concise description for the card display
        description = f"{persona.get('Age', 'N/A')}-year-old {persona.get('Occupation', 'Professional')} "
        description += f"earning ${persona.get('Annual Income', 'N/A')} annually. "
        description += f"Goals: Retire at {persona.get('Desired Retirement Age', 'N/A')}, "
        description += f"with {persona.get('Risk Tolerance', 'medium').lower()} risk tolerance. "
        
        return {
            "name": persona.get('Name', 'Custom Persona'),
            "description": description,
            "full_description": text,
            "profile": text  # For compatibility with existing personas format
        }
    except Exception as e:
        logging.error(f"Error parsing persona text: {str(e)}")
        raise ValueError("Failed to parse generated persona text")

def validate_persona(persona):
    """Validate the generated persona has required fields."""
    required_fields = ['name', 'description', 'full_description', 'profile']
    missing_fields = [field for field in required_fields if field not in persona]
    
    if missing_fields:
        raise ValueError(f"Generated persona missing required fields: {', '.join(missing_fields)}")
    
    return True

@app.route('/generate_persona', methods=['POST'])
def generate_persona():
    """Generate a new persona based on user description."""
    try:
        # Get and validate input
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({'error': 'No description provided'}), 400
            
        brief_description = data['description'].strip()
        if not brief_description:
            return jsonify({'error': 'Empty description provided'}), 400
            
        app.logger.info(f"Generating persona for description: {brief_description}")
        
        # Generate the prompt
        prompt = generate_prompt(brief_description)
        
        # Initialize Gemini model and generate response
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            if not response.parts or not response.parts[0].text:
                raise ValueError("Empty response from Gemini API")
                
            generated_text = response.parts[0].text
            
            # Parse and validate the generated persona
            new_persona = parse_persona_text(generated_text)
            validate_persona(new_persona)
            
            # Add default image
            new_persona['image'] = 'default-avatar.jpg'
            
            # Add to global personas list
            personas.append(new_persona)
            new_index = len(personas) - 1
            
            app.logger.info(f"Successfully generated new persona: {new_persona['name']}")
            
            return jsonify({
                'success': True,
                'persona': new_persona,
                'index': new_index
            })
            
        except Exception as e:
            app.logger.error(f"Error generating persona with Gemini: {str(e)}")
            return jsonify({'error': 'Failed to generate persona'}), 500
            
    except ValueError as ve:
        app.logger.error(f"Validation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
        
    except Exception as e:
        app.logger.error(f"Unexpected error in generate_persona: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred'}), 500
    

@app.route('/tts', methods=['POST'])
def tts():
    try:
        app.logger.info("TTS endpoint called")
        data = request.get_json()
        
        if not data or 'text' not in data:
            app.logger.error("No text provided for TTS")
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        voice = data.get('voice', 'alloy')  # Default to 'alloy' if not specified
        
        app.logger.info(f"Generating TTS for text: {text[:50]}... with voice: {voice}")

        # Create speech file path
        speech_file_path = TEMP_DIR / "speech.mp3"

        try:
            # Create speech using OpenAI
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            # Stream the response to a file
            response.stream_to_file(str(speech_file_path))

            # Read and encode the audio file
            with open(speech_file_path, 'rb') as audio_file:
                audio_content = audio_file.read()
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')

            # Clean up
            if speech_file_path.exists():
                speech_file_path.unlink()

            app.logger.info("TTS generated successfully")
            return jsonify({'audio': audio_base64})

        except Exception as e:
            app.logger.error(f"OpenAI TTS generation error: {str(e)}")
            return jsonify({'error': f'Failed to generate speech: {str(e)}'}), 500

    except Exception as e:
        app.logger.error(f"TTS endpoint error: {str(e)}")
        return jsonify({'error': f'TTS endpoint error: {str(e)}'}), 500
    
    
    
if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode)