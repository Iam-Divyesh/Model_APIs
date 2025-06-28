from flask import Flask, request, jsonify, render_template
import requests
import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
from flask_swagger_ui import get_swaggerui_blueprint

# Load environment variables
load_dotenv()

app = Flask(__name__)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'  # Make sure to place swagger.json in the static folder

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Recruiter AI API"}
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Initialize OpenAI client
try:
    client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
except Exception as e:
    print(f"Failed to initialize Azure OpenAI client: {e}")
    client = None

# Define major Indian cities
MAJOR_INDIAN_CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Surat", "Ahmedabad"]

def enhance_recruiter_query(query):
    """Enhance and improve recruiter queries using Azure OpenAI"""
    if not client:
        return {"error": "Azure OpenAI client not available", "enhanced_query": query}

    try:
        system_prompt = """You are a professional prompt enhancer for recruitment queries. Your task is to rewrite incomplete or shorthand recruiter queries into complete, structured, and professional recruiter-style sentences.

        Guidelines:
        1. Preserve the original meaning and requirements completely
        2. Make the query more specific and detailed
        3. Add professional recruiting language
        4. Ensure job title, skills, experience, and location (if mentioned) are clearly stated
        5. Don't add requirements that weren't in the original query
        6. Keep the enhanced query concise but complete
        7. Use proper grammar and professional tone

        Examples:
        - "python dev 3 yrs mumbai" → "Looking for a Python Developer with 3 years of experience based in Mumbai"
        - "need react guy" → "We are looking for a React Developer with relevant experience"
        - "data scientist ML 2-4 years bangalore" → "Seeking a Data Scientist with Machine Learning expertise and 2-4 years of experience in Bangalore"
        - "senior frontend typescript" → "Looking for a Senior Frontend Developer with TypeScript skills"

        Return ONLY the enhanced query without any explanation or additional text."""

        user_prompt = f"""Enhance this recruiter query: "{query}"

        Enhanced query:"""

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=200
        )

        enhanced_query = response.choices[0].message.content.strip()

        # Remove any quotes if the model added them
        if enhanced_query.startswith('"') and enhanced_query.endswith('"'):
            enhanced_query = enhanced_query[1:-1]

        return {
            "enhanced_query": enhanced_query,
            "original_query": query,
            "enhancement_status": "success"
        }

    except Exception as e:
        return {
            "error": f"Enhancement error: {e}",
            "enhanced_query": query,  # Fallback to original
            "original_query": query,
            "enhancement_status": "failed"
        }

def parse_recruiter_query(query):
    if not client:
        return {"error": "Azure OpenAI client not available"}

    try:
        system_prompt = """You are an expert recruitment assistant that extracts structured information from recruiter queries.

        Extract the following fields from the recruiter's input and return ONLY a valid JSON object:

        Fields to extract:
        - job_title: ONLY the exact position title they're hiring for (e.g., "Python Developer", "Data Scientist"). 
          DO NOT include phrases like "looking for", "need a", "hiring", etc.
        - skills: Array of required technical skills mentioned (e.g., ["Python", "Django", "SQL"])
        - experience: Required experience in years (numeric value or range)
        - location: Array of city names if multiple cities are mentioned, or single city name as string if only one city is mentioned.
        - work_preference: Work mode preference - one of: "remote", "onsite", "hybrid", null
        - job_type: Employment type - one of: "full-time", "part-time", "contract", "internship", null

        CRITICAL INSTRUCTIONS:
        1. For job_title, NEVER include phrases like "looking for", "need", "hiring", etc.
        2. Return ONLY valid JSON without any explanation or additional text.
        3. Use your knowledge to recognize job titles across all industries and domains."""

        user_prompt = f"""Extract recruitment information from this query: "{query}"

        Examples of correct extraction:

        Input: "We are looking for a Python developer with 3 years experience from Mumbai"
        Output: {{"job_title": "Python Developer", "skills": ["Python"], "experience": "3", "location": "Mumbai", "work_preference": null, "job_type": null}}

        Input: "Need a senior React frontend developer with Redux, TypeScript, 5+ years"
        Output: {{"job_title": "React Frontend Developer", "skills": ["React", "Redux", "TypeScript"], "experience": "5+", "location": null, "work_preference": null, "job_type": null}}

        Input: "python developer with 2 year of experience from surat, ahmedabad and mumbai"
        Output: {{"job_title": "Python Developer", "skills": ["Python"], "experience": "2", "location": ["Surat", "Ahmedabad", "Mumbai"], "work_preference": null, "job_type": null}}

        Now extract from the query: "{query}"

        Remember: 
        1. Extract ONLY the job title without any prefixes like "looking for", "need", etc.
        2. Extract ONLY the city/location name without additional text.
        3. Return ONLY valid JSON."""

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Clean up JSON if needed
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]

        content = content.strip()

        # Parse JSON
        parsed = json.loads(content)

        # Minimal post-processing to ensure clean data
        cleaned_result = {
            "job_title": parsed.get("job_title", "").strip() if parsed.get("job_title") else None,
            "skills": [skill.strip() for skill in parsed.get("skills", []) if skill.strip()],
            "experience": str(parsed.get("experience", "")).strip() if parsed.get("experience") else None,
            "location": parsed.get("location") if parsed.get("location") else None,
            "work_preference": parsed.get("work_preference"),
            "job_type": parsed.get("job_type"),
            "parsing_status": "success"
        }

        return cleaned_result

    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing error: {e}"}
    except Exception as e:
        return {"error": f"AI parsing error: {e}"}

def fetch_linkedin_profiles(parsed_data, page_number=1, results_per_page=10):
    job_title = parsed_data.get("job_title", "")
    location = parsed_data.get("location", "")
    skills = parsed_data.get("skills", [])
    experience = parsed_data.get("experience")
    work_preference = parsed_data.get("work_preference")

    # Build search query specifically for LinkedIn profiles
    search_query = "site:linkedin.com/in "

    if job_title:
        search_query += f'"{job_title}" '

    # Handle location
    if location:
        if isinstance(location, list) and len(location) > 0:
            for city in location:
                if city and city.strip():
                    search_query += f'"{city.strip()}" '
        elif isinstance(location, str) and location.strip():
            search_query += f'"{location.strip()}" '
        else:
            for city in MAJOR_INDIAN_CITIES[:3]:
                search_query += f'"{city}" '
    else:
        for city in MAJOR_INDIAN_CITIES[:3]:
            search_query += f'"{city}" '

    # Add top 3 skills
    if skills and len(skills) > 0:
        for skill in skills[:3]:
            search_query += f'"{skill}" '

    # Add experience if available
    if experience:
        search_query += f'"{experience} years" '

    # Add work preference if specified
    if work_preference:
        search_query += f'"{work_preference}" '

    # Exclude job posting keywords
    search_query += '-"job" -"jobs" -"hiring" -"vacancy" -"openings" -"career" -"apply"'

    # Calculate start parameter for pagination
    start_index = (page_number - 1) * results_per_page

    # SerpAPI parameters with pagination
    params = {
        "engine": "google",
        "q": search_query.strip(),
        "api_key": os.getenv("SERP_API_KEY"),
        "hl": "en",
        "gl": "in",
        "google_domain": "google.co.in",
        "location": "India",
        "num": results_per_page,
        "start": start_index,
        "safe": "active"
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code == 200:
            data = response.json()
            all_results = data.get("organic_results", [])

            # Filter to only include LinkedIn profile URLs and exclude job postings
            filtered_results = []
            for result in all_results:
                link = result.get("link", "")
                title = result.get("title", "").lower()
                snippet = result.get("snippet", "").lower()

                # Check if it's a LinkedIn profile URL
                if ("linkedin.com/in/" in link or "in.linkedin.com/in/" in link):
                    # Exclude job posting keywords in title and snippet
                    job_keywords = ["job", "jobs", "hiring", "vacancy", "openings", "career", "apply", "position"]

                    # Check if any job keywords are in title or snippet
                    has_job_keywords = any(keyword in title or keyword in snippet for keyword in job_keywords)

                    if not has_job_keywords:
                        # Fix LinkedIn URLs
                        if "in.linkedin.com" in link:
                            result["link"] = link.replace("in.linkedin.com", "linkedin.com")

                        filtered_results.append(result)

            return filtered_results
        else:
            return []
    except Exception as e:
        return []

def score_profile(profile, parsed_data):
    content = (profile.get("title", "") + " " + profile.get("snippet", "")).lower()
    score = 0

    # Score based on job title match (30 points max)
    if parsed_data.get("job_title"):
        job_title = parsed_data["job_title"].lower()
        if job_title in content:
            score += 30
        title_words = job_title.split()
        for word in title_words:
            if len(word) > 2 and word in content:
                score += 10

    # Score based on skills match (15 points per skill, max 45 points)
    if parsed_data.get("skills"):
        for skill in parsed_data["skills"]:
            if skill.lower() in content:
                score += 15

    # Score based on location match (20 points)
    if parsed_data.get("location"):
        if isinstance(parsed_data["location"], list):
            for city in parsed_data["location"]:
                if city.lower() in content:
                    score += 20
                    break
        elif parsed_data["location"].lower() in content:
            score += 20

    # Score based on experience match (15 points)
    if parsed_data.get("experience"):
        exp = str(parsed_data["experience"])
        exp_clean = exp.replace("+", "")
        exp_range = exp.replace("-", r"\s*-\s*")

        exp_patterns = [
            exp_clean + r'\s*(?:\+)?\s*(?:years?|yrs?)',
            exp_range + r'\s*(?:years?|yrs?)'
        ]

        for pattern in exp_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 15
                break

    return min(score, 100)  # Cap the score at 100

def get_match_category(score):
    if score >= 75:
        return "Excellent Match"
    elif score >= 50:
        return "Good Match"
    elif score >= 30:
        return "Fair Match"
    else:
        return "Basic Match"

def extract_experience_from_snippet(snippet):
    """Extract experience information from profile snippet"""
    if not snippet:
        return None

    snippet_lower = snippet.lower()

    # Enhanced experience patterns - comprehensive list from main.py
    experience_patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp|work)',
        r'(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\+\s*(?:years?|yrs?)',
        r'over\s*(\d+)\s*(?:years?|yrs?)',
        r'more\s*than\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\s*(?:years?|yrs?)\s*in',
        r'(\d+)\s*(?:years?|yrs?)\s*as',
        r'(\d+)\s*(?:years?|yrs?)\s*with',
        r'experienced\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\s*(?:years?|yrs?)\s*professional',
        r'(\d+)\s*(?:year|yr)\s*(?:experience|exp)',
        r'(\d+)\s*(?:year|yr)\s*in',
        r'(\d+)\s*(?:years?|yrs?)\s*at',
        r'(\d+)\s*(?:years?|yrs?)\s*working',
        r'(\d+)\s*(?:years?|yrs?)\s*background',
        r'(\d+)\s*(?:years?|yrs?)\s*expertise',
        r'(\d+)\s*(?:years?|yrs?)\s*specializing',
        r'(\d+)\s*(?:years?|yrs?)\s*specialised',
        r'(\d+)\+\s*(?:years?|yrs?)\s*experience',
        r'(\d+)\s*to\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)[\s\-]*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\s*(?:years?|yrs?)',  # Simple pattern as fallback
    ]

    for pattern in experience_patterns:
        matches = re.findall(pattern, snippet_lower)
        if matches:
            match = matches[0]
            if isinstance(match, tuple) and len(match) == 2:
                # Range pattern like "3-5 years" or "3 to 5 years"
                try:
                    year1, year2 = int(match[0]), int(match[1])
                    if 0 <= year1 <= 30 and 0 <= year2 <= 30:  # Reasonable range
                        return f"{year1}-{year2} years"
                except ValueError:
                    continue
            else:
                # Single number pattern
                try:
                    years = int(match) if isinstance(match, str) else int(match[0])
                    if 0 <= years <= 30:  # Reasonable experience range
                        return f"{years} years" if years > 0 else "Fresher"
                except ValueError:
                    continue

    # Check for fresher/entry level terms
    fresher_terms = [
        'fresher', 'entry level', 'graduate', 'new grad', 'recent graduate',
        'just graduated', 'starting career', 'entry-level', 'beginner',
        'trainee', 'intern', 'junior', 'associate'
    ]

    for term in fresher_terms:
        if term in snippet_lower:
            return "Fresher/Entry Level"

    return None

def generate_candidate_message(candidate_name, parsed_data, candidate_experience=None, candidate_location=None):
    """Generate a personalized message for each candidate"""
    job_title = parsed_data.get("job_title", "a professional")
    experience = parsed_data.get("experience", "")
    location = parsed_data.get("location", "")
    skills = parsed_data.get("skills", [])

    # Clean up the candidate name
    name = str(candidate_name).replace("Professional Profile", "").strip()
    if not name or name == "LinkedIn Profile" or name == "":
        name = "there"

    # Start building the message
    message = f"Hi {name},\n\n"
    message += f"I hope this message finds you well. We are looking for a {job_title}"

    # Add experience requirement
    if experience:
        message += f" with {experience} years of experience"

    # Add location if specified
    if location:
        if isinstance(location, list):
            location_str = ", ".join(location)
            message += f" in {location_str}"
        else:
            message += f" in {location}"

    message += "."

    # Add skills requirement if any
    if skills:
        if len(skills) == 1:
            message += f"\n\nKey skill we're looking for: {skills[0]}"
        elif len(skills) <= 3:
            skills_str = ", ".join(skills[:-1]) + f" and {skills[-1]}"
            message += f"\n\nKey skills we're looking for: {skills_str}"
        else:
            skills_str = ", ".join(skills[:3])
            message += f"\n\nKey skills we're looking for: {skills_str} and more"

    # Add candidate-specific details if available
    if candidate_experience and candidate_experience != "Not specified":
        message += f"\n\nWe noticed you have {candidate_experience} of experience"
        if candidate_location and candidate_location != "Not specified":
            message += f" and you're based in {candidate_location}"
        message += ", which aligns well with our requirements."
    elif candidate_location and candidate_location != "Not specified":
        message += f"\n\nWe noticed you're based in {candidate_location}, which fits our location preference."

    message += "\n\nWe would love to discuss this opportunity with you. Please let us know if you're interested in learning more about this position."
    message += "\n\nBest regards,\nRecruitment Team"

    return message

def extract_location_from_snippet(snippet, search_locations):
    """Extract location information from profile snippet"""
    if not snippet:
        return None

    snippet_lower = snippet.lower()

    # Check for searched locations first (exact match)
    if search_locations:
        locations_to_check = []
        if isinstance(search_locations, list):
            locations_to_check = search_locations
        else:
            locations_to_check = [search_locations]

        for location in locations_to_check:
            if location and location.lower() in snippet_lower:
                return location

    # Extended comprehensive list of Indian cities from main.py
    extended_cities = MAJOR_INDIAN_CITIES + [
        "Gurgaon", "Gurugram", "Noida", "Kochi", "Cochin", "Kolkata", "Indore", 
        "Jaipur", "Chandigarh", "Coimbatore", "Vadodara", "Mysore", "Nagpur",
        "Visakhapatnam", "Bhubaneswar", "Lucknow", "Kanpur", "Patna", "Goa",
        "Trivandrum", "Thiruvananthapuram", "Madurai", "Nashik", "Rajkot",
        "Faridabad", "Ghaziabad", "Agra", "Ludhiana", "Kanpur", "Varanasi",
        "Meerut", "Rajkot", "Kalyan-Dombivali", "Vasai-Virar", "Amritsar",
        "Allahabad", "Howrah", "Ranchi", "Jabalpur", "Gwalior", "Vijayawada",
        "Jodhpur", "Madurai", "Raipur", "Kota", "Loni", "Siliguri", "Jhansi",
        "Ulhasnagar", "Jammu", "Sangli-Miraj & Kupwad", "Mangalore", "Erode",
        "Belgaum", "Ambattur", "Tirunelveli", "Malegaon", "Gaya", "Jalgaon",
        "Udaipur", "Maheshtala", "Tirupur", "Davanagere", "Kozhikode", "Akola",
        "Kurnool", "Bokaro", "Rajahmundry", "Ballari", "Tirupati", "Bhilai",
        "Patiala", "Bidhannagar", "Panipat", "Durgapur", "Asansol", "Nanded",
        "Kolhapur", "Ajmer", "Gulbarga", "Jamnagar", "Ujjain", "Loni", "Siliguri",
        "Jhansi", "Ulhasnagar", "Nellore", "Jammu", "Sangli-Miraj & Kupwad",
        "Islampur", "Kadapa", "Cuttack", "Firozabad", "Kochi", "Bhavnagar",
        "Dehradun", "Durgapur", "Asansol", "Rourkela", "Nanded", "Kolhapur",
        "Ajmer", "Akola", "Gulbarga", "Jamnagar", "Ujjain", "Loni", "Siliguri"
    ]

    # Check for cities in snippet (exact match first)
    for city in extended_cities:
        if city.lower() in snippet_lower:
            return city

    # Enhanced location patterns from main.py
    location_patterns = [
        r'(?:based in|located in|from|working in|living in|residing in|situated in)\s*([a-zA-Z\s,]{2,30})(?:\s|,|\.|\||$)',
        r'([a-zA-Z\s]{2,25}),?\s*india',
        r'([a-zA-Z\s]{2,20})\s*(?:area|region|zone|district)',
        r'([a-zA-Z\s]{2,20})\s*(?:metropolitan|metro|city)',
        r'([a-zA-Z\s]{2,20})\s*(?:state|province)',
        r'(?:at|in)\s*([a-zA-Z\s]{2,25})(?:\s*,|\s*-|\s*\||\s*\.)',
        r'([a-zA-Z\s]{3,20})\s*(?:office|location|branch|center|centre)',
        r'(?:currently in|working at|employed in|job in)\s*([a-zA-Z\s]{2,25})(?:\s|,|\.|\||$)',
        r'([a-zA-Z\s]{2,25})\s*(?:based|located)',
        r'(?:lives in|living in)\s*([a-zA-Z\s]{2,25})(?:\s|,|\.|\||$)'
    ]

    for pattern in location_patterns:
        matches = re.findall(pattern, snippet_lower, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else (match[1] if len(match) > 1 else "")

                location = match.strip().title()

                # Enhanced filtering of non-location words
                non_locations = [
                    'linkedin', 'profile', 'experience', 'years', 'company', 'limited', 
                    'pvt', 'private', 'technologies', 'solutions', 'services', 'systems',
                    'software', 'developer', 'engineer', 'manager', 'specialist', 'consultant',
                    'analyst', 'architect', 'lead', 'senior', 'junior', 'associate',
                    'programming', 'development', 'technical', 'professional', 'expert',
                    'skilled', 'experienced', 'passionate', 'dedicated', 'motivated',
                    'seeking', 'looking', 'available', 'open', 'interested', 'explore',
                    'opportunities', 'career', 'growth', 'learning', 'knowledge',
                    'industry', 'domain', 'field', 'sector', 'business', 'corporate',
                    'team', 'projects', 'work', 'working', 'job', 'role', 'position'
                ]

                # Additional checks for valid location
                if (3 <= len(location) <= 25 and 
                    not any(word in location.lower() for word in non_locations) and
                    not location.lower().startswith(('the ', 'and ', 'or ', 'with ', 'for ')) and
                    not location.isnumeric() and
                    ' ' in location or location in extended_cities):

                    # Check if it's a real Indian city (even if not in our list)
                    words = location.split()
                    if len(words) <= 3:  # Reasonable city name length
                        return location

    return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse', methods=['POST'])
def parse_prompt():
    data = request.get_json()
    query = data.get('query', '')

    if not query.strip():
        return jsonify({"error": "Query is required"}), 400

    parsed_data = parse_recruiter_query(query)
    return jsonify(parsed_data)

# Endpoint 1: Query Enhancement
@app.route('/enhance', methods=['POST'])
def enhance_query():
    data = request.get_json()
    query = data.get('query', '')

    if not query.strip():
        return jsonify({"error": "Query is required"}), 400

    # Enhance the query
    enhancement_result = enhance_recruiter_query(query)
    
    response = {
        "original_query": query,
        "enhanced_query": enhancement_result.get('enhanced_query', query),
        "enhancement_status": enhancement_result.get('enhancement_status', 'completed')
    }

    return jsonify(response)


# Endpoint 2: Candidate Search
@app.route('/search', methods=['POST'])
def search_candidates():
    data = request.get_json()
    query = data.get('query', '')
    page = data.get('page', 1)

    if not query.strip():
        return jsonify({"error": "Query is required"}), 400

    # Directly parse the provided query (enhanced or not)
    parsed_data = parse_recruiter_query(query)

    if "error" in parsed_data:
        return jsonify(parsed_data), 400

    # Fetch candidates
    results = fetch_linkedin_profiles(parsed_data, page)

    # Score and sort profiles
    candidates = []
    for result in results:
        score = score_profile(result, parsed_data)

        # Extract name from title
        title = result.get('title', 'LinkedIn Profile')
        name_parts = title.split(' - ')[0].split(' | ')[0].split(' at ')[0]
        if len(name_parts) > 50:
            name = "Professional Profile"
        else:
            name = name_parts.strip()

        # Extract experience from snippet
        snippet = result.get('snippet', '')
        experience = extract_experience_from_snippet(snippet)

        # Extract location from snippet
        location = extract_location_from_snippet(snippet, parsed_data.get('location'))

        # Debug logging
        print(f"DEBUG - Candidate: {name}")
        print(f"DEBUG - Experience extracted: {experience}")
        print(f"DEBUG - Location extracted: {location}")
        print(f"DEBUG - Snippet: {snippet[:100]}...")
        print("---")

        # Generate personalized message
        try:
            personalized_message = generate_candidate_message(
                name,
                parsed_data,
                experience,
                location
            )
            print(f"DEBUG - Generated message for {name}: {personalized_message[:50]}...")
        except Exception as e:
            print(f"ERROR - Failed to generate message for {name}: {e}")
            personalized_message = f"Hi {name},\n\nWe are interested in discussing opportunities with you.\n\nBest regards,\nRecruitment Team"

        candidate = {
            "name": name,
            "profile_link": result.get('link', '#'),
            "description": result.get('snippet', 'No description available'),
            "score": score,
            "match_category": get_match_category(score),
            "experience": experience if experience else "Not specified",
            "location": location if location else "Not specified",
            "personalized_message": personalized_message
        }
        candidates.append(candidate)

    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    response = {
        "current_page": page,
        "candidates": candidates,
        "parsed_data": parsed_data
    }

    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)